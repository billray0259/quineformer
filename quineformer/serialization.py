import torch
from torch import Tensor


def vector_component_labels(config) -> list[str]:
    """Return a component-type label for each row of the serialized matrix.

    The label order exactly mirrors serialize(): global embeddings first,
    then per-layer attention/MLP/LayerNorm blocks.

    Args:
        config: BertConfig, BertModel, or BertForMaskedLM (config is extracted).

    Returns:
        List of strings, length N (same as serialize(model).shape[0]).
        Labels are: 'word_emb', 'pos_emb', 'type_emb', 'emb_ln_gamma',
        'emb_ln_beta', 'Q', 'K', 'V', 'O', 'b_O', 'attn_ln_gamma',
        'attn_ln_beta', 'mlp_up', 'mlp_down', 'b_2', 'mlp_ln_gamma',
        'mlp_ln_beta'.
    """
    from transformers import BertModel, BertForMaskedLM, BertConfig

    if isinstance(config, BertForMaskedLM):
        config = config.bert.config
    elif isinstance(config, BertModel):
        config = config.config
    elif not isinstance(config, BertConfig):
        raise TypeError(f"Expected BertConfig, BertModel, or BertForMaskedLM, got {type(config)}")

    d = config.hidden_size
    d_ff = config.intermediate_size
    vocab = config.vocab_size
    max_pos = config.max_position_embeddings
    n_type = config.type_vocab_size
    n_layers = config.num_hidden_layers

    labels: list[str] = []

    # ── Global embeddings ──────────────────────────────────────────
    labels.extend(['word_emb'] * vocab)
    labels.extend(['pos_emb'] * max_pos)
    labels.extend(['type_emb'] * n_type)
    labels.append('emb_ln_gamma')
    labels.append('emb_ln_beta')

    # ── Encoder layers ─────────────────────────────────────────────
    for _ in range(n_layers):
        labels.extend(['Q'] * d)
        labels.extend(['K'] * d)
        labels.extend(['V'] * d)
        labels.extend(['O'] * d)
        labels.append('b_O')
        labels.append('attn_ln_gamma')
        labels.append('attn_ln_beta')
        labels.extend(['mlp_up'] * d_ff)
        labels.extend(['mlp_down'] * d_ff)
        labels.append('b_2')
        labels.append('mlp_ln_gamma')
        labels.append('mlp_ln_beta')

    return labels


def serialize(model) -> Tensor:
    """Serialize a BERT model into an (N, d_model+1) matrix.

    Implements the d_model decomposition with bias absorption (README §1, §4.1):

    - Read-from-stream matrices (Q, K, V, W₁): rows of stored weight
    - Write-to-stream matrices (O, W₂): columns of stored weight
    - Q/K/V/W₁ vectors absorb their per-vector bias as the (d_model+1)-th dim
    - All other vectors (O, W₂, embeddings, LayerNorms, standalone biases)
      are zero-padded in the (d_model+1)-th dim

    Args:
        model: BertModel or BertForMaskedLM instance.

    Returns:
        (N, d_model+1) tensor.  For MultiBERT (BERT-base): N=141,702, d_model+1=769.
    """
    from transformers import BertModel, BertForMaskedLM

    if isinstance(model, BertForMaskedLM):
        bert = model.bert
    elif isinstance(model, BertModel):
        bert = model
    else:
        raise TypeError(f"Expected BertModel or BertForMaskedLM, got {type(model)}")

    d = bert.config.hidden_size
    vectors: list[Tensor] = []

    def pad_bias(w: Tensor) -> Tensor:
        """Append a zero column: (*, d) → (*, d+1)."""
        if w.dim() == 1:
            w = w.unsqueeze(0)
        return torch.cat([w, w.new_zeros(w.shape[0], 1)], dim=1)

    def concat_bias(w: Tensor, b: Tensor) -> Tensor:
        """Append per-row bias scalar: (N, d) + (N,) → (N, d+1)."""
        return torch.cat([w, b.unsqueeze(1)], dim=1)

    # ── Global embeddings ──────────────────────────────────────────
    vectors.append(pad_bias(bert.embeddings.word_embeddings.weight))
    vectors.append(pad_bias(bert.embeddings.position_embeddings.weight))
    vectors.append(pad_bias(bert.embeddings.token_type_embeddings.weight))
    vectors.append(pad_bias(bert.embeddings.LayerNorm.weight))  # γ
    vectors.append(pad_bias(bert.embeddings.LayerNorm.bias))    # β

    # ── Encoder layers ─────────────────────────────────────────────
    for layer in bert.encoder.layer:
        # layer is a BertLayer module
        sa = layer.attention.self
        ao = layer.attention.output

        # Q/K/V read from residual stream → rows, with absorbed bias
        vectors.append(concat_bias(sa.query.weight, sa.query.bias))
        vectors.append(concat_bias(sa.key.weight,   sa.key.bias))
        vectors.append(concat_bias(sa.value.weight, sa.value.bias))

        # O writes to residual stream → columns (= weight.T rows), zero-padded
        vectors.append(pad_bias(ao.dense.weight.t()))
        vectors.append(pad_bias(ao.dense.bias))          # b_O standalone

        vectors.append(pad_bias(ao.LayerNorm.weight))    # post-attn LN γ
        vectors.append(pad_bias(ao.LayerNorm.bias))      # post-attn LN β

        # MLP up (W₁) reads from stream → rows, with absorbed bias
        vectors.append(concat_bias(layer.intermediate.dense.weight,
                              layer.intermediate.dense.bias))

        # MLP down (W₂) writes to stream → columns, zero-padded
        vectors.append(pad_bias(layer.output.dense.weight.t()))
        vectors.append(pad_bias(layer.output.dense.bias))    # b₂ standalone

        vectors.append(pad_bias(layer.output.LayerNorm.weight))  # post-MLP LN γ
        vectors.append(pad_bias(layer.output.LayerNorm.bias))    # post-MLP LN β

    result = torch.cat(vectors, dim=0)

    # ── Sanity check ───────────────────────────────────────────────
    n_layers = bert.config.num_hidden_layers
    d_ff = bert.config.intermediate_size
    vocab = bert.embeddings.word_embeddings.weight.shape[0]
    max_pos = bert.embeddings.position_embeddings.weight.shape[0]
    n_type = bert.embeddings.token_type_embeddings.weight.shape[0]
    expected_n = (vocab + max_pos + n_type + 2
                  + n_layers * (d * 4 + d_ff * 2 + 6))
    assert result.shape == (expected_n, d + 1), \
        f"Shape mismatch: expected {(expected_n, d + 1)}, got {tuple(result.shape)}"

    return result


def deserialize(data: Tensor, config) -> dict[str, Tensor]:
    """Reconstruct BERT parameter tensors from an (N, d_model+1) matrix.

    Inverse of serialize(). Returns a dict of tensors derived from slicing
    data — all ops (slice, transpose, index) preserve the autograd graph,
    so gradients flow from any downstream loss back through data.

    Args:
        data:   (N, d_model+1) tensor produced by serialize().
        config: BertConfig, BertModel, or BertForMaskedLM (config is extracted).

    Returns:
        dict mapping BertModel state_dict keys to tensors.
        Use with torch.func.functional_call() for differentiable forward passes,
        or model.load_state_dict() for checkpoint loading.
    """
    from transformers import BertModel, BertForMaskedLM, BertConfig

    if isinstance(config, BertForMaskedLM):
        config = config.bert.config
    elif isinstance(config, BertModel):
        config = config.config
    elif not isinstance(config, BertConfig):
        raise TypeError(f"Expected BertConfig, BertModel, or BertForMaskedLM, got {type(config)}")

    d = config.hidden_size
    d_ff = config.intermediate_size
    vocab = config.vocab_size
    max_pos = config.max_position_embeddings
    n_type = config.type_vocab_size
    n_layers = config.num_hidden_layers

    params: dict[str, Tensor] = {}
    idx = 0

    def take(n: int) -> Tensor:
        nonlocal idx
        chunk = data[idx : idx + n]
        idx += n
        return chunk

    def split_bias(rows: Tensor) -> tuple[Tensor, Tensor]:
        """(N, d+1) → weight (N, d), bias (N,)."""
        return rows[:, :d], rows[:, d]

    def to_vec(rows: Tensor) -> Tensor:
        """(1, d+1) → (d,) dropping the zero-padded bias slot."""
        return rows[0, :d]

    # ── Global embeddings ──────────────────────────────────────────
    params['embeddings.word_embeddings.weight'] = take(vocab)[:, :d]
    params['embeddings.position_embeddings.weight'] = take(max_pos)[:, :d]
    params['embeddings.token_type_embeddings.weight'] = take(n_type)[:, :d]
    params['embeddings.LayerNorm.weight'] = to_vec(take(1))
    params['embeddings.LayerNorm.bias'] = to_vec(take(1))

    # ── Encoder layers ─────────────────────────────────────────────
    for i in range(n_layers):
        pre = f'encoder.layer.{i}'

        # Q/K/V: rows with absorbed bias
        w, b = split_bias(take(d))
        params[f'{pre}.attention.self.query.weight'] = w
        params[f'{pre}.attention.self.query.bias'] = b

        w, b = split_bias(take(d))
        params[f'{pre}.attention.self.key.weight'] = w
        params[f'{pre}.attention.self.key.bias'] = b

        w, b = split_bias(take(d))
        params[f'{pre}.attention.self.value.weight'] = w
        params[f'{pre}.attention.self.value.bias'] = b

        # O: columns (stored as transposed rows), zero-padded
        params[f'{pre}.attention.output.dense.weight'] = take(d)[:, :d].t()
        params[f'{pre}.attention.output.dense.bias'] = to_vec(take(1))

        params[f'{pre}.attention.output.LayerNorm.weight'] = to_vec(take(1))
        params[f'{pre}.attention.output.LayerNorm.bias'] = to_vec(take(1))

        # MLP up (W₁): rows with absorbed bias
        w, b = split_bias(take(d_ff))
        params[f'{pre}.intermediate.dense.weight'] = w
        params[f'{pre}.intermediate.dense.bias'] = b

        # MLP down (W₂): columns (stored as transposed rows), zero-padded
        params[f'{pre}.output.dense.weight'] = take(d_ff)[:, :d].t()
        params[f'{pre}.output.dense.bias'] = to_vec(take(1))

        params[f'{pre}.output.LayerNorm.weight'] = to_vec(take(1))
        params[f'{pre}.output.LayerNorm.bias'] = to_vec(take(1))

    assert idx == data.shape[0], \
        f"Consumed {idx} rows but data has {data.shape[0]}"

    return params


def encoder_layer_row_bounds(config, layer_idx: int) -> tuple[int, int]:
    """Return the serialized row bounds for one encoder layer."""
    from transformers import BertModel, BertForMaskedLM, BertConfig

    if isinstance(config, BertForMaskedLM):
        config = config.bert.config
    elif isinstance(config, BertModel):
        config = config.config
    elif not isinstance(config, BertConfig):
        raise TypeError(f"Expected BertConfig, BertModel, or BertForMaskedLM, got {type(config)}")

    if not 0 <= layer_idx < config.num_hidden_layers:
        raise IndexError(
            f"layer_idx={layer_idx} out of range for {config.num_hidden_layers} layers"
        )

    d = config.hidden_size
    d_ff = config.intermediate_size
    prefix = config.vocab_size + config.max_position_embeddings + config.type_vocab_size + 2
    rows_per_layer = 4 * d + 2 * d_ff + 6
    start = prefix + layer_idx * rows_per_layer
    return start, start + rows_per_layer


def deserialize_encoder_layer(data: Tensor, config) -> dict[str, Tensor]:
    """Reconstruct one BertLayer state dict from its serialized row slice.

    Args:
        data: (4*d + 2*d_ff + 6, d+1) tensor containing exactly one encoder
            layer's serialized rows.
        config: BertConfig, BertModel, or BertForMaskedLM.

    Returns:
        dict keyed like a BertLayer state_dict, e.g. `attention.self.query.weight`.
    """
    from transformers import BertModel, BertForMaskedLM, BertConfig

    if isinstance(config, BertForMaskedLM):
        config = config.bert.config
    elif isinstance(config, BertModel):
        config = config.config
    elif not isinstance(config, BertConfig):
        raise TypeError(f"Expected BertConfig, BertModel, or BertForMaskedLM, got {type(config)}")

    d = config.hidden_size
    d_ff = config.intermediate_size
    expected_rows = 4 * d + 2 * d_ff + 6
    if data.shape != (expected_rows, d + 1):
        raise ValueError(
            f"Expected layer data shape {(expected_rows, d + 1)}, got {tuple(data.shape)}"
        )

    params: dict[str, Tensor] = {}
    idx = 0

    def take(n: int) -> Tensor:
        nonlocal idx
        chunk = data[idx : idx + n]
        idx += n
        return chunk

    def split_bias(rows: Tensor) -> tuple[Tensor, Tensor]:
        return rows[:, :d], rows[:, d]

    def to_vec(rows: Tensor) -> Tensor:
        return rows[0, :d]

    w, b = split_bias(take(d))
    params["attention.self.query.weight"] = w
    params["attention.self.query.bias"] = b

    w, b = split_bias(take(d))
    params["attention.self.key.weight"] = w
    params["attention.self.key.bias"] = b

    w, b = split_bias(take(d))
    params["attention.self.value.weight"] = w
    params["attention.self.value.bias"] = b

    params["attention.output.dense.weight"] = take(d)[:, :d].t()
    params["attention.output.dense.bias"] = to_vec(take(1))
    params["attention.output.LayerNorm.weight"] = to_vec(take(1))
    params["attention.output.LayerNorm.bias"] = to_vec(take(1))

    w, b = split_bias(take(d_ff))
    params["intermediate.dense.weight"] = w
    params["intermediate.dense.bias"] = b

    params["output.dense.weight"] = take(d_ff)[:, :d].t()
    params["output.dense.bias"] = to_vec(take(1))
    params["output.LayerNorm.weight"] = to_vec(take(1))
    params["output.LayerNorm.bias"] = to_vec(take(1))

    assert idx == data.shape[0], f"Consumed {idx} rows but layer data has {data.shape[0]}"
    return params
