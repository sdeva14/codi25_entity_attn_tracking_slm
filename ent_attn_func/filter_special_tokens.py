def filter_attn_special_tokens(llm_id, attn_layers):
    """Remove special tokens (BOS/sep/etc.) from attention per model family."""
    filtered_layers = []
    if "meta-llama/Llama-3" in llm_id or "google/gemma-2" in llm_id:
        for cur_layer in attn_layers:  # (batch, mh, seq_len, seq_len)
            filtered_layers.append(cur_layer[:, :, 1:, 1:])
    
    elif "xlnet" in llm_id:
        for cur_layer in attn_layers:  # (batch, mh, seq_len, seq_len)
            filtered_layers.append(cur_layer[:, :, :-2, :-2])

    elif "google-bert/bert-" in llm_id:
        for cur_layer in attn_layers:  # (batch, mh, seq_len, seq_len)
            filtered_layers.append(cur_layer[:, :, 1:-1, 1:-1])
    else:
        return attn_layers
    
    return filtered_layers