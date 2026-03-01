"""
Entity-centric attention flow and ranking (top-k) over NP/VP spans.
Used for evaluating how much attention links entities vs. others.
"""
import torch

from . import filter_special_tokens as filter_sp


def flat_index_to_list(ind_ent):
    """Flatten (start, end) index pairs into a list of indices."""
    list_flat_ind_ent = []
    for cur_ind_pair in ind_ent:
        ind_start, ind_end = cur_ind_pair[0], cur_ind_pair[1]
        list_flat_ind_ent.extend(list(range(ind_start, ind_end + 1)))
    return list_flat_ind_ent


def attn_ranking_entity(model_weights, attn_weights, output_np_parser, top_k):
    """
    Rank attention by strength and compute entity/VP proportions in top-k.
    Returns ent_top_k, vp_top_k, and type1--type4 (inner entity, entity-VP, entity-other, VP-other).
    """
    list_np_sbw_loc = output_np_parser["np_sbw_loc"]
    list_vp_sbw_loc = output_np_parser["vp_sbw_loc"]

    batch_size, len_seq = attn_weights.size(0), attn_weights.size(1)
    attn_diag = torch.tril(attn_weights, diagonal=-1)
    denorm = torch.sum(attn_diag)
    attn_weights_norm = attn_diag / denorm

    ind_ent = [curr[1] for curr in list_np_sbw_loc]
    ind_vp = [curr[1] for curr in list_vp_sbw_loc]
    list_all_ind_ent = flat_index_to_list(ind_ent)
    list_all_ind_vp = flat_index_to_list(ind_vp)
    list_all_ind_others = list(set(range(len_seq)).difference(list_all_ind_ent))

    map_rank_attn = {}
    for i in range(len_seq):
        for j in range(0, i):
            map_rank_attn[(j, i)] = float(attn_weights_norm[0][i][j])
    sorted_map_rank_attn = dict(sorted(map_rank_attn.items(), key=lambda item: item[1], reverse=True))

    ent_k_cnt = vp_k_cnt = ind = 0
    for key, val in sorted_map_rank_attn.items():
        src, dst = key[0], key[1]
        if src in list_all_ind_ent or dst in list_all_ind_ent:
            ent_k_cnt += 1
        if src in list_all_ind_vp or dst in list_all_ind_vp:
            vp_k_cnt += 1
        ind += 1
        if ind >= top_k:
            break

    ent_top_k = ent_k_cnt / float(top_k)
    vp_top_k = vp_k_cnt / float(top_k)

    type1_cnt = type2_cnt = type3_cnt = type4_cnt = 0
    ind = 0
    for key, val in sorted_map_rank_attn.items():
        src, dst = key[0], key[1]
        if src in list_all_ind_ent and dst in list_all_ind_ent:
            type1_cnt += 1
        elif (src in list_all_ind_ent and dst in list_all_ind_vp) or (src in list_all_ind_vp and dst in list_all_ind_ent):
            type2_cnt += 1
        elif src in list_all_ind_ent or dst in list_all_ind_ent:
            type3_cnt += 1
        elif (src in list_all_ind_ent and dst not in list_all_ind_vp and dst not in list_all_ind_ent) or (
            src not in list_all_ind_ent and src not in list_all_ind_vp and dst in list_all_ind_vp
        ):
            type4_cnt += 1
        ind += 1
        if ind >= top_k:
            break

    outputs_attn_rank = {
        "ent_top_k": ent_top_k,
        "vp_top_k": vp_top_k,
        "val_type1": type1_cnt / float(top_k),
        "val_type2": type2_cnt / float(top_k),
        "val_type3": type3_cnt / float(top_k),
        "val_type4": type4_cnt / float(top_k),
        "len_subwords_ent": len(list_all_ind_ent),
        "len_subwords_vp": len(list_all_ind_vp),
    }
    return outputs_attn_rank


def attn_flow_entity(model_weights, attn_weights, list_np_sbw_loc):
    """
    Sum-based attention flow between entity–other, entity–entity, other–other, inner-entity (type1–4).
    Alternative to attn_ranking_entity; kept for legacy/experimental metrics.
    """
    batch_size, len_seq = attn_weights.size(0), attn_weights.size(1)
    attn_diag = torch.tril(attn_weights, diagonal=-1)
    denorm = torch.sum(attn_diag)
    attn_weights_norm = attn_diag / denorm

    ind_ent = [curr[1] for curr in list_np_sbw_loc]
    list_all_ind_ent = flat_index_to_list(ind_ent)
    list_all_ind_others = list(set(range(len_seq)).difference(list_all_ind_ent))

    inner_ent_masked_attn = attn_weights_norm.clone()
    for cur_ind_pair in ind_ent:
        ind_start, ind_end = cur_ind_pair[0], cur_ind_pair[1]
        list_ind_ent = list(range(ind_start, ind_end + 1))
        for i in range(len(list_ind_ent) - 1):
            for j in range(i + 1, len(list_ind_ent)):
                ind_a, ind_b = list_ind_ent[i], list_ind_ent[j]
                inner_ent_masked_attn[:, ind_a, ind_b] = 0.0
                inner_ent_masked_attn[:, ind_b, ind_a] = 0.0

    device = attn_weights_norm.device
    mask_type1 = torch.ones(batch_size, len_seq, len_seq, device=device)
    for i in range(len(list_all_ind_ent) - 1):
        for j in range(i + 1, len(list_all_ind_ent)):
            ind_a, ind_b = list_all_ind_ent[i], list_all_ind_ent[j]
            mask_type1[:, ind_a, ind_b] = 0.0
            mask_type1[:, ind_b, ind_a] = 0.0
    mask_type1 = mask_type1.bool()

    mask_type2 = torch.zeros(batch_size, len_seq, len_seq, device=device)
    for i in range(len(list_all_ind_ent) - 1):
        for j in range(i + 1, len(list_all_ind_ent)):
            ind_a, ind_b = list_all_ind_ent[i], list_all_ind_ent[j]
            mask_type2[:, ind_a, ind_b] = 1.0
            mask_type2[:, ind_b, ind_a] = 1.0
    mask_type2 = mask_type2.bool()

    mask_type3 = torch.zeros(batch_size, len_seq, len_seq, device=device)
    for i in range(len(list_all_ind_others) - 1):
        for j in range(i + 1, len(list_all_ind_others)):
            ind_a, ind_b = list_all_ind_others[i], list_all_ind_others[j]
            mask_type3[:, ind_a, ind_b] = 1.0
            mask_type3[:, ind_b, ind_a] = 1.0
    mask_type3 = mask_type3.bool()

    mask_type4 = torch.zeros(batch_size, len_seq, len_seq, device=device)
    for cur_ind_pair in ind_ent:
        ind_start, ind_end = cur_ind_pair[0], cur_ind_pair[1]
        list_ind_ent = list(range(ind_start, ind_end + 1))
        for i in range(len(list_ind_ent) - 1):
            for j in range(i + 1, len(list_ind_ent)):
                ind_a, ind_b = list_ind_ent[i], list_ind_ent[j]
                mask_type4[:, ind_a, ind_b] = 1.0
                mask_type4[:, ind_b, ind_a] = 1.0
    mask_type4 = mask_type4.bool()

    def _sum_masked(attn, mask):
        vals = torch.masked_select(attn, mask)
        nz = vals[torch.nonzero(vals)]
        return float(torch.sum(nz, dim=0))

    return {
        "val_type1": _sum_masked(inner_ent_masked_attn, mask_type1),
        "val_type2": _sum_masked(inner_ent_masked_attn, mask_type2),
        "val_type3": _sum_masked(inner_ent_masked_attn, mask_type3),
        "val_type4": _sum_masked(inner_ent_masked_attn, mask_type4),
    }


def run_attn_flow(cfg, tokenizer, encoder, np_parser, cur_sent, list_tags_sbw_loc, top_k=5, target_layer=-1, cache_entity=False):
    """Run encoder, get attention, parse NPs/VPs, and return ranking metrics."""
    tokenized = tokenizer(cur_sent, return_tensors="pt")
    tokenized.to("cuda")
    cur_sent_ids = tokenized["input_ids"]
    cur_attn_mask = tokenized["attention_mask"]
    seq_len = cur_sent_ids.shape[1]

    encoder_out = encoder(cur_sent_ids, cur_attn_mask, output_attentions=True, output_hidden_states=True)
    attn_layers = encoder_out["attentions"]
    attn_layers = filter_sp.filter_attn_special_tokens(cfg.model.llm_id, attn_layers)

    attn_target_layer = attn_layers[target_layer]
    attn = torch.div(torch.sum(attn_target_layer, dim=1), attn_target_layer.shape[1])

    is_parsed_all = is_vp_parsed_all = False
    if not cache_entity:
        output_np_parser = np_parser.get_np_index_subwords(cur_sent, is_add_vp=True)
        if output_np_parser is not None:
            list_np_sbw_loc = output_np_parser["np_sbw_loc"]
            list_vp_sbw_loc = output_np_parser["vp_sbw_loc"]
            is_parsed_all = output_np_parser["is_parsed_all"]
            is_vp_parsed_all = output_np_parser["is_vp_parsed_all"]

    outputs = {"is_parsed_all": is_parsed_all, "is_vp_parsed_all": is_vp_parsed_all}
    if is_parsed_all and is_vp_parsed_all:
        outputs_vals = attn_ranking_entity(cfg.model.llm_id, attn, output_np_parser, top_k=top_k)
        outputs["ent_top_k"] = outputs_vals["ent_top_k"]
        outputs["vp_top_k"] = outputs_vals["vp_top_k"]
        outputs["val_type1"] = outputs_vals["val_type1"]
        outputs["val_type2"] = outputs_vals["val_type2"]
        outputs["val_type3"] = outputs_vals["val_type3"]
        outputs["val_type4"] = outputs_vals["val_type4"]
        outputs["list_np_sbw_loc"] = list_np_sbw_loc
        outputs["list_vp_sbw_loc"] = list_vp_sbw_loc
        outputs["seq_len_subwords"] = seq_len
        outputs["len_subwords_ent"] = outputs_vals["len_subwords_ent"]
        outputs["len_subwords_vp"] = outputs_vals["len_subwords_vp"]
    return outputs
