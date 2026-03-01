import os
import re
import sys
import json
import time
import math
import logging

import numpy as np
import pandas as pd
import statistics
import torch
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra

from transformers import AutoModel, AutoTokenizer
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()

from entity_parser.np_parser_backt import NP_Parser_BackT
import ent_attn_func.attn_flow as ent_attn_flow
from utils.text_utils import filter_sentence
from utils.stats import get_avg_from_lists
from corpus.load_toefl import load_dataset_toefl

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def wrapper_ent_attn_flow(cfg, encoder, tokenizer, sent_corpus, num_sents_all, top_k=5, target_layer=-1):
    '''
        encoder: Huggingface model
        tokenizer: Huggingface tokenizer
        sent_corpus: pandas dataframe
        target_layer: target attention layer
    '''


    np_parser = NP_Parser_BackT(tokenizer=tokenizer, encoder_weights=cfg.model.llm_id)
    list_doc_np_sbw_loc = []
    list_doc_vp_sbw_loc = []
    cache_entity = False
    list_val_ent_topk = []
    list_val_vp_topk = []

    list_val_interact_type1 = []  # type 1: between entities and others
    list_val_interact_type2 = []  # type 2: between entities
    list_val_interact_type3 = []  # type 3: between non-entities
    list_val_interact_type4 = []

    num_sents_non_ent = 0
    num_sents_corrupted = 0  # which is not parsed perfectly somehow
    num_sents_no_vp = 0  # which does not include VP or wrongly parsed
    num_sents_corrupted_vp = 0
    num_sents_short = 0  # which is not considered due to short length
    num_sents_considered = 0

    list_len_subwords = []  # the number of subwords for each sentence    
    list_len_subwords_ent = []  # the number of subwords of all entities for each sentence    
    list_len_subwords_vp = []  # the number of subwords of all verbs for each sentence    

    list_num_entities = []  # the number of entities for each sentence

    for ind_doc, cur_doc in enumerate(sent_corpus):

        doc_val_ent_topk = []
        doc_val_vp_topk = []

        doc_val_interact_type1 = []  # type 1: between entities and others
        doc_val_interact_type2 = []  # type 2: between entities
        doc_val_interact_type3 = []  # type 3: between non-entities
        doc_val_interact_type4 = []

        if cache_entity:    
            doc_np_sbw_loc = list_doc_np_sbw_loc[ind_doc]
            doc_vp_sbw_loc = []
        else:
            doc_np_sbw_loc = []
            doc_vp_sbw_loc = []

        for sent_ind, cur_sent in enumerate(cur_doc):
            if cache_entity:
                list_tags_sbw_loc = doc_np_sbw_loc[sent_ind]
            else:
                list_tags_sbw_loc = []
            cur_sent = filter_sentence(cur_sent)
            if len(cur_sent.split()) < 5:
                num_sents_short += 1
                continue

            output_attn_flow = ent_attn_flow.run_attn_flow(
                cfg, tokenizer, encoder, np_parser, cur_sent, list_tags_sbw_loc,
                top_k=cfg.exp_args.top_k, target_layer=target_layer, cache_entity=cache_entity
            )
            is_parsed_all = output_attn_flow["is_parsed_all"] 
            is_vp_parsed_all = output_attn_flow["is_vp_parsed_all"]

            if is_parsed_all and is_vp_parsed_all:

                ent_top_k = output_attn_flow["ent_top_k"]
                vp_top_k = output_attn_flow["vp_top_k"]

                aflow_type1 = output_attn_flow["val_type1"]
                aflow_type2 = output_attn_flow["val_type2"]
                aflow_type3 = output_attn_flow["val_type3"]
                aflow_type4 = output_attn_flow["val_type4"]
                list_np_sbw_loc = output_attn_flow["list_np_sbw_loc"]
                list_vp_sbw_loc = output_attn_flow["list_vp_sbw_loc"]
                seq_len = output_attn_flow["seq_len_subwords"]
                seq_len_ent = output_attn_flow["len_subwords_ent"]
                seq_len_vp = output_attn_flow["len_subwords_vp"]
                if not math.isnan(ent_top_k):
                    doc_val_ent_topk.append(ent_top_k)
                if not math.isnan(vp_top_k):
                    doc_val_vp_topk.append(vp_top_k)
                if not math.isnan(aflow_type1):
                    doc_val_interact_type1.append(aflow_type1)
                if not math.isnan(aflow_type2):
                    doc_val_interact_type2.append(aflow_type2)
                if not math.isnan(aflow_type3):
                    doc_val_interact_type3.append(aflow_type3)
                if not math.isnan(aflow_type4):
                    doc_val_interact_type4.append(aflow_type4)

                doc_np_sbw_loc.append(list_np_sbw_loc)
                doc_vp_sbw_loc.append(list_vp_sbw_loc)

                list_num_entities.append(len(list_np_sbw_loc))
                list_len_subwords.append(seq_len)
                list_len_subwords_ent.append(seq_len_ent)
                list_len_subwords_vp.append(seq_len_vp)

                num_sents_considered += 1

            else:
                num_sents_corrupted += 1
            list_np_sbw_loc = output_attn_flow.get("list_np_sbw_loc", [])
            list_vp_sbw_loc = output_attn_flow.get("list_vp_sbw_loc", [])
            if len(list_np_sbw_loc) < 1:
                num_sents_non_ent += 1
            if len(list_vp_sbw_loc) < 1:
                num_sents_no_vp += 1

        if len(doc_val_ent_topk)>0: list_val_ent_topk.append(doc_val_ent_topk)
        if len(doc_val_vp_topk)>0: list_val_vp_topk.append(doc_val_vp_topk)

        if len(doc_val_interact_type1)>0:   list_val_interact_type1.append(doc_val_interact_type1)
        if len(doc_val_interact_type2)>0:   list_val_interact_type2.append(doc_val_interact_type2)
        if len(doc_val_interact_type3)>0:   list_val_interact_type3.append(doc_val_interact_type3)
        if len(doc_val_interact_type4)>0:   list_val_interact_type4.append(doc_val_interact_type4)

        list_doc_np_sbw_loc.append(doc_np_sbw_loc)
        list_doc_vp_sbw_loc.append(doc_vp_sbw_loc)
                
        if cfg.exp_args.sample_num != -1 and ind_doc > cfg.exp_args.sample_num:
            break

        if ind_doc % 500 == 0:
            logger.info(f"Document: {ind_doc}")
    
    avg_seq_len = statistics.mean(list_len_subwords)
    logger.info(f"Avg # of subwords in a sentence: {avg_seq_len}")
    avg_seq_len_ent = statistics.mean(list_len_subwords_ent)
    logger.info(f"Avg # of all subwords for entities in a sentence: {avg_seq_len_ent}")
    avg_seq_len_vp = statistics.mean(list_len_subwords_vp)
    logger.info(f"Avg # of all subwords for verbs in a sentence: {avg_seq_len_vp}")
    avg_num_ent = statistics.mean(list_num_entities)
    logger.info(f"Avg # of entities in a sentence: {avg_num_ent}")
    logger.info("")
    logger.info(f"# of sentences without entities: {num_sents_non_ent} ({num_sents_non_ent/float(num_sents_all):.2f}%)")
    ratio_corrupted = num_sents_corrupted * 100 / float(num_sents_all)
    logger.info(f"# of sentences not properly parsed: {num_sents_corrupted} ({ratio_corrupted:.2f}%)")
    ratio_short = num_sents_short * 100 / float(num_sents_all)
    logger.info(f"# of sentences not considered due to short: {num_sents_short} ({ratio_short:.2f}%)")
    ratio_considered = num_sents_considered * 100 / float(num_sents_all)
    logger.info(f"# of sentences considered in total: {num_sents_considered} ({ratio_considered:.2f}%)")  # in terms of entity
    logger.info("")

    logger.info(f"# of sentences without VP: {num_sents_no_vp} ({num_sents_no_vp/float(num_sents_all):.2f}%")
    ratio_corrupted_vp = num_sents_corrupted_vp * 100 / float(num_sents_all)
    logger.info(f"# of sentences not properly VP parsed: {num_sents_corrupted_vp} ({ratio_corrupted_vp:.2f})")
    logger.info("")

    ## ent@K
    outputs = get_avg_from_lists(list_val=list_val_ent_topk)
    avg_sents_entk, std_sents_entk = float(outputs[0]), float(outputs[1])
    logger.info(f"Ent@K Avg(Std): {avg_sents_entk} ({std_sents_entk})" )

    outputs = get_avg_from_lists(list_val=list_val_vp_topk)
    avg_sents_vpk, std_sents_vpk = float(outputs[0]), float(outputs[1])
    logger.info(f"VP@K Avg(Std): {avg_sents_vpk} ({std_sents_vpk})" )

    logger.info("")

    ##

    outputs = get_avg_from_lists(list_val=list_val_interact_type1)
    avg_sents_type1, std_sents_type1 = float(outputs[0]), float(outputs[1])
    logger.info(f"Type1 Avg(Std): {avg_sents_type1} ({std_sents_type1})" )

    outputs = get_avg_from_lists(list_val=list_val_interact_type2)
    avg_sents_type2, std_sents_type2 = float(outputs[0]), float(outputs[1])
    logger.info(f"Type2 Avg(Std): {avg_sents_type2} ({std_sents_type2})" )

    outputs = get_avg_from_lists(list_val=list_val_interact_type3)
    avg_sents_type3, std_sents_type3 = float(outputs[0]), float(outputs[1])
    logger.info(f"Type3 Avg(Std): {avg_sents_type3} ({std_sents_type3})" )

    outputs = get_avg_from_lists(list_val=list_val_interact_type4)
    avg_sents_type4, std_sents_type4 = float(outputs[0]), float(outputs[1])
    logger.info(f"Type4 Avg(Std): {avg_sents_type4} ({std_sents_type4})" )

    logger.info("")

    outputs = []
    outputs.append(list_val_interact_type1)
    outputs.append(list_val_interact_type2)
    outputs.append(list_val_interact_type3)
    outputs.append(list_val_interact_type4)

    outputs.append((avg_sents_type1, std_sents_type1))
    outputs.append((avg_sents_type2, std_sents_type2))
    outputs.append((avg_sents_type3, std_sents_type3))
    outputs.append((avg_sents_type4, std_sents_type4))

    return outputs


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    hf_token = os.environ.get("HF_TOKEN") or OmegaConf.select(cfg, "hf_token") or None

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm_id, low_cpu_mem_usage=True, token=hf_token)
    encoder_dtype = torch.bfloat16
    if "xlnet" in cfg.model.llm_id or "google-bert" in cfg.model.llm_id:
        encoder_dtype = "auto"
    encoder = AutoModel.from_pretrained(
        cfg.model.llm_id,
        low_cpu_mem_usage=True,
        token=hf_token,
        offload_folder="offload",
        offload_state_dict=True,
        max_memory={"cpu": "10GIB"},
        torch_dtype=encoder_dtype,
    ).to("cuda")
    for param in encoder.parameters():
        param.requires_grad = False

    logger.info(f"Target LLM: {cfg.model.llm_id}")

    sent_corpus, num_sents = load_dataset_toefl(
        cfg.dataset.path_data,
        num_samples=0,
        filter_key="essay_score",
        filter_value=2,
    )
    num_docs = len(sent_corpus)
    num_sents_all = sum(num_sents)
    if cfg.exp_args.sample_num != -1:
        num_docs = cfg.exp_args.sample_num
        num_sents_all = sum(num_sents[:cfg.exp_args.sample_num])

    logger.info(f"Total # of Documents: {num_docs}")
    logger.info(f"Total # of Sentences: {num_sents_all}")

    output_ent_attn_flow = wrapper_ent_attn_flow(
        cfg, encoder, tokenizer, sent_corpus, num_sents_all, cfg.exp_args.top_k, target_layer=-1
    )
    list_val_interact_type1 = output_ent_attn_flow[0]
    list_val_interact_type2 = output_ent_attn_flow[1]
    list_val_interact_type3 = output_ent_attn_flow[2]

    output_dir = "log_entity_attn"
    sub_dir = cfg.model.llm_id
    sub_dir = sub_dir.replace("/", "-")
    Path(os.path.join(output_dir, sub_dir)).mkdir(parents=True, exist_ok=True)

    output_name = "type1.log"
    with open(os.path.join(output_dir, sub_dir, output_name), 'w') as f:
        json.dump(list_val_interact_type1, f)
    output_name = "type2.log"
    with open(os.path.join(output_dir, sub_dir, output_name), 'w') as f:
        json.dump(list_val_interact_type2, f)
    output_name = "type3.log"
    with open(os.path.join(output_dir, sub_dir, output_name), "w") as f:
        json.dump(list_val_interact_type3, f)
    logger.info("")


if __name__ == "__main__":
    start = time.time()
    main()
    logging.info(f"Processing time: {(time.time() - start) / 60:.2f} min")
