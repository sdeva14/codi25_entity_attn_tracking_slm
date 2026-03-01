"""
Entity attention flow evaluation on TOEFL-style data (list of documents, each doc = list of sentences).
Run from project root. Requires local CSV folds; set DATA_PATH or corpus/config_toefl.yaml.
"""
import logging
import os
import time

import torch
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra

from transformers import AutoModel, AutoTokenizer
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()

from entity_parser.np_parser_backt import NP_Parser_BackT
import ent_attn_func.attn_flow as ent_attn_flow
from ent_attn_func.attn_flow_runner import (
    MIN_WORDS_SENTENCE,
    LOG_DIR,
    AttnFlowAggregator,
    write_logs,
)
from utils.text_utils import filter_sentence
from corpus.load_toefl import load_dataset_toefl

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run_entity_attn_flow_docs(
    cfg,
    encoder,
    tokenizer,
    sent_corpus,
    num_sents_all: int,
    top_k: int = 5,
    target_layer: int = -1,
):
    """
    Run entity attention flow over a doc-level corpus (list of list of sentences).
    Returns results list: [list_type1, list_type2, list_type3, list_type4, (avg1,std1), ...].
    """
    np_parser = NP_Parser_BackT(tokenizer=tokenizer, encoder_weights=cfg.model.llm_id)
    cache_entity = False
    aggregator = AttnFlowAggregator.for_doc_corpus()

    for ind_doc, cur_doc in enumerate(sent_corpus):
        doc_np_sbw_loc = (
            list(aggregator.list_doc_np_sbw_loc[ind_doc])
            if cache_entity and ind_doc < len(aggregator.list_doc_np_sbw_loc)
            else []
        )

        for sent_ind, cur_sent in enumerate(cur_doc):
            list_tags_sbw_loc = (
                doc_np_sbw_loc[sent_ind] if cache_entity and sent_ind < len(doc_np_sbw_loc) else []
            )
            cur_sent = filter_sentence(cur_sent)
            if len(cur_sent.split()) < MIN_WORDS_SENTENCE:
                aggregator.num_sents_short += 1
                continue

            output_attn_flow = ent_attn_flow.run_attn_flow(
                cfg,
                tokenizer,
                encoder,
                np_parser,
                cur_sent,
                list_tags_sbw_loc,
                top_k=cfg.exp_args.top_k,
                target_layer=target_layer,
                cache_entity=cache_entity,
            )
            list_np_sbw_loc = output_attn_flow.get("list_np_sbw_loc", [])
            list_vp_sbw_loc = output_attn_flow.get("list_vp_sbw_loc", [])
            aggregator.update(output_attn_flow, list_np_sbw_loc, list_vp_sbw_loc)

        aggregator.end_document()
        if cfg.exp_args.sample_num != -1 and ind_doc > cfg.exp_args.sample_num:
            break
        if ind_doc % 500 == 0:
            logger.info(f"Document: {ind_doc}")

    aggregator.log_summary(num_sents_all, logger)
    return aggregator.build_results()


def load_encoder_and_tokenizer(cfg: DictConfig):
    """Load HuggingFace tokenizer and encoder (frozen) from config."""
    hf_token = os.environ.get("HF_TOKEN") or OmegaConf.select(cfg, "hf_token") or None
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.llm_id, low_cpu_mem_usage=True, token=hf_token
    )
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
    return tokenizer, encoder


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    tokenizer, encoder = load_encoder_and_tokenizer(cfg)
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

    results = run_entity_attn_flow_docs(
        cfg,
        encoder,
        tokenizer,
        sent_corpus,
        num_sents_all,
        top_k=cfg.exp_args.top_k,
        target_layer=-1,
    )
    sub_dir = cfg.model.llm_id.replace("/", "-")
    write_logs(LOG_DIR, sub_dir, results, logger)


if __name__ == "__main__":
    start = time.time()
    main()
    logging.info(f"Processing time: {(time.time() - start) / 60:.2f} min")
