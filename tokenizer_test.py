"""Dev script: test NP parser and tokenizer for a given model (Hydra config)."""
import os
import sys

import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from transformers import AutoModel, AutoTokenizer
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()

from entity_parser.np_parser_backt import NP_Parser_BackT
from utils.text_utils import filter_sentence


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    hf_token = os.environ.get("HF_TOKEN") or OmegaConf.select(cfg, "hf_token") or None
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm_id, low_cpu_mem_usage=True, token=hf_token)

    encoder = AutoModel.from_pretrained(
        cfg.model.llm_id,
        low_cpu_mem_usage=True,
        token=hf_token,
        offload_folder="offload",
        offload_state_dict=True,
        max_memory={"cpu": "10GIB"},
        torch_dtype=torch.bfloat16,
    )
    for param in encoder.parameters():
        param.requires_grad = False

    cur_sent = "'If a bird spreads its wings beyond its reach, it will crash'."
    cur_sent = filter_sentence(cur_sent)
    print(f"Preprocessed: {cur_sent}")

    ### tokenization as a whole sentence
    sent_ids = tokenizer(cur_sent, return_tensors="pt").input_ids
    print(sent_ids)
    decoded_word = tokenizer.batch_decode(sent_ids)
    print(decoded_word)

    i_ids = sent_ids[0].tolist()
    decoded_tokens = tokenizer.convert_ids_to_tokens(i_ids)
    print(decoded_tokens)

    # ascii
    for decoded in decoded_tokens:
        li = [ord(c) for c in decoded]
        print(decoded, li)

    print("------------")

    np_parser_backt = NP_Parser_BackT(tokenizer=tokenizer, encoder_weights=cfg.model.llm_id)
    output_np_parser = np_parser_backt.get_np_index_subwords(cur_sent)
    print("NP parser output:", output_np_parser)


if __name__ == "__main__":
    main()