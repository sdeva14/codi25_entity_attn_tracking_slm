import os
import sys
import time
import math
import logging

import numpy as np
import pandas as pd
import statistics
import torch
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import datasets
from datasets import load_dataset

from entity_parser.np_parser_backt import NP_Parser_BackT
import ent_attn_func.attn_flow as ent_attn_flow
from utils.text_utils import filter_sentence
from utils.sentiment import (
    SENTIMENT_PROMPT,
    convert_label_str,
    convert_label_list_int,
    clean_generated_sentiment_class,
    get_hist_preds_labels,
    fill_empty_label,
)
from utils.stats import get_avg_from_lists
from utils.llm_utils import get_response_delimiters, filter_generated_text
from utils.plot_utils import plot_preds_labels_dists

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def peft_encoder(llm_id, encoder, tokenizer, train_dataset, test_dataset, messages):
    train_dataset = train_dataset.map(convert_label_str)
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    encoder.gradient_checkpointing_enable()
    encoder = prepare_model_for_kbit_training(encoder)

    encoder = get_peft_model(
        encoder,
        config,
    )
    print_trainable_parameters(encoder)
    dataset_sft_train = convert_msgs_format(train_dataset)
    logger.info(dataset_sft_train)
    logger.info(dataset_sft_train[0])
    tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})

    from transformers import TrainingArguments, Trainer
    from trl import SFTTrainer

    fp16 = False
    bf16 = True

    output_train_dir = "hf_train_results_lora"
    sub_dir = llm_id
    sub_dir = sub_dir.replace("/", "-")
    output_path_train = Path(os.path.join(output_train_dir, sub_dir)).mkdir(parents=True, exist_ok=True)

    training_arguments = TrainingArguments(
        output_dir=os.path.join(output_train_dir, sub_dir),
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=1e-4,
        weight_decay=0.001,
        bf16=bf16,
        fp16=fp16,
        max_grad_norm=0.0,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )

    trainer = SFTTrainer(
        model=encoder,
        train_dataset=dataset_sft_train,
        peft_config=config,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_arguments,
        packing= False,
    )
    trainer.train()

    preds_peft, labels_peft = inference_sentiment(test_dataset, llm_id, encoder, tokenizer, messages)
    preds_peft = clean_generated_sentiment_class(preds_peft)
    preds_discrete_peft = convert_label_list_int(preds_peft)
    acc = eval_accuracy(preds_discrete_peft, labels_peft)
    kappa_el_linear, kappa_el_quad = eval_kappa(preds_discrete_peft, labels_peft)

    logger.info("Accuracy: %.2f" % (acc))
    logger.info("Kappa El: %.2f %.2f" % (kappa_el_linear, kappa_el_quad))
    logger.info("-------")
    logger.info("")
    hist_preds_peft, hist_labels_peft = get_hist_preds_labels(preds_discrete_peft, labels_peft)
    logger.info(hist_preds_peft)
    logger.info(hist_labels_peft)

    hist_preds_peft = fill_empty_label(hist_preds_peft)

    plot_preds_labels_dists(hist_preds_peft, hist_labels_peft, "lora", is_equal_label_dist=True)


    return encoder


def convert_msgs_format(curr_dataset):
    """Convert dataset to chat format for SFT trainer (user + assistant)."""
    import pandas as pd
    list_dict_msgs = []
    for curr in curr_dataset:
        curr_msg = [
            {"role": "user", "content": SENTIMENT_PROMPT + "\n\n" + curr["text"]},
            {"role": "assistant", "content": curr["label"]},
        ]
        list_dict_msgs.append({"messages": curr_msg})
    return datasets.Dataset.from_pandas(pd.DataFrame(data=list_dict_msgs))


def inference_sentiment(dataset_curr, llm_id, model, tokenizer, messages, max_new_tokens=5):
    preds = []
    labels = []
    pre_header, post_header = get_response_delimiters(llm_id)
    for i in range(len(dataset_curr)):
        cur_user_response = dataset_curr[i]["text"]
        label_curr = dataset_curr[i]["label"]
        messages[-1]["content"] = SENTIMENT_PROMPT + "\n\n" + cur_user_response
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
        text = tokenizer.batch_decode(outputs)[0]
        only_generated_text = filter_generated_text(text, pre_header, post_header)
        preds.append(only_generated_text)
        labels.append(label_curr)
    return preds, labels

def eval_accuracy(preds, labels):
    return sum(1 for x, y in zip(preds, labels) if x == y) / len(preds)


def eval_kappa(preds, labels, is_quad_also=True):
    from sklearn.metrics import cohen_kappa_score
    kappa_linear = cohen_kappa_score(preds, labels, weights="linear")
    kappa_quad = cohen_kappa_score(preds, labels, weights="quadratic") if is_quad_also else 0.0
    return kappa_linear, kappa_quad


def sample_dataset(curr_dataset, num_labels=5, sample_ratio=0.1, num_samples_label=40):
    sample_num = round(len(curr_dataset) * sample_ratio)
    list_indices = list(range(sample_num))
    dataset_sampled_ne_test = curr_dataset.select(list_indices)
    import pandas as pd
    df_dataset = pd.DataFrame(curr_dataset)
    list_pd_labels = []
    for i in range(num_labels):
        pd_label_i = df_dataset.loc[df_dataset['label'] == i]
        list_pd_labels.append(pd_label_i)
    for i, curr_pd in enumerate(list_pd_labels):
        if len(curr_pd) > num_samples_label:
            list_pd_labels[i] = curr_pd[:num_samples_label]
        else:
            logger.info("num samples per label is larger than each label frame")
    merged_equal_pd = pd.concat(list_pd_labels)
    dataset_sampled_equal_test = datasets.Dataset.from_pandas(merged_equal_pd)


    return dataset_sampled_ne_test, dataset_sampled_equal_test


def print_trainable_parameters(model):
    """
    Outputs the number of trainable parameters in the model and the total number of parameters.
    This allows you to see the size of your model and the percentage of parameters used for training.
    """
    trainable_params = 0
    total_params = 0

    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_percent = 100 * trainable_params / total_params

    logger.info(f"Trainable Parameters: {trainable_params}")
    logger.info(f"Total Parameters: {total_params}")
    logger.info(f"Trainable %: {trainable_percent:.2f}")

def wrapper_ent_attn_flow(cfg, encoder, tokenizer, sent_corpus, top_k=5, target_layer=-1):
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
    num_sents_short = 0  # which is not considered due to short length

    num_sents_no_vp = 0  # which does not include VP or wrongly parsed
    num_sents_corrupted_vp = 0
    num_sents_considered = 0

    list_len_subwords = []  # the number of subwords for each sentence    
    list_len_subwords_ent = []  # the number of subwords of all entities for each sentence    
    list_len_subwords_vp = []  # the number of subwords of all verbs for each sentence    

    list_num_entities = []
    ind_row = 0
    for _, row in sent_corpus.iterrows():

        if cache_entity:
            list_tags_sbw_loc = list_doc_np_sbw_loc[ind_row]
        else:
            list_tags_sbw_loc = []
        cur_sent = row["text"]
        cur_label = row["label"]
        cur_sent = filter_sentence(cur_sent)
        if len(cur_sent.split()) < 5:
            num_sents_short += 1
            continue
        output_attn_flow = ent_attn_flow.run_attn_flow(cfg, tokenizer, encoder, np_parser, cur_sent, list_tags_sbw_loc,
                                        top_k=top_k, target_layer=target_layer, cache_entity=cache_entity)
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

            list_num_entities.append(len(list_np_sbw_loc))
            
            seq_len = output_attn_flow["seq_len_subwords"]
            seq_len_ent = output_attn_flow["len_subwords_ent"]
            seq_len_vp = output_attn_flow["len_subwords_vp"]

            list_len_subwords.append(seq_len)
            list_len_subwords_ent.append(seq_len_ent)
            list_len_subwords_vp.append(seq_len_vp)
            if not math.isnan(ent_top_k):
                list_val_ent_topk.append(ent_top_k)
            if not math.isnan(vp_top_k):
                list_val_vp_topk.append(vp_top_k)
            if not math.isnan(aflow_type1):
                list_val_interact_type1.append(aflow_type1)
            if not math.isnan(aflow_type2):
                list_val_interact_type2.append(aflow_type2)
            if not math.isnan(aflow_type3):
                list_val_interact_type3.append(aflow_type3)
            if not math.isnan(aflow_type4):
                list_val_interact_type4.append(aflow_type4)

            list_doc_np_sbw_loc.append(list_np_sbw_loc)
            list_doc_vp_sbw_loc.append(list_vp_sbw_loc)

            num_sents_considered += 1

        else:
            num_sents_corrupted += 1
        list_np_sbw_loc = output_attn_flow.get("list_np_sbw_loc", [])
        list_vp_sbw_loc = output_attn_flow.get("list_vp_sbw_loc", [])
        if len(list_np_sbw_loc) < 1:
            num_sents_non_ent += 1
        if len(list_vp_sbw_loc) < 1:
            num_sents_no_vp += 1
        if cfg.exp_args.sample_num != -1 and ind_row > cfg.exp_args.sample_num:
            break

        if ind_row % 1000 == 0:
            logger.info(f"Document: {ind_row}")
        
        ind_row += 1

    num_sents_all = len(sent_corpus)
       
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
    outputs = get_avg_from_lists(list_val=list_val_ent_topk)
    avg_sents_entk, std_sents_entk = float(outputs[0]), float(outputs[1])
    logger.info(f"Ent@K Avg(Std): {avg_sents_entk} ({std_sents_entk})" )

    outputs = get_avg_from_lists(list_val=list_val_vp_topk)
    avg_sents_vpk, std_sents_vpk = float(outputs[0]), float(outputs[1])
    logger.info(f"VP@K Avg(Std): {avg_sents_vpk} ({std_sents_vpk})" )

    logger.info("")
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

    outputs.append((avg_sents_type1, std_sents_type1))
    outputs.append((avg_sents_type2, std_sents_type2))
    outputs.append((avg_sents_type3, std_sents_type3))

    return outputs


def naive_inference(cfg, dataset_test, encoder, tokenizer, messages):
    preds, labels = inference_sentiment(dataset_test, cfg.model.llm_id, encoder, tokenizer, messages)
    preds_discrete = convert_label_list_int(preds)
    acc = eval_accuracy(preds_discrete, labels)
    kappa_el_linear, kappa_el_quad = eval_kappa(preds_discrete, labels)

    logger.info("Evaluation: Before PEFT ----")
    logger.info("Accuracy: %.2f" % (acc))
    logger.info("Kappa El: %.2f %.2f" % (kappa_el_linear, kappa_el_quad))
    logger.info("-------")
    logger.info("")

    hist_preds_peft, hist_labels_peft = get_hist_preds_labels(preds_discrete, labels)
    logger.info(hist_preds_peft)
    logger.info(hist_labels_peft)

    hist_preds_peft = fill_empty_label(hist_preds_peft)

    plot_preds_labels_dists(hist_preds_peft, hist_labels_peft, "naive", is_equal_label_dist=True)


def ent_attn_layer_analysis(cfg, encoder, tokenizer, sent_corpus, max_layer=32):
    for target_layer in range(max_layer):
        logger.info(f"Target Layer: {target_layer+1}")
        wrapper_ent_attn_flow(cfg, encoder, tokenizer, sent_corpus, cfg.exp_args.top_k, target_layer)
    return


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    hf_token = os.environ.get("HF_TOKEN") or OmegaConf.select(cfg, "hf_token") or None
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.llm_id, low_cpu_mem_usage=True, token=hf_token)
    encoder_dtype = torch.bfloat16
    if "xlnet" in cfg.model.llm_id or "google-bert" in cfg.model.llm_id:
        encoder_dtype = "auto"
    encoder = AutoModelForCausalLM.from_pretrained(cfg.model.llm_id, low_cpu_mem_usage=True, token=hf_token, offload_folder="offload", offload_state_dict=True, max_memory={"cpu": "10GIB"}, torch_dtype=encoder_dtype)
    encoder.to("cuda")
    logger.info(f"Target LLM:   {cfg.model.llm_id}")
    dataset_sst5 = load_dataset("SetFit/sst5")
    train_ds, valid_ds, test_ds = dataset_sst5["train"], dataset_sst5["validation"], dataset_sst5["test"]

    dataset_sampled_ne_train, dataset_sampled_equal_train = sample_dataset(train_ds, num_labels=5, sample_ratio=0.1, num_samples_label=1000)
    dataset_sampled_ne_test, dataset_sampled_equal_test = sample_dataset(test_ds, num_labels=5, sample_ratio=0.1, num_samples_label=200)
    messages = [{"role": "user", "content": ""}]

    logger.info(f"Top_K:    {cfg.exp_args.top_k}")
    logger.info("")
    logger.info("----- Naive Inference ----- ")
    naive_inference(cfg, dataset_sampled_equal_test, encoder, tokenizer, messages)
    logger.info("")
    train_pd = pd.DataFrame(dataset_sampled_equal_train)
    test_pd = pd.DataFrame(dataset_sampled_equal_test)
    sent_corpus = pd.concat([train_pd, test_pd])

    num_docs = len(sent_corpus)
    logger.info(f"Total # of Sentences: {num_docs}")
    logger.info("------- Ent Attn Flow ------")
    output_ent_attn_flow = wrapper_ent_attn_flow(cfg, encoder, tokenizer, sent_corpus, cfg.exp_args.top_k, target_layer=-1)
    logger.info("")
    logger.info("------- PEFT ------")
    encoder = peft_encoder(cfg.model.llm_id, encoder, tokenizer, dataset_sampled_equal_train, dataset_sampled_equal_test, messages)
    logger.info("")
    train_pd = pd.DataFrame(dataset_sampled_equal_train)
    test_pd = pd.DataFrame(dataset_sampled_equal_test)
    sent_corpus = pd.concat([train_pd, test_pd])

    num_docs = len(sent_corpus)
    logger.info(f"Total # of Sentences: {num_docs}")
    logger.info("------- Ent Attn Flow ------")
    output_ent_attn_flow = wrapper_ent_attn_flow(cfg, encoder, tokenizer, sent_corpus, cfg.exp_args.top_k, target_layer=-1)
    
    list_val_interact_type1 = output_ent_attn_flow[0]
    list_val_interact_type2 = output_ent_attn_flow[1]
    list_val_interact_type3 = output_ent_attn_flow[2]


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    ptime = (end - start) / 60.0 # minutes
    # print(f"Processing Time: {ptime:.2f} mins")
    logging.info(f"Processing Time: {ptime:.2f} mins")
