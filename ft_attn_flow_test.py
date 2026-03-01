import os, sys
import numpy as np
import pandas as pd
import statistics
import logging
import collections
import re
    
import torch
import torch.nn.functional as f

from entity_parser.np_parser_backt import NP_Parser_BackT

from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import XLNetModel, T5EncoderModel
from transformers import logging
logging.set_verbosity_error()

import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import json

import time

from pathlib import Path

import logging

import math

import ent_attn_func.attn_flow as ent_attn_flow

import datasets
from datasets import load_dataset, concatenate_datasets

from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from peft import AutoPeftModel, AutoPeftModelForCausalLM
from peft import prepare_model_for_kbit_training

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

from matplotlib import pyplot as plt
import numpy as np

def plot_preds_labels_dists(hist_preds, hist_labels, label_out, is_equal_label_dist=False):
    bar_width = 0.20
    fig, ax = plt.subplots()
    list_preds = sorted(hist_preds.items())  # tuple (x, y)
    x_preds, y_preds = zip(*list_preds)

    list_labels = sorted(hist_labels.items())  # tuple (x, y)
    x_labels, y_labels = zip(*list_labels)

    x_name = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    labels = ["Predictions", "Gold-Labels"]

    shift = 0.5  # x-axis shift location from zero center in the figure

    rects1 = plt.bar(np.arange(len(x_name)) - (shift * bar_width), y_preds, color='#7eb0d5', label=labels[0], width=bar_width)
    rects2 = plt.bar(np.arange(len(x_name)) + (shift * bar_width), y_labels, color='#bd7ebe', label=labels[1], width=bar_width)

    plt.xticks(np.arange(len(y_preds)), x_name, rotation=0)  ## for ASAP
    if is_equal_label_dist:
        plt.title('Sentiment Analysis Distribution: Preds vs. Labels (Equal)')
    else:
        plt.title('Sentiment Analysis Distribution: Preds vs. Labels (Non-Equal)')

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.005*height,
                    '%d' % float(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    ax.legend()
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, 120))
    plt.ylabel('The number of predictions/labels')
    fig.tight_layout()

    plt.savefig(label_out + '_' + 'preds' + '.png', format='png', dpi=300)

def get_hist_preds_labels(preds, labels):
  hist_preds = collections.Counter(preds)
  hist_labels = collections.Counter(labels)

  return hist_preds, hist_labels


def fill_empty_label(hist_curr, num_class=5):
  for i in range(num_class):
    if i not in hist_curr:
      hist_curr[i] = 0

  return hist_curr

def convert_msgs_format(curr_dataset):
    '''
        converting dataset to chat-based format, so that SFT trainer uses chat template with the given tokenizer
        https://huggingface.co/docs/trl/en/sft_trainer
    '''
    from datasets import Dataset
    import pandas as pd

    list_dict_msgs = []
    for i, curr in enumerate(curr_dataset):
        curr_msg = []
        curr_msg.append({"role": "user", "content": "Your role is to classify the sentiment of conversation into 5 classes: very_positive, positive, neutral, negative, or very_negative. You must generate only one word of sentiment class" + "\n\n" + curr["text"]})
        curr_msg.append({"role": "assistant", "content": curr["label"]})

        curr_msgs_dict = {"messages": curr_msg}
        list_dict_msgs.append(curr_msgs_dict)
    dataset_chat_format = datasets.Dataset.from_pandas(pd.DataFrame(data=list_dict_msgs))
    return dataset_chat_format

def clean_generated_sentiment_class(preds):
    cleaned_preds = []
    sentiment_labels = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    for i, curr in enumerate(preds):
        splitted = curr.split()
        if len(splitted) > 1 and splitted[0] not in sentiment_labels:
            label_str = "neutral"
        else:
            label_str = curr

        cleaned_preds.append(label_str)
    return cleaned_preds

def convert_label_str(sample):
    '''
        converting label from number to string (to work with chat-based model)
    '''
    label_int = sample["label"]
    label_str = ""
    if label_int == 0:
        label_str = "very_negative"
    elif label_int == 1:
        label_str = "negative"
    elif label_int == 2:
        label_str = "neutral"
    elif label_int == 3:
        label_str = "positive"
    elif label_int == 4:
        label_str = "very_positive"

    sample["label"] = label_str
    return sample

def convert_label_int(sample):
    label_str = sample["label"]
    label_int = 0
    if  label_str == "very_negative":
        label_int = 0
    elif label_str == "negative":
        label_int = 1
    elif label_str == "neutral":
        label_int = 2
    elif label_str == "positive":
        label_int = 3
    elif label_str == "very_positive":
        label_int = 4

    sample["label"] = label_int
    return sample

def filter_generated_text(text, pre_header, post_header):
    ind_response_start = text.rfind(pre_header)
    len_prefix = len(pre_header)
    only_generated_text = text[ind_response_start + len_prefix:]
    ind_suffix = only_generated_text.find(post_header)
    only_generated_text = only_generated_text[:ind_suffix]

    only_generated_text = only_generated_text.strip()
    only_generated_text = only_generated_text.lower()
    
    return only_generated_text

def inference_sentiment(dataset_curr, llm_id, model, tokenizer, messages, max_new_tokens=5):
    preds = []
    labels = []
    for i in range(len(dataset_curr)):
        prompt = "Your role is to classify the sentiment of conversation into 5 classes: very_positive, positive, neutral, negative, or very_negative. You must generate only one word of sentiment class."
        cur_user_response = dataset_curr[i]["text"]
        label_curr = dataset_curr[i]["label"]
        messages[-1]["content"] = prompt + "\n\n" + cur_user_response
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to('cuda')
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
        text = tokenizer.batch_decode(outputs)[0]
        if "microsoft/Phi-" in llm_id:
            pre_header = "<|assistant|>"
            post_header = "<|end|>"
        elif "meta-llama/Llama-3" in llm_id:
            pre_header = "assistant<|end_header_id|>"
            post_header = "<|eot_id|>"
        elif "google/gemma-2" in llm_id:
            pre_header = "<start_of_turn>model"
            post_header = "<end_of_turn>"
        elif "Qwen2.5" in llm_id:
            pre_header = "<|im_start|>assistant"
            post_header = "<|im_end|>"

        only_generated_text = filter_generated_text(text, pre_header, post_header)
        preds.append(only_generated_text)
        labels.append(label_curr)

    return preds, labels

def eval_accuracy(preds, labels):
  acc = sum(1 for x, y in zip(preds, labels) if x == y) / len(preds)
  return acc

def eval_kappa(preds, labels, is_quad_also=True):
  from sklearn.metrics import cohen_kappa_score
  kappa_linear = cohen_kappa_score(preds, labels, weights="linear")
  kappa_quad = 0.0
  if is_quad_also:
    kappa_quad = cohen_kappa_score(preds, labels, weights="quadratic")
  return kappa_linear, kappa_quad

def convert_label_list_int(labels):
    labels_int = []
    for i, curr in enumerate(labels):
        if  curr == "very_negative":
            label_int = 0
        elif curr == "negative":
            label_int = 1
        elif curr == "neutral":
            label_int = 2
        elif curr == "positive":
            label_int = 3
        elif curr == "very_positive":
            label_int = 4
        else:
            label_int = 2
        labels_int.append(label_int)

    return labels_int

def convert_pred_numeric(preds):
  '''
    convert langchain document class to numerical labels
    just to keep the given output format (otherwise output format can be modified depending on the dataset)
  '''
  preds_numeric = []

  for curr in preds:
    curr_pred = curr.name  # e.g., negative

    if curr_pred == "very_negative": label = 0
    elif curr_pred == "negative": label = 1
    elif curr_pred == "neutral": label = 2
    elif curr_pred == "positive": label = 3
    elif curr_pred == "very_positive": label = 4

    preds_numeric.append(label)

  return preds_numeric


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

def get_avg_from_lists(list_val):
    if all(isinstance(item, list) for item in list_val):
        flatten_sents = [x for xs in list_val for x in xs]
    elif all(not isinstance(item, list) for item in list_val):
        flatten_sents = list_val
    np_flatten = np.array(flatten_sents)
    avg_sents = np.nanmean(np_flatten)
    std_sents = np.nanstd(np_flatten)
    outputs = []
    outputs.append(avg_sents)
    outputs.append(std_sents)

    return outputs


def filter_sentence(cur_sent):

    filter_punctation = [".", ":"]

    cur_sent = cur_sent.replace(",", ", ")
    cur_sent = cur_sent.replace("\'", " ")
    cur_sent = cur_sent.replace("\"", ", ")
    cur_sent = cur_sent.replace("-", " ")
    cur_sent = cur_sent.replace("/", " ")
    cur_sent = cur_sent.replace("*", " ")
    cur_sent = cur_sent.replace("<", " ")
    cur_sent = cur_sent.replace(">", " ")
    cur_sent = re.sub(r'\.{2,}', '. ', cur_sent)
    cur_sent = re.sub(r'\!{2,}', '! ', cur_sent)
    cur_sent = re.sub(r'\?{2,}', '? ', cur_sent)

    cur_sent = re.sub(r"\s+", " ", cur_sent, flags=re.UNICODE)
    cur_sent = cur_sent.strip()

    if len(cur_sent) > 1 and cur_sent[-1] in filter_punctation:
        cur_sent = cur_sent[:-1]

    return cur_sent

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


def load_dataset_toefl(cfg, num_samples=0):

    cur_fold = 0
    str_cur_fold = str(cur_fold)

    #### load dataset from files
    train_pd = pd.read_csv(os.path.join(cfg.dataset.path_data, "sst_train_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c', index_col=0)
    valid_pd = pd.read_csv(os.path.join(cfg.dataset.path_data, "sst_valid_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c', index_col=0)
    test_pd = pd.read_csv(os.path.join(cfg.dataset.path_data, "sst_test_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c', index_col=0)
    train_pd = train_pd.loc[train_pd['prompt'] == cfg.dataset.target_prompt]
    valid_pd = valid_pd.loc[valid_pd['prompt'] == cfg.dataset.target_prompt]
    test_pd = test_pd.loc[test_pd['prompt'] == cfg.dataset.target_prompt]
    if num_samples > 0:
        train_pd = train_pd[:num_samples]
        valid_pd = valid_pd[:num_samples]
        test_pd = test_pd[:num_samples]
    total_pd = pd.concat([train_pd, valid_pd, test_pd], sort=True)
    total_corpus = total_pd['essay'].values
    import stanza
    tokenizer_stanza = stanza.Pipeline('en', processors='tokenize', use_gpu=True)
    num_sents = []
    sent_corpus = []
    for cur_doc in total_corpus:
        doc_stanza = tokenizer_stanza(cur_doc)
        sent_list = [sentence.text for sentence in doc_stanza.sentences]
        
        sent_corpus.append(sent_list)
        num_sents.append(len(sent_list))


    return sent_corpus, num_sents


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    hf_token = os.environ.get("HF_TOKEN") or OmegaConf.select(cfg, "hf_token") or None
    use_4bit = True
    use_nested_quant = False
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    fp16 = True
    bf16 = False

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

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
