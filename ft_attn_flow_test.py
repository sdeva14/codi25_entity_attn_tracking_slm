import os, sys
# from datasets import Dataset, load_dataset
# import datasets
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

#
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

# from corpus.dataset_toefl_hf import Dataset_TOEFL

#############################
## utility for PEFT exp

def peft_encoder(llm_id, encoder, tokenizer, train_dataset, test_dataset, messages):

    ## data load and preprocessing
    '''
        1092
        2218
        1624
        2322
        1288
    '''
    train_dataset = train_dataset.map(convert_label_str)

    ## prepare PEFT model setup
    config = LoraConfig(
        r=8,
        lora_alpha=32,

        # target_modules = ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"] #targetting all modules for more intensive training
        # target_modules = ["q_proj", "v_proj"], #There are options to deepen the finetuning by unfreezing more weights but with a cost in performance

        # target_modules = ["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],  # only for MS Phi-3 model architecture
        # target_modules = ["o_proj", "qkv_proj"],  # only for MS Phi-3 model architecture
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],  # meta-llama3 -> 0.14%; google/gemma-2-2b-it -> 0.12%; 
        # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # meta-llama3 -> 0.38%
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # # https://huggingface.co/docs/peft/en/package_reference/peft_types
    # config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    # )

    encoder.gradient_checkpointing_enable()
    encoder = prepare_model_for_kbit_training(encoder)

    encoder = get_peft_model(
        encoder,
        config,
    )
    print_trainable_parameters(encoder)

    # print(ewlkfjlkwef)

    ## training with PEFT

    # converting dataset to chat-based format
    dataset_sft_train = convert_msgs_format(train_dataset)
    logger.info(dataset_sft_train)
    logger.info(dataset_sft_train[0])

    # SFT Trainer

    # # set up [pad] in tokenizer
    # tokenizer.pad_token = tokenizer.eos_token
    # self.pad_id: int = self.special_tokens["<|reserved_special_token_0|>"]
    # tokenizer.pad_token = tokenizer.special_tokens["<|reserved_special_token_0|>"]
    # tokenizer.pad_id: int = tokenizer.special_tokens["<|reserved_special_token_0|>"]
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # better when llm uses 0 index for something e.g.,) llama3
    tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})

    from transformers import TrainingArguments, Trainer
    from trl import SFTTrainer

    fp16 = False
    # Enable bf16 training
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

    # Setting sft parameters
    trainer = SFTTrainer(
        model=encoder,
        train_dataset=dataset_sft_train,  # sampling (converted to chat-based format)
        # train_dataset=dataset_sst5["train"],  # whole training set
        peft_config=config,
        max_seq_length= None,
        # dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing= False,
    )
    trainer.train()

    preds_peft, labels_peft = inference_sentiment(test_dataset, llm_id, encoder ,tokenizer, messages)
    # print(preds_peft)
    # print(labels_peft)

    preds_peft = clean_generated_sentiment_class(preds_peft)
    # print(preds_peft)

    # eval
    preds_discrete_peft = convert_label_list_int(preds_peft)
    acc = eval_accuracy(preds_discrete_peft, labels_peft)
    kappa_el_linear, kappa_el_quad = eval_kappa(preds_discrete_peft, labels_peft)

    logger.info("Accuracy: %.2f" % (acc))
    # print("Kappa Ne: %.2f %.2f" % (kappa_ne_linear, kappa_ne_quad))
    logger.info("Kappa El: %.2f %.2f" % (kappa_el_linear, kappa_el_quad))
    logger.info("-------")
    logger.info("")

    ## analysis

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

    # plt.show()
    plt.savefig(label_out + '_' + 'preds' + '.png', format='png', dpi=300)

def get_hist_preds_labels(preds, labels):
  hist_preds = collections.Counter(preds)
  hist_labels = collections.Counter(labels)

  return hist_preds, hist_labels


# fill the empty labels
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
        # curr_msg.append({"role": "system", "content": "Classify the sentiment of conversation into 5 classes: very_positive, positive, neutral, negative, or very_negative. You must generate only one word of sentiment class"})
        # curr_msg.append({"role": "assistant", "content": "Hello, which one is the review to classify the sentiment?"})
        # curr_msg.append({"role": "user", "content": curr["text"]})
        # curr_msg.append({"role": "assistant", "content": curr["label"]})

        curr_msg.append({"role": "user", "content": "Your role is to classify the sentiment of conversation into 5 classes: very_positive, positive, neutral, negative, or very_negative. You must generate only one word of sentiment class" + "\n\n" + curr["text"]})
        curr_msg.append({"role": "assistant", "content": curr["label"]})

        curr_msgs_dict = {"messages": curr_msg}
        list_dict_msgs.append(curr_msgs_dict)

    # dataset_sft_train = Dataset.from_list(list_dict_msgs)
    dataset_chat_format = datasets.Dataset.from_pandas(pd.DataFrame(data=list_dict_msgs))
    return dataset_chat_format

def clean_generated_sentiment_class(preds):
    # there could be a case that chat-based model didn't work as prompting, then we need to extract the sentiment class from the generated text
    # e.g., "this conversation of sentiment is positive. Despite of ..."
    cleaned_preds = []
    sentiment_labels = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    for i, curr in enumerate(preds):
        splitted = curr.split()
        if len(splitted) > 1  and splitted[0] not in sentiment_labels:
            # list_loc_sentiment= []
            # for target_sentiment in sentiment_labels:
            #     loc_sentiment = curr.find(target_sentiment)
            #     list_loc_sentiment.append(loc_sentiment)

            # first_emotion = list_loc_sentiment.index(min(list_loc_sentiment))
            # label_str = sentiment_labels[first_emotion]
            label_str = "neutral"  # Phi-3 model struggles in the neural case, so simply assume neutral for now
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

def inference_sentiment(dataset_curr, llm_id, model ,tokenizer, messages, max_new_tokens=5):

    #
    preds = []
    labels = []

    # dataset_curr = dataset_sampled_equal_test
    for i in range(len(dataset_curr)):

        # cur_user_response = dataset_curr[len(dataset_curr)-i-1]["text"]
        prompt = "Your role is to classify the sentiment of conversation into 5 classes: very_positive, positive, neutral, negative, or very_negative. You must generate only one word of sentiment class."
        cur_user_response = dataset_curr[i]["text"]
        # chat_curr = sentiment_template.format(conversation=cur_user_response)
        label_curr = dataset_curr[i]["label"]

        # update the messages with an input
        messages[-1]["content"] = prompt + "\n\n" + cur_user_response
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")  # (batch, len_seq)
        inputs = inputs.to('cuda')

        # inputs = inputs.bfloat16()

        # classify the sentiment via chat model
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
        text = tokenizer.batch_decode(outputs)[0]

        # print(text)
        # print(ewlkfjwelf)

        # formatting to extract only the response
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

        # print(only_generated_text)
        # print(weklfjwlwef)

        # formatting check
        # if the model generate a sentence ignoring the intruction, then we need to split sentences, then extract the sentiment class in the first sentence
        # e.g.) The sentiment of this conversation .... is positive. Despite that it shows ...

        # print(only_generated_text)
        # print("----")

        ##
        preds.append(only_generated_text)  # "positive", "very_positive", ...
        labels.append(label_curr)  # 0, 1, ...

    return preds, labels

def eval_accuracy(preds, labels):
  # accuracy
  # print(preds)
  # print(labels)
  acc = sum(1 for x,y in zip(preds,labels) if x == y) / len(preds)
  # print(acc)
  return acc

def eval_kappa(preds, labels, is_quad_also=True):
  # kappa (inter-annotator agreement)
  from sklearn.metrics import cohen_kappa_score
  kappa_linear = cohen_kappa_score(preds, labels, weights="linear")
  kappa_quad = 0.0
  if is_quad_also:
    kappa_quad = cohen_kappa_score(preds, labels, weights="quadratic")

  # print(kappa_linear)
  # print(kappa_quad)

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
            # print(welkfjewlkfjwelf)
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

#### sampling dataset

def sample_dataset(curr_dataset, num_labels=5, sample_ratio=0.1, num_samples_label=40):

    ## sampling dataset without label balancing
    # sample_ratio = 0.1
    sample_num = round(len(curr_dataset) * sample_ratio)
    # sample_num = 500

    list_indices = list(range(sample_num))

    dataset_sampled_ne_test = curr_dataset.select(list_indices)

    ## sampling with equal label distribution
    # convert huggingface dataset class to Pandas dataframe to filter labels easily
    import pandas as pd
    df_dataset = pd.DataFrame(curr_dataset)

    # split dataframe by their label
    list_pd_labels = []  # each pd including their labels of index in the list

    for i in range(num_labels):
        pd_label_i = df_dataset.loc[df_dataset['label'] == i]
        list_pd_labels.append(pd_label_i)
        # print(len(pd_label_i)) # SST5: 279, 633, 389, 510, 399

    # sampling as the target number
    for i, curr_pd in enumerate(list_pd_labels):
        if len(curr_pd) > num_samples_label:
            list_pd_labels[i] = curr_pd[:num_samples_label]
        else:
            logger.info("num samples per label is larger than each label frame")

    merged_equal_pd = pd.concat(list_pd_labels)
    # print(merged_equal_pd)
    # print(len(merged_equal_pd))

    # convert dataframe to huggingface dataset again
    dataset_sampled_equal_test = datasets.Dataset.from_pandas(merged_equal_pd)


    return dataset_sampled_ne_test, dataset_sampled_equal_test

#############################
### stat analysis

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

    # avg in sentence level
    if all(isinstance(item, list) for item in list_val):
        flatten_sents = [x for xs in list_val for x in xs]
    elif all(not isinstance(item, list) for item in list_val):
        flatten_sents = list_val

    # print(flatten_sents)
    # print(len(flatten_sents))
    # sent_num = len(flatten_sents)

    # avg_sents = statistics.mean(flatten_sents)
    # std_sents = statistics.stdev(flatten_sents)

    np_flatten = np.array(flatten_sents)
    avg_sents = np.nanmean(np_flatten)
    std_sents = np.nanstd(np_flatten)

    # # avg in document level?
    # flatten_docs = []
    # for cur_doc in list_val:
    #     cur_flatten = []
    #     for cur_sent in cur_doc:
    #         cur_flatten.append(cur_sent)
    #     flatten_docs.append(cur_flatten)
    
    outputs = []
    outputs.append(avg_sents)
    outputs.append(std_sents)

    return outputs

#############################
### entity attention

# def attn_flow_entity(model_weights, attn_weights, list_np_sbw_loc):
#     '''
#     inputs
#     - attn_weights: attention weights from LLMs (batch_size, len_seq, len_seq)
#     - ind_ent: indexes of entities, mapped to subwords

#     outputs
#     - the amount of flow between entities and others
#     - the amount of flow beteen entities##### check cache for entity parsing
#     '''
    
#     # print(attn_weights.shape)

#     # ## cut the last special tokens (e.g., ".")
#     # if model_weights != "susnato/phi-1_5_dev":
#     #     attn_weights = attn_weights[:, :-1, :-1]
    
#     batch_size, len_seq = attn_weights.size(0), attn_weights.size(1)

#     ## normalize attention weights first

#     attn_diag = torch.tril(attn_weights, diagonal=-1)  # only keep the lower triangle, excluding the diagonal item (masking the item itself)
#     # attn_weights_norm = attn_weights / attn_weights.sum(dim=-1).unsqueeze(-1)  # normalization used in "Entropy- and Distance-Based Predictors From GPT-2..."
#     # attn_weights_norm = f.normalize(attn_diag, p=1, dim=-1, eps=1e-6)  # normalize by each row vector its 2-norm

#     denorm = torch.sum(attn_diag)
#     attn_weights_norm = attn_diag / denorm  # because we need to investigate the interaction between all combination in the sequence

#     # print(attn_weights)
#     # print(attn_weights_norm)
#     # print(ewklfjwelf)

#     #### calculate attention flow
#     ind_ent = [curr[1] for curr in list_np_sbw_loc]
#     # print(ind_ent)
#     # print(elwkjflwef)

#     ## prepare masking all entities, non-entities
#     list_all_ind_ent = []  # index of all tokens included in the all entities
#     for cur_ind_pair in ind_ent:
#         ind_start, ind_end = cur_ind_pair[0], cur_ind_pair[1]  # index of entities mapped to subwords
#         list_ind_ent = list(range(ind_start, ind_end+1))
#         list_all_ind_ent.extend(list_ind_ent)
    
#     list_all_ind_others = list(range(len_seq))
#     list_all_ind_others = list(set(list_all_ind_others).difference(list_all_ind_ent))

#     # print(attn_weights.shape)

#     # mask_all_ent = torch.zeros(batch_size, len_seq)
#     # mask_all_ent[:, list_all_ind_ent] = 1
#     # mask_all_ent = mask_all_ent.bool()
#     # mask_ohters = ~mask_all_ent

#     # num_tokens_entity = len(list_all_ind_ent)
#     # num_tokens_others = len_seq - num_tokens_entity

#     ########################################################
#     #### calculating at once by maskign in 2d matrix?

#     ## masking inner-entity items -> because we do not want to consider relationships between items in the same entity but want to consider relationships in entity and non-entity levels?
#     inner_ent_masked_attn = attn_weights_norm.clone()
#     for cur_ind_pair in ind_ent:
#         ind_start, ind_end = cur_ind_pair[0], cur_ind_pair[1]  # index of entities mapped to subwords
#         list_ind_ent = list(range(ind_start, ind_end+1))

#         # print(list_ind_ent)
#         # inner item masking in the current entity
#         for i in range(len(list_ind_ent)-1):
#             for j in range(i+1, len(list_ind_ent)):
#                 ind_a = list_all_ind_ent[i]
#                 ind_b = list_all_ind_ent[j]
#                 inner_ent_masked_attn[:, ind_a, ind_b] = 0.0  # already self-tokens are maksed in diagonal operations, so dont need to consider
#                 inner_ent_masked_attn[:, ind_b, ind_a] = 0.0

#     # print(attn_weights_norm)
#     # print()
#     # print(inner_ent_masked_attn)
#     # print(elkfjweewf)

#     ## type1: between entities and non-entities (iterate tokens in the current entity)
#     # self-masking by the entity included, select non-entities

#     # print(list_all_ind_ent)
#     mask_type1 = torch.ones(batch_size, len_seq, len_seq).cuda()
#     for i in range(len(list_all_ind_ent)-1):
#         for j in range(i+1, len(list_all_ind_ent)):
#             ind_a = list_all_ind_ent[i]
#             ind_b = list_all_ind_ent[j]

#             # mask_type1[:, ind_ent_a, ind_ent_a] = 0.0  # indeed it is not needed since already intra entity tokens are masked
#             # mask_type1[:, ind_ent_b, ind_ent_b] = 0.0
#             mask_type1[:, ind_a, ind_b] = 0.0
#             mask_type1[:, ind_b, ind_a] = 0.0

#     mask_type1 = mask_type1.bool()
#     # print(mask_type1)
#     # print(ewkljflwef)

#     ## type2: between entities
#     mask_type2 = torch.zeros(batch_size, len_seq, len_seq).cuda()
#     for i in range(len(list_all_ind_ent)-1):
#         for j in range(i+1, len(list_all_ind_ent)):
#             ind_a = list_all_ind_ent[i]
#             ind_b = list_all_ind_ent[j]

#             mask_type2[:, ind_a, ind_b] = 1.0
#             mask_type2[:, ind_b, ind_a] = 1.0

#     mask_type2 = mask_type2.bool()
#     # print(mask_type2)

#     ## type3: between others (non-entities)
#     # print(list_all_ind_others)
#     mask_type3 = torch.zeros(batch_size, len_seq, len_seq).cuda()
#     for i in range(len(list_all_ind_others)-1):
#         for j in range(i+1, len(list_all_ind_others)):
#             ind_a = list_all_ind_others[i]
#             ind_b = list_all_ind_others[j]

#             mask_type3[:, ind_a, ind_b] = 1.0
#             mask_type3[:, ind_b, ind_a] = 1.0

#     mask_type3 = mask_type3.bool()
#     # print(mask_type3)

#     ########################################################
#     ## calculating attention flow

#     # type1
#     vals = torch.masked_select(inner_ent_masked_attn, mask_type1)
#     nz_vals = vals[torch.nonzero(vals)]
#     # avg_vals = torch.mean(nz_vals, dim=0)
#     sum_vals = torch.sum(nz_vals)
#     # print(avg_vals)
#     # if torch.isnan(avg_vals):
#     #     list_val_interact_type1.append(avg_vals)
#     val_type1 = float(sum_vals)

#     # type2
#     vals = torch.masked_select(inner_ent_masked_attn, mask_type2)
#     nz_vals = vals[torch.nonzero(vals)]
#     # avg_vals = torch.mean(nz_vals, dim=0)
#     sum_vals = torch.sum(nz_vals)
#     # print(nz_vals)
#     # print(avg_vals)
#     # if torch.isnan(avg_vals):
#     #     list_val_interact_type2.append(avg_vals)
#     val_type2 = float(sum_vals)

#     # type3
#     vals = torch.masked_select(inner_ent_masked_attn, mask_type3)
#     nz_vals = vals[torch.nonzero(vals)]
#     # avg_vals = torch.mean(nz_vals, dim=0)
#     sum_vals = torch.sum(nz_vals)
#     # print(nz_vals)
#     # print(avg_vals)
#     # if torch.isnan(avg_vals):
#     #     list_val_interact_type3.append(avg_vals)
#     val_type3 = float(sum_vals)

#     # print(list_val_interact_type1)
#     # print(list_val_interact_type2)
#     # print(list_val_interact_type3)
#     # print(ewklfjklwef)

#     return val_type1, val_type2, val_type3

########

# def filter_attn_special_tokens(llm_id, attn_layers):
#     filtered_layers = []
#     if "meta-llama/Llama-3" in llm_id or "google/gemma-2" in llm_id:
#         for cur_layer in attn_layers:  # (batch, mh, seq_len, seq_len)
#             filtered_layers.append(cur_layer[:, :, 1:, 1:])
    
#     elif "xlnet" in llm_id:
#         for cur_layer in attn_layers:  # (batch, mh, seq_len, seq_len)
#             filtered_layers.append(cur_layer[:, :, :-2, :-2])

#     elif "google-bert/bert-" in llm_id:
#         for cur_layer in attn_layers:  # (batch, mh, seq_len, seq_len)
#             filtered_layers.append(cur_layer[:, :, 1:-1, 1:-1])
#     else:
#         return attn_layers
    
#     return filtered_layers


# def run_attn_flow(cfg, tokenizer, encoder, np_parser, cur_sent, list_np_sbw_loc, target_layer=-1, cache_entity=False):
        
#     # print("--------------")
#     # print("Curr Sent: {}".format(cur_sent))

#     # encode sentence
#     tokenized = tokenizer(cur_sent, return_tensors="pt")
#     tokenized.to("cuda")
#     cur_sent_ids = tokenized["input_ids"]
#     cur_attn_mask = tokenized["attention_mask"]
#     # tokenized_seg_ids.append(cur_sent_ids)
#     # tokenized_attn_mask.append(cur_attn_mask)

#     # ###################
#     # # ### debuging
#     # decoded_word = tokenizer.batch_decode(cur_sent_ids)
#     # print(decoded_word)

#     # i_ids = cur_sent_ids[0].tolist()
#     # decoded_tokens = tokenizer.convert_ids_to_tokens(i_ids)
#     # print(decoded_tokens)
#     # #####################

#     encoder_out = encoder(cur_sent_ids, cur_attn_mask, output_attentions=True, output_hidden_states=True)
#     # print(encoder_out.keys())

#     ## encoder model output
#     # repr_encoded = encoder_out["last_hidden_state"]  # (batch_size, max_sent_len, dim)
#     # attn_layers = encoder_out["attentions"]  # (batch, layer_num, max_sent_len, max_sent_len)

#     ## causal model output
#     repr_encoded = encoder_out["hidden_states"][-1]  # (batch_size, max_sent_len, dim)
#     attn_layers = encoder_out["attentions"]  # list of (batch, mh, max_sent_len, max_sent_len)

#     ## filter special token from sent ids (e.g., "<|begin_of_text|>"" in llama)
#     attn_layers = filter_attn_special_tokens(cfg.model.llm_id, attn_layers)

#     # print(repr_encoded.shape)
#     # print("!!")
#     # print(attn_layers[-1])
#     # print(attn_layers[-1].shape)
#     # print(wekljflwewf)

#     # attention scores pooling: averaging all layers
#     # print(len(attn_layers))
#     # print(welkfjlewf)
#     attn_target_layer = attn_layers[target_layer]  # the target layer of attention (e.g., MS-phi-3.5 consists of 32 layers, 32 mh)
#     attn_target_avg = torch.div(torch.sum(attn_target_layer, dim=1), attn_target_layer.shape[1])  # average all mh
#     attn = attn_target_avg

#     # print(attn.shape)
#     # print(attn_last_avg)
#     # print("--")

#     # print(repr_encoded.shape)
#     # print(attn.shape)
#     # print(ewlkfjlkwef)

#     # encoded_sents = torch.stack(encoded_sents, dim=0).transpose(1, 0)
#     # attn_sents = torch.stack(attn_sents, dim=0).transpose(1, 0)

#     # batch_size = 4
#     # len_seq = 13
#     # attn = torch.randn(batch_size, len_seq, len_seq)  # assume attention wegiths are given, ignore layers


#     #### stage 2: get the entity index information using external linguistic parser
#     if not cache_entity:
#         # list_cur_sent = [cur_sent]
#         output_np_parser = np_parser.get_np_index_subwords(cur_sent)  # [("The important of education", (0, 3), ("our world of 21st centry", (15, 20)), ...]

#         if output_np_parser is not None:
#             list_np_sbw_loc = output_np_parser["np_sbw_loc"]

#             is_parsed_all = output_np_parser["is_parsed_all"]
#         else:
#             is_parsed_all = False

#     # print(list_np_sbw_loc)

#     #### stage 3: get attnetion interactions
#     outputs = {}
#     outputs["is_parsed_all"] = is_parsed_all

#     if is_parsed_all:
#         val_type1, val_type2, val_type3 = attn_flow_entity(cfg.model.llm_id, attn, list_np_sbw_loc)
    
#         outputs["val_type1"] = val_type1
#         outputs["val_type2"] = val_type2
#         outputs["val_type3"] = val_type3
#         outputs["list_np_sbw_loc"] = list_np_sbw_loc
    
#     return outputs

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

    ##### check cache for entity parsing
    np_parser = NP_Parser_BackT(tokenizer=tokenizer, encoder_weights=cfg.model.llm_id)

    list_doc_np_sbw_loc = []  # cahcing for all entity parsing results
    list_doc_vp_sbw_loc = []
    cache_entity = False

    #######################################
    ### caching 
    # output_dir = "cache_entity_parsing"
    # sub_dir = "benepar3"
    # # Path(os.path.join(output_dir, sub_dir)).mkdir(parents=True, exist_ok=True)
    # output_name = cfg.model.llm_id
    # output_name = output_name.replace("/", "-")

    # path_cache_file = os.path.join(output_dir, sub_dir, output_name)
    # if os.path.exists(path_cache_file):
    #     with open(path_cache_file) as f:
    #         list_doc_np_sbw_loc = json.load(f)
    #     cache_entity = True
    ########################################

    ####
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

    list_num_entities = []  # the number of entities for each sentence

    # sent_corpus = test_ds  # only consider test partition

    # for ind_doc in range(len(sent_corpus)):  # in this case, the document and the sentence are the same
    ind_row = 0
    for _, row in sent_corpus.iterrows():

        if cache_entity:    list_tags_sbw_loc = list_doc_np_sbw_loc[ind_row]
        else:   list_tags_sbw_loc = []  # caching for entity parsing for a document

        # cur_sent = sent_corpus[ind_doc]["text"]
        # cur_label = sent_corpus[ind_doc]["label"]
        # # chat_curr = sentiment_template.format(conversation=cur_user_response)

        cur_sent = row["text"]
        cur_label = row["label"]

        # debugging
        # if ind_doc < 1277:
            # continue
        # logger.info(f"{ind_doc}")
        # print(ind_doc)

        cur_sent = filter_sentence(cur_sent)
        
        ## in the case of too short: only consider a sentecne which has 3 words at least
        # print(len(cur_sent))
        if len(cur_sent.split()) < 5:
            num_sents_short += 1
            continue

        # print(cur_sent)

        ## calculate entity attention
        output_attn_flow = ent_attn_flow.run_attn_flow(cfg, tokenizer, encoder, np_parser, cur_sent, list_tags_sbw_loc,
                                        top_k=top_k, target_layer=target_layer, cache_entity=cache_entity)

        ## output
        is_parsed_all = output_attn_flow["is_parsed_all"] 
        is_vp_parsed_all = output_attn_flow["is_vp_parsed_all"]

        if is_parsed_all and is_vp_parsed_all:
            ent_top_k = output_attn_flow["ent_top_k"]
            vp_top_k = output_attn_flow["vp_top_k"]

            aflow_type1 = output_attn_flow["val_type1"]
            aflow_type2 = output_attn_flow["val_type2"]
            aflow_type3 = output_attn_flow["val_type3"]
            aflow_type4 = output_attn_flow["val_type4"]
            list_np_sbw_loc = output_attn_flow["list_np_sbw_loc"]  # used for caching entity parsing
            list_vp_sbw_loc = output_attn_flow["list_vp_sbw_loc"]

            list_num_entities.append(len(list_np_sbw_loc))
            
            seq_len = output_attn_flow["seq_len_subwords"]
            seq_len_ent = output_attn_flow["len_subwords_ent"]
            seq_len_vp = output_attn_flow["len_subwords_vp"]

            list_len_subwords.append(seq_len)
            list_len_subwords_ent.append(seq_len_ent)
            list_len_subwords_vp.append(seq_len_vp)
            
            ## save for the doc
            if not math.isnan(ent_top_k):   list_val_ent_topk.append(ent_top_k)
            if not math.isnan(vp_top_k):   list_val_vp_topk.append(vp_top_k)

            if not math.isnan(aflow_type1):    list_val_interact_type1.append(aflow_type1)
            if not math.isnan(aflow_type2):    list_val_interact_type2.append(aflow_type2)
            if not math.isnan(aflow_type3):    list_val_interact_type3.append(aflow_type3)
            if not math.isnan(aflow_type4):    list_val_interact_type4.append(aflow_type4)

            list_doc_np_sbw_loc.append(list_np_sbw_loc)
            list_doc_vp_sbw_loc.append(list_vp_sbw_loc)

            num_sents_considered += 1

        else:
            num_sents_corrupted += 1

        if len(list_np_sbw_loc) < 1:    num_sents_non_ent += 1
        if len(list_vp_sbw_loc) < 1:    num_sents_no_vp += 1
        
        ####

        # # !! debugging
        # if ind_doc > 50:
        #     break

        ## sampling
        if cfg.exp_args.sample_num != -1 and ind_row > cfg.exp_args.sample_num:
            break

        if ind_row % 1000 == 0:
            logger.info(f"Document: {ind_row}")
        
        ind_row += 1
    
    # num_docs = len(sent_corpus)
    # logger.info(f"# of Sentences without entities: {num_sents_non_ent}")
    # ratio_corrupted = num_sents_corrupted * 100 / float(num_docs)
    # logger.info(f"# of Sentences not properly parsed: {num_sents_corrupted} ({ratio_corrupted:.2f})")
    # ratio_short = num_sents_short * 100 / float(num_docs)
    # logger.info(f"# of Sentences not considered due to short: {num_sents_short} ({ratio_short:.2f})")
    # logger.info("")

    # num_sents_considered = num_docs - num_sents_non_ent - num_sents_corrupted - num_sents_short
    # ratio_considered = num_sents_considered * 100 / float(num_docs)
    # logger.info(f"# of Sentences considered in total: {num_sents_considered} ({ratio_considered:.2f})")
    # logger.info("")

    # ########################################################################
    # # stat analysis

    # # print(list_val_interact_type1)
    # # print(list_val_interact_type2)
    # # print(list_val_interact_type3)

    # outputs = get_avg_from_lists(list_val=list_val_interact_type1)
    # avg_sents_type1, std_sents_type1 = float(outputs[0]), float(outputs[1])
    # logger.info(f"Type1 Avg(Std): {avg_sents_type1} ({std_sents_type1})" )

    # outputs = get_avg_from_lists(list_val=list_val_interact_type2)
    # avg_sents_type2, std_sents_type2 = float(outputs[0]), float(outputs[1])
    # logger.info(f"Type1 Avg(Std): {avg_sents_type2} ({std_sents_type2})" )

    # outputs = get_avg_from_lists(list_val=list_val_interact_type3)
    # avg_sents_type3, std_sents_type3 = float(outputs[0]), float(outputs[1])
    # logger.info(f"Type1 Avg(Std): {avg_sents_type3} ({std_sents_type3})" )

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
    # num_sents_considered = num_sents_all - num_sents_non_ent - num_sents_corrupted - num_sents_short
    ratio_considered = num_sents_considered * 100 / float(num_sents_all)
    logger.info(f"# of sentences considered in total: {num_sents_considered} ({ratio_considered:.2f}%)")  # in terms of entity
    logger.info("")

    logger.info(f"# of sentences without VP: {num_sents_no_vp} ({num_sents_no_vp/float(num_sents_all):.2f}%")
    ratio_corrupted_vp = num_sents_corrupted_vp * 100 / float(num_sents_all)
    logger.info(f"# of sentences not properly VP parsed: {num_sents_corrupted_vp} ({ratio_corrupted_vp:.2f})")
    logger.info("")

    ########################################################################
    # stat analysis

    # print(list_val_interact_type1)
    # print(list_val_interact_type2)
    # print(list_val_interact_type3)

    #### NP Printing

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

    #######

    outputs = []
    outputs.append(list_val_interact_type1)
    outputs.append(list_val_interact_type2)
    outputs.append(list_val_interact_type3)

    outputs.append((avg_sents_type1, std_sents_type1))
    outputs.append((avg_sents_type2, std_sents_type2))
    outputs.append((avg_sents_type3, std_sents_type3))

    return outputs

######################################

def naive_inference(cfg, dataset_test, encoder, tokenizer, messages):

    preds, labels = inference_sentiment(dataset_test, cfg.model.llm_id, encoder, tokenizer, messages)
    # print(preds)
    # print(labels)
    # print("")
    # print(ewkljfeklw)

    preds_discrete = convert_label_list_int(preds)
    acc = eval_accuracy(preds_discrete, labels)
    kappa_el_linear, kappa_el_quad = eval_kappa(preds_discrete, labels)

    logger.info("Evaluation: Before PEFT ----")
    logger.info("Accuracy: %.2f" % (acc))
    # print("Kappa Ne: %.2f %.2f" % (kappa_ne_linear, kappa_ne_quad))
    logger.info("Kappa El: %.2f %.2f" % (kappa_el_linear, kappa_el_quad))
    logger.info("-------")
    logger.info("")

    hist_preds_peft, hist_labels_peft = get_hist_preds_labels(preds_discrete, labels)
    logger.info(hist_preds_peft)
    logger.info(hist_labels_peft)

    hist_preds_peft = fill_empty_label(hist_preds_peft)

    plot_preds_labels_dists(hist_preds_peft, hist_labels_peft, "naive", is_equal_label_dist=True)


####################################
#### layer analysis

def ent_attn_layer_analysis(cfg, encoder, tokenizer, sent_corpus, max_layer=32):

    for target_layer in range(max_layer):
        logger.info(f"Target Layer: {target_layer+1}")
        output_ent_attn_flow = wrapper_ent_attn_flow(cfg, encoder, tokenizer, sent_corpus, cfg.exp_args.top_k, target_layer)

    # # file write
    # output_dir = "log_ent_attn_layers"
    # sub_dir = cfg.model.llm_id
    # sub_dir = sub_dir.replace("/", "-")
    # Path(os.path.join(output_dir, sub_dir)).mkdir(parents=True, exist_ok=True)

    # output_name = "type1.log"
    # with open(os.path.join(output_dir, sub_dir, output_name), 'w') as f:
    #     json.dump(list_avg_type1, f)

    return


######################################
#### datasets

def load_dataset_toefl(cfg, num_samples=0):

    cur_fold = 0
    str_cur_fold = str(cur_fold)

    #### load dataset from files
    train_pd = pd.read_csv(os.path.join(cfg.dataset.path_data, "sst_train_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c', index_col=0)
    valid_pd = pd.read_csv(os.path.join(cfg.dataset.path_data, "sst_valid_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c', index_col=0)
    test_pd = pd.read_csv(os.path.join(cfg.dataset.path_data, "sst_test_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c', index_col=0)

    # print(train_pd.head())
    # print(train_pd.columns)

    # extract only the essays in the target prompt (1 to 8)
    train_pd = train_pd.loc[train_pd['prompt'] == cfg.dataset.target_prompt]
    valid_pd = valid_pd.loc[valid_pd['prompt'] == cfg.dataset.target_prompt]
    test_pd = test_pd.loc[test_pd['prompt'] == cfg.dataset.target_prompt]

    # sampling
    if num_samples > 0:
        train_pd = train_pd[:num_samples]
        valid_pd = valid_pd[:num_samples]
        test_pd = test_pd[:num_samples]

    # merge them all
    total_pd = pd.concat([train_pd, valid_pd, test_pd], sort=True)
    # only extract documents
    total_corpus = total_pd['essay'].values

    #### sentence tokenizing

    # first, 

    # use linguistic parser
    import stanza  # stanford library for tokenizer
    tokenizer_stanza = stanza.Pipeline('en', processors='tokenize', use_gpu=True)

    # doc_stanza = self.stanza_pipeline(cur_text)
    # tokenized_sents = [sentence.text for sentence in doc_stanza.sentences]  # convert to list of list

    num_sents = []
    sent_corpus = []  # tokenized to form of [doc, list of sentences]
    for cur_doc in total_corpus:

        ## stanza version
        doc_stanza = tokenizer_stanza(cur_doc)
        sent_list = [sentence.text for sentence in doc_stanza.sentences]
        
        sent_corpus.append(sent_list)
        num_sents.append(len(sent_list))


    return sent_corpus, num_sents

################################
################

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:

    #### stage 1: encode sentence

    # load tokenizer and encoder
    # encoder_weights = "susnato/phi-1_5_dev"
    # encoder_weights = "google/flan-t5-xl"
    # encoder_weights = "xlnet-base-cased"
    # encoder_weights = "google-bert/bert-base-uncased"
    # encoder_weights = "facebook/opt-350m"
    # encoder_weights = "facebook/opt-1.3b"
    # encoder_weights = "facebook/opt-2.7b"
    # encoder_weights = "microsoft/phi-2"

    # encoder_weights = "microsoft/Phi-3.5-mini-instruct"
    encoder_weights = "meta-llama/Llama-3.2-1B"
    # encoder_weights = "meta-llama/Llama-3.1-8B"
    # encoder_weights = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    # encoder_weights = "mistralai/Mistral-7B-Instruct-v0.3"
    # encoder_weights = "google/gemma-2-2b-it"

    hf_token = os.environ.get("HF_TOKEN") or OmegaConf.select(cfg, "hf_token") or None

    ## QLORA config

    # Activate 4-bit precision base model loading
    use_4bit = True
    # Activate nested quantization for 4-bit base models
    use_nested_quant = False
    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    # Enable fp16 training
    # fp16 = False
    fp16 = True
    # Enable bf16 training
    # bf16 = True
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
    
    # encoder_dtype = torch.float32

    # encoder = AutoModel.from_pretrained(cfg.model.llm_id, low_cpu_mem_usage=True, token=hf_token, offload_folder="offload", offload_state_dict = True, max_memory={"cpu": "10GIB"}, torch_dtype=encoder_dtype)
    encoder = AutoModelForCausalLM.from_pretrained(cfg.model.llm_id, low_cpu_mem_usage=True, token=hf_token, offload_folder="offload", offload_state_dict = True, max_memory={"cpu": "10GIB"}, torch_dtype=encoder_dtype)
    encoder.to("cuda")

    # encoder = AutoModel.from_pretrained(
    # encoder = AutoModelForCausalLM.from_pretrained(
    #     cfg.model.llm_id,
    #     quantization_config=bnb_config, #comment out to do a full fine-tune.
    #     device_map='auto',
    #     # attn_implementation="flash_attention_2", #comment out if you are not using an Ampere GPU (e.g. A100, H100, A6000).
    #     torch_dtype=compute_dtype, #set to torch.float16 if you are not using an Ampere GPU (e.g. A100, H100, A6000).
    #     )

    # # disabling graident updating for the encoder
    # for param in encoder.parameters():
    #     param.requires_grad = False
    
    # encoder.config.use_cache = False
    # encoder.config.pretraining_tp = 1

    # tokenized_seg_ids = []
    # tokenized_attn_mask = []

    logger.info(f"Target LLM:   {cfg.model.llm_id}")
    
    ###### load dataset -> list of list of sentences
    
    ## TOEFL dataset
    # sent_corpus, num_sents = load_dataset_toefl(cfg, num_samples=0)  # num_sample==0 -> no sampling, 2 => 2 for each sets

    ## SST5 (movie review)
    dataset_sst5 = load_dataset("SetFit/sst5")

    '''
        num texts: 8544 / 1101 / 2210
        num_labels: 5 (0:very negative / negative / 2:neutral / positive / 4:very positive)
        label distribution: 12.8% / 26% / 19% / 27.2% / 15.1%
    '''
    train_ds, valid_ds, test_ds = dataset_sst5["train"], dataset_sst5["validation"], dataset_sst5["test"]

    dataset_sampled_ne_train, dataset_sampled_equal_train = sample_dataset(train_ds, num_labels=5, sample_ratio=0.1, num_samples_label=1000)
    dataset_sampled_ne_test, dataset_sampled_equal_test = sample_dataset(test_ds, num_labels=5, sample_ratio=0.1, num_samples_label=200)
    # print(dataset_sampled_ne_test)
    # print(dataset_sampled_equal_test)

    # messages = [
    #     # {"role": "system", "content": "Classify the sentiment of conversation into 5 classes: very_positive, positive, neutral, negative, or very_negative. You must generate only one word of sentiment class"},
    #     {"role": "assistant", "content": "Hello, which one is the review to classify the sentiment?"},
    #     {"role": "user", "content": ""},
    # ]  # gemma does not support "system" role, so put into the user prompt

    messages = [
        # {"role": "system", "content": "Classify the sentiment of conversation into 5 classes: very_positive, positive, neutral, negative, or very_negative. You must generate only one word of sentiment class"},
        # {"role": "assistant", "content": "Hello, which one is the review to classify the sentiment?"},
        {"role": "user", "content": ""},
    ]  # gemma does not support "system" role, so put into the user prompt

    logger.info(f"Top_K:    {cfg.exp_args.top_k}")
    logger.info("")

    ##
    logger.info("----- Naive Inference ----- ")
    naive_inference(cfg, dataset_sampled_equal_test, encoder, tokenizer, messages)
    logger.info("")

    ## 
    train_pd = pd.DataFrame(dataset_sampled_equal_train)
    test_pd = pd.DataFrame(dataset_sampled_equal_test)
    sent_corpus = pd.concat([train_pd, test_pd])

    num_docs = len(sent_corpus)

    # logger.info(f"Total # of Documents: {num_docs}")
    logger.info(f"Total # of Sentences: {num_docs}")

    #####
    # logger.info(" ---- Layer Analysis ---- ")
    # ent_attn_layer_analysis(cfg, encoder, tokenizer, sent_corpus, max_layer=32)
    # print(weklfjweljwelf)

    # max_layer = 32
    # list_avg_type1 = []
    # list_std_type1 = []
    # list_avg_type2 = []
    # list_std_type2 = []
    # list_avg_type3 = []
    # list_std_type3 = []
    # for target_layer in range(max_layer):
    #     print(f"Target Layer: {target_layer}")
    #     output_ent_attn_flow = wrapper_ent_attn_flow(cfg, encoder, tokenizer, sent_corpus, target_layer)

    #     list_avg_type1.append(output_ent_attn_flow[3][0])
    #     list_std_type1.append(output_ent_attn_flow[3][1])
    #     list_avg_type2.append(output_ent_attn_flow[4][0])
    #     list_std_type2.append(output_ent_attn_flow[4][1])
    #     list_avg_type3.append(output_ent_attn_flow[5][0])
    #     list_std_type3.append(output_ent_attn_flow[5][1])

    # print("")

    # print(list_avg_type1)
    # print(list_avg_type2)
    # print(list_avg_type3)
    
    # print("")

    # print(list_std_type1)
    # print(list_std_type2)
    # print(list_std_type3)

    # # file write
    # output_dir = "log_ent_attn_layers"
    # sub_dir = cfg.model.llm_id
    # sub_dir = sub_dir.replace("/", "-")
    # Path(os.path.join(output_dir, sub_dir)).mkdir(parents=True, exist_ok=True)

    # output_name = "type1.log"
    # with open(os.path.join(output_dir, sub_dir, output_name), 'w') as f:
    #     json.dump(list_avg_type1, f)
    #     json.dump(list_std_type1, f)
    # output_name = "type2.log"
    # with open(os.path.join(output_dir, sub_dir, output_name), 'w') as f:
    #     json.dump(list_avg_type2, f)
    #     json.dump(list_std_type2, f)
    # output_name = "type3.log"
    # with open(os.path.join(output_dir, sub_dir, output_name), 'w') as f:
    #     json.dump(list_avg_type3, f)
    #     json.dump(list_std_type3, f)

    logger.info("------- Ent Attn Flow ------")
    output_ent_attn_flow = wrapper_ent_attn_flow(cfg, encoder, tokenizer, sent_corpus, cfg.exp_args.top_k, target_layer=-1)
    logger.info("")

    # print(ewlkfjewljelwf)

    ###################################################################################################3
    ###################################################################################################3
    #### peft

    logger.info("------- PEFT ------")
    encoder = peft_encoder(cfg.model.llm_id, encoder, tokenizer, dataset_sampled_equal_train, dataset_sampled_equal_test, messages)
    logger.info("")
    ##

    train_pd = pd.DataFrame(dataset_sampled_equal_train)
    test_pd = pd.DataFrame(dataset_sampled_equal_test)
    sent_corpus = pd.concat([train_pd, test_pd])

    num_docs = len(sent_corpus)

    # logger.info(f"Total # of Documents: {num_docs}")
    logger.info(f"Total # of Sentences: {num_docs}")

    logger.info("------- Ent Attn Flow ------")
    output_ent_attn_flow = wrapper_ent_attn_flow(cfg, encoder, tokenizer, sent_corpus, cfg.exp_args.top_k, target_layer=-1)
    
    list_val_interact_type1 = output_ent_attn_flow[0]
    list_val_interact_type2 = output_ent_attn_flow[1]
    list_val_interact_type3 = output_ent_attn_flow[2]


    # ########################################################################
    # # save to log files
    # output_dir = "log_entity_attn"
    # sub_dir = cfg.model.llm_id
    # sub_dir = sub_dir.replace("/", "-")
    # Path(os.path.join(output_dir, sub_dir)).mkdir(parents=True, exist_ok=True)

    # output_name = "type1.log"
    # with open(os.path.join(output_dir, sub_dir, output_name), 'w') as f:
    #     json.dump(list_val_interact_type1, f)
    # output_name = "type2.log"
    # with open(os.path.join(output_dir, sub_dir, output_name), 'w') as f:
    #     json.dump(list_val_interact_type2, f)
    # output_name = "type3.log"
    # with open(os.path.join(output_dir, sub_dir, output_name), 'w') as f:
    #     json.dump(list_val_interact_type3, f)
    
    # # read json
    # with open(os.path.join(output_dir, sub_dir, output_name)) as f:
    #     d = json.load(f)
    #     print(d)

    # ## cache entity parsing results
    # output_dir = "cache_entity_parsing"
    # sub_dir = "benepar3"
    # Path(os.path.join(output_dir, sub_dir)).mkdir(parents=True, exist_ok=True)
    # output_name = cfg.model.llm_id
    # output_name = output_name.replace("/", "-")
    # path_cache_file = os.path.join(output_dir, sub_dir, output_name)
    # if not os.path.exists(path_cache_file):
    #     with open(os.path.join(output_dir, sub_dir, output_name), 'w') as f:
    #         json.dump(list_doc_np_sbw_loc, f)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    ptime = (end - start) / 60.0 # minutes
    # print(f"Processing Time: {ptime:.2f} mins")
    logging.info(f"Processing Time: {ptime:.2f} mins")
