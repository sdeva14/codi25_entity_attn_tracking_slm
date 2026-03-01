"""
PEFT fine-tuning (LoRA) on SST-5 + entity attention flow evaluation.
Run from project root. Uses SetFit/sst5 from Hugging Face; no local TOEFL data required.
"""
import logging
import os
import time

import pandas as pd
import torch
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as tf_logging
from transformers import TrainingArguments

tf_logging.set_verbosity_error()

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

import datasets
from datasets import load_dataset

from entity_parser.np_parser_backt import NP_Parser_BackT
import ent_attn_func.attn_flow as ent_attn_flow
from ent_attn_func.attn_flow_runner import (
    MIN_WORDS_SENTENCE,
    AttnFlowAggregator,
)
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

# PEFT / training defaults
OUTPUT_TRAIN_DIR = "hf_train_results_lora"
PAD_TOKEN_FINETUNE = "<|finetune_right_pad_id|>"
LORA_R = 8
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
TRAIN_EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
LOG_PROGRESS_EVERY = 1000


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
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
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
    kappa_quad = (
        cohen_kappa_score(preds, labels, weights="quadratic") if is_quad_also else 0.0
    )
    return kappa_linear, kappa_quad


def sample_dataset(
    curr_dataset,
    num_labels=5,
    sample_ratio=0.1,
    num_samples_label=40,
):
    sample_num = round(len(curr_dataset) * sample_ratio)
    list_indices = list(range(sample_num))
    dataset_sampled_ne = curr_dataset.select(list_indices)
    import pandas as pd
    df_dataset = pd.DataFrame(curr_dataset)
    list_pd_labels = []
    for i in range(num_labels):
        pd_label_i = df_dataset.loc[df_dataset["label"] == i]
        list_pd_labels.append(pd_label_i)
    for i, curr_pd in enumerate(list_pd_labels):
        if len(curr_pd) > num_samples_label:
            list_pd_labels[i] = curr_pd[:num_samples_label]
        else:
            logger.info("num samples per label is larger than each label frame")
    merged_equal_pd = pd.concat(list_pd_labels)
    dataset_sampled_equal = datasets.Dataset.from_pandas(merged_equal_pd)
    return dataset_sampled_ne, dataset_sampled_equal


def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params
    logger.info(f"Trainable Parameters: {trainable_params}")
    logger.info(f"Total Parameters: {total_params}")
    logger.info(f"Trainable %: {trainable_percent:.2f}")


def run_entity_attn_flow_flat(
    cfg,
    encoder,
    tokenizer,
    sent_corpus,
    top_k=5,
    target_layer=-1,
):
    """
    Run entity attention flow over a flat corpus (DataFrame with 'text' column).
    Returns results list: [list_type1, list_type2, list_type3, list_type4, (avg1,std1), ...].
    """
    np_parser = NP_Parser_BackT(tokenizer=tokenizer, encoder_weights=cfg.model.llm_id)
    cache_entity = False
    aggregator = AttnFlowAggregator.for_flat_corpus()
    sample_num = getattr(cfg.exp_args, "sample_num", -1)

    for ind_row, (_, row) in enumerate(sent_corpus.iterrows()):
        list_tags_sbw_loc = (
            aggregator.list_doc_np_sbw_loc[ind_row] if cache_entity else []
        )
        cur_sent = filter_sentence(row["text"])
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
            top_k=top_k,
            target_layer=target_layer,
            cache_entity=cache_entity,
        )
        list_np_sbw_loc = output_attn_flow.get("list_np_sbw_loc", [])
        list_vp_sbw_loc = output_attn_flow.get("list_vp_sbw_loc", [])
        aggregator.update(output_attn_flow, list_np_sbw_loc, list_vp_sbw_loc)

        if sample_num != -1 and ind_row > sample_num:
            break
        if ind_row % LOG_PROGRESS_EVERY == 0:
            logger.info(f"Document: {ind_row}")

    num_sents_all = len(sent_corpus)
    aggregator.log_summary(num_sents_all, logger)
    return aggregator.build_results()


def peft_encoder(llm_id, encoder, tokenizer, train_dataset, test_dataset, messages):
    train_dataset = train_dataset.map(convert_label_str)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    encoder.gradient_checkpointing_enable()
    encoder = prepare_model_for_kbit_training(encoder)
    encoder = get_peft_model(encoder, config)
    print_trainable_parameters(encoder)

    dataset_sft_train = convert_msgs_format(train_dataset)
    logger.info(dataset_sft_train)
    logger.info(dataset_sft_train[0])
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN_FINETUNE})

    sub_dir = llm_id.replace("/", "-")
    Path(os.path.join(OUTPUT_TRAIN_DIR, sub_dir)).mkdir(parents=True, exist_ok=True)
    training_arguments = TrainingArguments(
        output_dir=os.path.join(OUTPUT_TRAIN_DIR, sub_dir),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        bf16=True,
        fp16=False,
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
        packing=False,
    )
    trainer.train()

    preds_peft, labels_peft = inference_sentiment(
        test_dataset, llm_id, encoder, tokenizer, messages
    )
    preds_peft = clean_generated_sentiment_class(preds_peft)
    preds_discrete_peft = convert_label_list_int(preds_peft)
    acc = eval_accuracy(preds_discrete_peft, labels_peft)
    kappa_el_linear, kappa_el_quad = eval_kappa(preds_discrete_peft, labels_peft)
    logger.info("Accuracy: %.2f" % acc)
    logger.info("Kappa El: %.2f %.2f" % (kappa_el_linear, kappa_el_quad))
    logger.info("-------")
    logger.info("")

    hist_preds_peft, hist_labels_peft = get_hist_preds_labels(
        preds_discrete_peft, labels_peft
    )
    logger.info(hist_preds_peft)
    logger.info(hist_labels_peft)
    hist_preds_peft = fill_empty_label(hist_preds_peft)
    plot_preds_labels_dists(hist_preds_peft, hist_labels_peft, "lora", is_equal_label_dist=True)

    return encoder


def naive_inference(cfg, dataset_test, encoder, tokenizer, messages):
    preds, labels = inference_sentiment(
        dataset_test, cfg.model.llm_id, encoder, tokenizer, messages
    )
    preds_discrete = convert_label_list_int(preds)
    acc = eval_accuracy(preds_discrete, labels)
    kappa_el_linear, kappa_el_quad = eval_kappa(preds_discrete, labels)
    logger.info("Evaluation: Before PEFT ----")
    logger.info("Accuracy: %.2f" % acc)
    logger.info("Kappa El: %.2f %.2f" % (kappa_el_linear, kappa_el_quad))
    logger.info("-------")
    logger.info("")
    hist_preds_peft, hist_labels_peft = get_hist_preds_labels(preds_discrete, labels)
    logger.info(hist_preds_peft)
    logger.info(hist_labels_peft)
    hist_preds_peft = fill_empty_label(hist_preds_peft)
    plot_preds_labels_dists(
        hist_preds_peft, hist_labels_peft, "naive", is_equal_label_dist=True
    )


def ent_attn_layer_analysis(cfg, encoder, tokenizer, sent_corpus, max_layer=32):
    for target_layer in range(max_layer):
        logger.info(f"Target Layer: {target_layer + 1}")
        run_entity_attn_flow_flat(
            cfg, encoder, tokenizer, sent_corpus, cfg.exp_args.top_k, target_layer
        )


def load_encoder_and_tokenizer(cfg: DictConfig):
    hf_token = os.environ.get("HF_TOKEN") or OmegaConf.select(cfg, "hf_token") or None
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.llm_id, low_cpu_mem_usage=True, token=hf_token
    )
    encoder_dtype = torch.bfloat16
    if "xlnet" in cfg.model.llm_id or "google-bert" in cfg.model.llm_id:
        encoder_dtype = "auto"
    encoder = AutoModelForCausalLM.from_pretrained(
        cfg.model.llm_id,
        low_cpu_mem_usage=True,
        token=hf_token,
        offload_folder="offload",
        offload_state_dict=True,
        max_memory={"cpu": "10GIB"},
        torch_dtype=encoder_dtype,
    )
    encoder.to("cuda")
    return tokenizer, encoder


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    tokenizer, encoder = load_encoder_and_tokenizer(cfg)
    logger.info(f"Target LLM:   {cfg.model.llm_id}")

    dataset_sst5 = load_dataset("SetFit/sst5")
    train_ds, valid_ds, test_ds = (
        dataset_sst5["train"],
        dataset_sst5["validation"],
        dataset_sst5["test"],
    )
    _, dataset_sampled_equal_train = sample_dataset(
        train_ds, num_labels=5, sample_ratio=0.1, num_samples_label=1000
    )
    _, dataset_sampled_equal_test = sample_dataset(
        test_ds, num_labels=5, sample_ratio=0.1, num_samples_label=200
    )
    messages = [{"role": "user", "content": ""}]

    logger.info(f"Top_K:    {cfg.exp_args.top_k}")
    logger.info("")
    logger.info("----- Naive Inference ----- ")
    naive_inference(cfg, dataset_sampled_equal_test, encoder, tokenizer, messages)
    logger.info("")

    sent_corpus = pd.concat(
        [pd.DataFrame(dataset_sampled_equal_train), pd.DataFrame(dataset_sampled_equal_test)]
    )
    logger.info(f"Total # of Sentences: {len(sent_corpus)}")
    logger.info("------- Ent Attn Flow ------")
    run_entity_attn_flow_flat(
        cfg, encoder, tokenizer, sent_corpus, cfg.exp_args.top_k, target_layer=-1
    )
    logger.info("")

    logger.info("------- PEFT ------")
    encoder = peft_encoder(
        cfg.model.llm_id,
        encoder,
        tokenizer,
        dataset_sampled_equal_train,
        dataset_sampled_equal_test,
        messages,
    )
    logger.info("")

    sent_corpus = pd.concat(
        [pd.DataFrame(dataset_sampled_equal_train), pd.DataFrame(dataset_sampled_equal_test)]
    )
    logger.info(f"Total # of Sentences: {len(sent_corpus)}")
    logger.info("------- Ent Attn Flow ------")
    run_entity_attn_flow_flat(
        cfg, encoder, tokenizer, sent_corpus, cfg.exp_args.top_k, target_layer=-1
    )


if __name__ == "__main__":
    start = time.time()
    main()
    logging.info(f"Processing Time: {(time.time() - start) / 60:.2f} mins")
