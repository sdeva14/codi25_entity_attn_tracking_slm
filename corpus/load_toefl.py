"""Load TOEFL-style CSV data and split into sentences with Stanza."""
import os
import pandas as pd


def load_dataset_toefl(
    path_data: str,
    num_samples: int = 0,
    filter_key: str = "essay_score",
    filter_value: int = 2,
    cur_fold: int = 0,
):
    """
    Load train/valid/test CSVs (sst_*_fold_*.csv), filter by column, concat, sentence-split with Stanza.
    path_data: base path to CSV folds (overridden by DATA_PATH env if set).
    filter_key: e.g. "essay_score" (attn_flow_test) or "prompt" (ft with TOEFL).
    filter_value: value to keep.
    Returns (sent_corpus, num_sents): list of list of sentences per doc, and list of sentence counts.
    """
    path_data = os.environ.get("DATA_PATH") or path_data
    str_fold = str(cur_fold)
    train_pd = pd.read_csv(
        os.path.join(path_data, "sst_train_fold_" + str_fold + ".csv"),
        sep=",",
        header=0,
        encoding="utf-8",
        engine="c",
        index_col=0,
    )
    valid_pd = pd.read_csv(
        os.path.join(path_data, "sst_valid_fold_" + str_fold + ".csv"),
        sep=",",
        header=0,
        encoding="utf-8",
        engine="c",
        index_col=0,
    )
    test_pd = pd.read_csv(
        os.path.join(path_data, "sst_test_fold_" + str_fold + ".csv"),
        sep=",",
        header=0,
        encoding="utf-8",
        engine="c",
        index_col=0,
    )
    train_pd = train_pd.loc[train_pd[filter_key] == filter_value]
    valid_pd = valid_pd.loc[valid_pd[filter_key] == filter_value]
    test_pd = test_pd.loc[test_pd[filter_key] == filter_value]
    if num_samples > 0:
        train_pd = train_pd[:num_samples]
        valid_pd = valid_pd[:num_samples]
        test_pd = test_pd[:num_samples]
    total_pd = pd.concat([train_pd, valid_pd, test_pd], sort=True)
    total_corpus = total_pd["essay"].values
    import stanza
    tokenizer_stanza = stanza.Pipeline("en", processors="tokenize", use_gpu=True)
    num_sents = []
    sent_corpus = []
    for cur_doc in total_corpus:
        doc_stanza = tokenizer_stanza(cur_doc)
        sent_list = [sentence.text for sentence in doc_stanza.sentences]
        sent_corpus.append(sent_list)
        num_sents.append(len(sent_list))
    return sent_corpus, num_sents
