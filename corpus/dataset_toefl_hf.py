import os
import datasets
import pandas as pd

class Dataset_TOEFL():
    def __init__(self, tokenizer, cfg):

        self.dataset_name = "TOEFL"  # for logging

        import stanza
        self.stanza_pipeline = stanza.Pipeline('en', processors="tokenize", use_gpu=True)
        self.tokenizer = tokenizer  # pretrained tokenizer for the models

        self.output_size = 3  # 3-class classification

        self.pretrained_weights = cfg.model.pretrained_weights

        self.max_num_sent = 0
        self.max_len_sent = 0  # max length of a sentence in the whole dataset (w.r.t input_ids) -> will be set in the tokenize_map

        self.filter_low_sent_num = 2  # filter texts which have sentences lower than this number
        self.len_force_sent_trunc = 50  # w.r.t words, not subwords

        self.len_segment_trunc = cfg.dataset.len_segment_trunc  # parameter to tokenize texts into segments, w.r.t. words

        return
    
    def filter_special_tokens(self, tokenized, idx_sent):
        """Filter special tokens from per-sentence tokenization (model-specific)."""
        sent_ids = tokenized["input_ids"]
        attn_mask = tokenized["attn_mask"]
        tokenized["input_ids"] = sent_ids
        tokenized["attn_mask"] = attn_mask
        return tokenized

    def tokenize_map_length(self, sample):
        """Map: tokenize text by fixed-length segments (e.g. every len_segment_trunc words)."""
        text = sample["essay"]
        tokenized_seg_ids = []
        tokenized_attn_mask = []
        max_len_seg = 0  # max length of a sentence in the whole dataset
        # print("----")
        # print(text)

        ## segmentation by the given length
        words = text.split()
        splitted = []
        if len(words) > self.len_force_sent_trunc:
            for loc in range(0, len(words), self.len_segment_trunc):
                curr = words[loc:loc+self.len_segment_trunc]
                curr = " ".join(curr)
                splitted.append(curr)
        else:
            splitted.append(text)

        for curr in splitted:            
            ## tokenize
            tokenized = self.tokenizer(curr)
            tokenized_seg_ids.append(tokenized["input_ids"])
            tokenized_attn_mask.append(tokenized["attention_mask"])

            cur_len_sent = len(tokenized["input_ids"])
            if cur_len_sent > max_len_seg:
                    max_len_seg = cur_len_sent
        
        if max_len_seg > self.max_len_sent:
            self.max_len_sent = max_len_seg

        #     print(curr)
        #     print(tokenized)
        # print(ewklfjwelew)

        tokenized = {"input_ids": tokenized_seg_ids, "attention_mask": tokenized_attn_mask}


        return tokenized

    def tokenize_map_sent(self, sample):
        """Map: tokenize pre-segmented essay_sents (list of sentences)."""
        sents_list = eval(sample["essay_sents"])
        tokenized_sents_ids = []
        tokenized_attn_mask = []
        max_len_sent = 0  # max length of a sentence in the whole dataset
        max_text = ""
        for idx_sent, sent in enumerate(sents_list):
            tokenized = self.tokenizer(sent)
            tokenized_sents_ids.append(tokenized["input_ids"])
            tokenized_attn_mask.append(tokenized["attention_mask"])

            cur_len_sent = len(tokenized["input_ids"])
            if cur_len_sent > max_len_sent:
                max_len_sent = cur_len_sent
                max_text = sent

        tokenized = {"input_ids": tokenized_sents_ids, "attention_mask": tokenized_attn_mask}
        
        # self.max_len_sent = max_len_sent if max_len_sent > self.max_len_sent else self.max_len_sent

        if max_len_sent > self.max_len_sent:
            self.max_len_sent = max_len_sent
            # print("--")
            # print(max_text)
            # print(self.max_len_sent)

        # print("--")
        # # print(self.max_len_sent)
        # print(max_len_sent)
        # print(max_text)
        # print(ewklfjwklewf)

        return tokenized
    
    def load_hf_dataset(self, path_data, target_prompt, cur_fold, tokenize_method):
        '''
            load a dataset converted into HF style
        '''
        str_cur_fold = str(cur_fold)

        train_pd = pd.read_csv(os.path.join(path_data, "sst_train_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c', index_col=0)
        valid_pd = pd.read_csv(os.path.join(path_data, "sst_valid_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c', index_col=0)
        test_pd = pd.read_csv(os.path.join(path_data, "sst_test_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c', index_col=0)

        # print(train_pd.head())
        # print(train_pd.columns)

        # extract only the essays in the target prompt (1 to 8)
        train_pd = train_pd.loc[train_pd['prompt'] == target_prompt]
        valid_pd = valid_pd.loc[valid_pd['prompt'] == target_prompt]
        test_pd = test_pd.loc[test_pd['prompt'] == target_prompt]

        # # sampling
        # num_sample = 10
        # train_pd = train_pd[:num_sample]
        # valid_pd = valid_pd[:num_sample]
        # test_pd = test_pd[:num_sample]

        # total dataset including both train and test -> for debugging or analysis
        total_pd = pd.concat([self.train_pd, self.valid_pd, self.test_pd], sort=True)

        ## converting to HF datasets
        dataset_hf = datasets.DatasetDict({"train": datasets.Dataset.from_pandas(train_pd), 
                                           "validate": datasets.Dataset.from_pandas(valid_pd),
                                           "test": datasets.Dataset.from_pandas(test_pd),
                                           
                                           "total": datasets.Dataset.from_pandas(total_pd)}) 
        
        tokenized_dataset = None
        if tokenize_method == "sent":
            tokenized_dataset = dataset_hf.map(self.tokenize_map_sent, batched=False)
        elif tokenize_method == "length":
            tokenized_dataset = dataset_hf.map(self.tokenize_map_length, batched=False)
        else: raise Exception("Not defined tokenized method")

        # print(tokenized_dataset)
        # print(tokenized_dataset["train"]["input_ids"][:1])
        # print(ewlkjelwf)

        return tokenized_dataset

    def tokenize_sent(self, cur_text):

        doc_stanza = self.stanza_pipeline(cur_text)
        tokenized_sents = [sentence.text for sentence in doc_stanza.sentences]  # convert to list of list
        # print(tokenized_sents)

        return tokenized_sents

    def filter_single_sent_text(self, data_pd):

        filtered = data_pd[data_pd['sent_num'] > self.filter_low_sent_num] 

        print(len(data_pd))
        print(len(filtered))

        '''
        origianl vs filtered num (threshold: 2)
        7265
        7229
        2418
        2406
        2417
        2407
        '''

        return filtered
    
    def force_split_sent(self, sent):
        '''
        it is a brutal method to split a sentence, which couldn't be splitted by linguistic libraries
        it is needed to deal with practical constraints, GPU memory to encode
        '''

        splitted = []
        words = sent.split()
        
        for loc in range(0, len(words), self.len_force_sent_trunc):
            curr = words[loc:loc+self.len_force_sent_trunc]
            curr = " ".join(curr)
            splitted.append(curr)

        return splitted

    def sentence_toeknieze(self, data_pd):
        list_sents = []
        list_num_sents = []
        for index, row in data_pd.iterrows():
            essay = row["essay"]
            sents = self.tokenize_sent(essay)

            # additional check (Stanza 1.5(6) library does not consider newline character or other spaces)
            p_sents = []
            for sent in sents:
                p_sents.extend(sent.splitlines()) 
            
            # chunk each sentence by length (due to GPU memory usage to encdoe, otherwise it cannot be done)
            # Note that this force chunking will destroy syntax of a sentence, but it is inevitable to deal with large LLMs
            p2_sents = []
            for sent in p_sents:
                curr = [sent]
                if len(sent) > self.len_force_sent_trunc: curr = self.force_split_sent(sent)
                p2_sents.extend(curr)

            list_sents.append(p2_sents)
            list_num_sents.append(len(p2_sents))
        
        # add a new column of tokenized essay sentences (to save time for every run later)
        data_pd["essay_sents"] = list_sents
        data_pd["sent_num"] = list_num_sents

        return data_pd

    def tokenize_convert_HF_dataset(self, path_data, data_pd, split, str_cur_fold="0"):
        '''
            tokenize input text into sentences, then convert into HF dataset, finally save into the disk
        '''

        # list_sents = []
        # for index, row in data_pd.iterrows():
        #     essay = row["essay"]
        #     sents = self.tokenize_sent(essay)
        #     list_sents.append(sents)
        
        # # add a new column of tokenized essay sentences (to save the time for every run later)
        # data_pd["essay_sents"] = list_sents
        # # map_train = data_pd.set_index("essay_id").T.to_dict()  # key: essay_id, values: prompt, native_lang, essay_score, essay, essay_sents        

        data_pd = self.sentence_toeknieze(data_pd)

        # convert it into the HuggingFace dataset style to deal with it easily with others
        ds = datasets.Dataset.from_pandas(data_pd)
        # print(ds)

        # ds.save_to_disk(os.path.join(path_data, "toefl_" + str_cur_fold + "_hf", split))  # this approach does not support iteratable dataset, hence use JSON
        ds.to_json(os.path.join(path_data, "hf_" + split + "_toefl_" + str_cur_fold + ".json"))

        # datasets.concatenate_datasets

        return ds
    
    def load_and_convert_HF_dataset(self, path_data, str_cur_fold="0"):
        '''
            load a raw dataset and convert it to the HF dataset format
        '''

        self.train_pd = pd.read_csv(os.path.join(path_data, "train_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c')
        self.valid_pd = pd.read_csv(os.path.join(path_data, "valid_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c')
        self.test_pd = pd.read_csv(os.path.join(path_data, "test_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c')

        self.tokenize_convert_HF_dataset(path_data, self.train_pd, split="train")
        self.tokenize_convert_HF_dataset(path_data, self.valid_pd, split="valid")
        self.tokenize_convert_HF_dataset(path_data, self.test_pd, split="test")

        # save: pd.to_csv

        return

    def tokenize_sents_save_pd(self, path_data, cur_fold="0", force_sent_tokenize=False):
        '''
            load a raw dataset and convert it to the HF dataset format
        '''

        str_cur_fold = str(cur_fold)

        if force_sent_tokenize or not os.path.exists(os.path.join(path_data, "sst_train_fold_" + str_cur_fold + ".csv")):

            self.train_pd = pd.read_csv(os.path.join(path_data, "train_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c')
            self.valid_pd = pd.read_csv(os.path.join(path_data, "valid_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c')
            self.test_pd = pd.read_csv(os.path.join(path_data, "test_fold_" + str_cur_fold + ".csv"), sep=",", header=0, encoding="utf-8", engine='c')

            self.train_pd = self.sentence_toeknieze(self.train_pd)
            self.valid_pd = self.sentence_toeknieze(self.valid_pd)
            self.test_pd = self.sentence_toeknieze(self.test_pd)

            self.train_pd = self.filter_single_sent_text(self.train_pd)
            self.valid_pd = self.filter_single_sent_text(self.valid_pd)
            self.test_pd = self.filter_single_sent_text(self.test_pd)

            self.train_pd.to_csv(os.path.join(path_data, "sst_train_fold_" + str_cur_fold + ".csv"), sep=",", encoding="utf-8")
            self.valid_pd.to_csv(os.path.join(path_data, "sst_valid_fold_" + str_cur_fold + ".csv"), sep=",", encoding="utf-8")
            self.test_pd.to_csv(os.path.join(path_data, "sst_test_fold_" + str_cur_fold + ".csv"), sep=",", encoding="utf-8")

        return
    