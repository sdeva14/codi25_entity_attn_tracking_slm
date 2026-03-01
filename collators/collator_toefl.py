import torch

class CollatorPaddingTOEFL_Sent:

    def __init__(self, tokenizer, pad_token, max_num_sent, max_len_sent, pad_to_multiple_of=None, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

        self.pad_token = pad_token
        self.max_num_sent = max_num_sent
        self.max_len_sent = max_len_sent
    
    def __call__(self, features):

        ## iterate each sentence as batch to tokenize (since nested list cannot be tokenized in HF)
        # identify the maximum number of sents in this batch (max_num_sent can be used instead)
        num_sents_batch = []
        max_num_sent_batch = 0
        for curr in features:
            # the number of sents
            sents_ids = curr["input_ids"]
            num_sents = len(sents_ids)
            num_sents_batch.append(num_sents)
            max_num_sent_batch = num_sents if num_sents > max_num_sent_batch else max_num_sent_batch

        num_sents_batch = torch.as_tensor(num_sents_batch)
        
        # mask for sentence dimension; (batch_size, max_num_sents)
        mask_sent = torch.arange(max_num_sent_batch).expand(len(num_sents_batch), max_num_sent_batch) < num_sents_batch.unsqueeze(1)

        # length of each sentence
        len_sents_batch = []
        for curr in features:
            sents_ids = curr["input_ids"]
            cur_num_sents = len(sents_ids)

            len_sents = []
            for idx_sent in range(max_num_sent_batch):
                len_cur_sent = 0
                if idx_sent < cur_num_sents:
                    cur_sent = sents_ids[idx_sent]
                    len_cur_sent = len(cur_sent)
                len_sents.append(len_cur_sent)
                
            len_sents_batch.append(len_sents)
        len_sents_batch = torch.as_tensor(len_sents_batch)
    
        ## Tokenize
        # instead of a real batch, consider each sentence as a batch, since batch size will be lower due to GPU memory
        labels = []
        tokenized_ids = []
        tokenized_mask = []
        for cur_doc in features:  # iterate batch
            sents_ids = cur_doc["input_ids"]
            attn_mask = cur_doc["attention_mask"]
            labels.append(cur_doc["labels"])

            t_batch = []  # [{"input_ids": ..., "attention_mask": ...}, {"input_ids": ..., "attention_mask": ...}, ...]
            cur_num_sents = len(sents_ids)
            for idx_sent in range(max_num_sent_batch):
                if idx_sent < cur_num_sents:
                    t_batch.append({"input_ids": sents_ids[idx_sent], "attention_mask": attn_mask[idx_sent]})
                else:
                    t_batch.append({"input_ids": [], "attention_mask": []})  # fill the empty padded list

            cur_tokenized = self.tokenizer.pad(
                t_batch,
                padding = "max_length",
                max_length=self.max_len_sent,
                # pad_to_multiple_of = self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            tokenized_ids.append(cur_tokenized["input_ids"])
            tokenized_mask.append(cur_tokenized["attention_mask"])
            
        tokenized_ids = torch.stack(tokenized_ids, dim=0)
        tokenized_mask = torch.stack(tokenized_mask, dim=0)
        labels = torch.as_tensor(labels)
        batch = {"input_ids": tokenized_ids, "attention_mask": tokenized_mask, "labels": labels}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        # add more meta-data
        batch["sent_num"] = num_sents_batch  # is it necessary?
        batch["mask_sent"] = mask_sent
        batch["len_sents"] = len_sents_batch

        return batch
