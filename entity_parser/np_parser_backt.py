import re
import pandas as pd
import torch
import benepar
import spacy
import stanza

from transformers import AutoModel, AutoTokenizer

"""
NP parser (look-back): tokenize the whole input first, then align tokens to NPs/VPs.
"""


class NP_Parser_BackT:
    def __init__(self, tokenizer, encoder_weights):
        self.nlp = spacy.load("en_core_web_md")
        self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

        #
        self.tokenizer = tokenizer
        self.encoder_weights = encoder_weights

        self.punctuation_list = [".", ",", "!", ":", ";", "-", "#", "'", "_", ")"]
        self.prefix_list = ["(", "#", "*"]

    def filter_two_or_more_dots(self, str_target):
        str_target = re.sub(r'\.{2,}', ' ', str_target)
        str_target = str_target.strip()

        return str_target

    def filter_comma(self, str_target):
        str_target = str_target.replace(",", " ")
        str_target = str_target.strip()

        return str_target

    def filter_sbw_str(self, sbw_str):
        sbw_str = self.filter_comma(sbw_str)
        if len(sbw_str) < 1:
            return sbw_str

        # filter as cases
        if sbw_str[0] == "," or sbw_str[0] == "?" or sbw_str[0] == "!" or sbw_str[0] == "(" \
            or sbw_str[0] == "\"" or sbw_str[0] == "\'" or sbw_str[0]=="[":
            sbw_str = sbw_str[1:]

        if len(sbw_str) > 0 and \
            (sbw_str[-1] == ")" or sbw_str[-1] == "]" or sbw_str[-1] == "\"" or sbw_str[-1] == "\'" or sbw_str[-1] == ":" or 
            sbw_str[-1] == ";" or sbw_str[-1] == "/"):
            sbw_str = sbw_str[:-1]
        if len(sbw_str) > 1 and (sbw_str[-2:] == "\'m"):
            sbw_str = sbw_str[:-2]

        if len(sbw_str) > 2 and (sbw_str[-3:] == "\'ll" or sbw_str[-3:] == "\'ve"):
            sbw_str = sbw_str[:-3]

        return sbw_str

    def match_np_sbw(self, np_in_words, list_sbw_loc):

        ind_sbw = 0
        list_np_sbw = []
        temp = []
        for cur_np_pair in np_in_words:
            np_str = cur_np_pair[0]
            split_np_str = np_str.split()
            if len(split_np_str) > 0:
                temp.append(cur_np_pair)
        np_in_words = temp
        for cur_np_pair in np_in_words:
            np_str = cur_np_pair[0]
            temp = []
            for curr in np_str.split():
                curr = self.filter_comma(curr)
                temp.append(curr)
            split_np_str = temp
            if len(split_np_str) < 1:
                continue

            np_sbw_ind_start = -1
            np_sbw_ind_end = -1
            while ind_sbw < len(list_sbw_loc):
                sbw_str = list_sbw_loc[ind_sbw][0]
                sbw_loc = list_sbw_loc[ind_sbw][1]
                if self.filter_sbw_str(sbw_str) == split_np_str[0]:
                    match_ind = 0
                    np_sbw_ind_start = sbw_loc[0]

                    def compare_ent_str(str_input, str_ent):
                        is_continue = False
                        filtered_str = self.filter_sbw_str(str_input)
                        if filtered_str == str_ent:
                            is_continue = True
                        elif len(str_input) == 1 and ord(str_input[0]) == 9601:
                            is_continue = True
                        return is_continue

                    while compare_ent_str(list_sbw_loc[ind_sbw][0], split_np_str[match_ind]):
                        cur_token = list_sbw_loc[ind_sbw][0]
                        if len(cur_token) == 1 and ord(cur_token[0])==9601:
                            ind_sbw += 1
                            continue

                        match_ind += 1
                        if match_ind == len(split_np_str):
                            break

                        if ind_sbw == len(list_sbw_loc)-1:
                            break

                        ind_sbw += 1

                    np_sbw_ind_end = list_sbw_loc[ind_sbw][1][1]
                    if match_ind == len(split_np_str) and np_sbw_ind_end > -1:
                        list_np_sbw.append((np_str, (np_sbw_ind_start, np_sbw_ind_end)))
                        break
                
                ind_sbw += 1
        return list_np_sbw

    def map_subwords_type1(self, list_sbw, prev_token, cur_token, ind_start, i, word):
        is_added = False
        if ord(cur_token[0]) == 9601 or ord(cur_token[0]) == 63:
            list_sbw.append((word, (ind_start, i-1)))
            
            word = ""
            ind_start = i

            is_added = True
        if ord(cur_token[0]) == 9601:
            cur_token = cur_token[1:]
        if not is_added and prev_token is not None and len(prev_token) > 0:
            if ord(prev_token[-1]) == 9601:
                list_sbw.append((word, (ind_start, i-1)))
            
                word = ""
                ind_start = i
        word += cur_token
        prev_token = cur_token
        return ind_start, word, prev_token

    def map_subwords_type2(self, list_sbw, prev_token, cur_token, ind_start, i, word):
        is_added = False
        if ord(cur_token[0]) == 288 or ord(cur_token[0]) == 63:
            list_sbw.append((word, (ind_start, i-1)))
            
            word = ""
            ind_start = i

            is_added = False
        if ord(cur_token[0]) == 288:
            cur_token = cur_token[1:]
        if not is_added and prev_token is not None and len(prev_token) > 0:
            if ord(prev_token[-1]) == 288:
                list_sbw.append((word, (ind_start, i-1)))
            
                word = ""
                ind_start = i

        word += cur_token
        prev_token = cur_token

        return ind_start, word, prev_token

    def map_subwords_googlebert(self, list_sbw, prev_token, cur_token, ind_start, i, word):
        is_added = True
        if not (len(cur_token) > 2 and ord(cur_token[0]) == 35 and ord(cur_token[1]) == 35):
            list_sbw.append((word, (ind_start, i-1)))
            
            word = ""
            ind_start = i

            is_added = False
        if len(cur_token) > 1 and ord(cur_token[0]) == 35 and ord(cur_token[1]) == 35:
            cur_token = cur_token[2:]
        if not is_added and prev_token is not None and len(prev_token) > 0:
            if len(prev_token) > 1 and ord(prev_token[-2]) == 35 and ord(prev_token[-1]) == 35:
                word = ""
                ind_start = i

                is_added = False
        word += cur_token
        prev_token = cur_token
        return ind_start, word, prev_token

    def track_sbw_loc(self, sent, sent_ids, decoded_tokens, filter_punctation=True):
        list_sbw = []
        ind_start = 0
        word = decoded_tokens[0]
        if ord(word[0]) == 9601:
            word = word[1:]

        prev_token = None
        for i in range(1, len(decoded_tokens)):
            cur_token = decoded_tokens[i]
            if "google-bert/bert-base" in self.encoder_weights:
                ind_start, word, prev_token = self.map_subwords_googlebert(list_sbw, prev_token, cur_token, ind_start, i, word)
            elif "meta-llama/Llama-3" in self.encoder_weights or "Qwen/Qwen2.5" in self.encoder_weights:
                ind_start, word, prev_token = self.map_subwords_type2(list_sbw, prev_token, cur_token, ind_start, i, word)
            else:
                ind_start, word, prev_token = self.map_subwords_type1(list_sbw, prev_token, cur_token, ind_start, i, word)
        if len(word) > 0:
            list_sbw.append((word, (ind_start, len(decoded_tokens)-1)))


        return list_sbw


    def extract_vp_with_len(self, sent, max_len_phrase=4, remove_duplicate_whitespace=True):
        if remove_duplicate_whitespace:
            sent = re.sub(' +', ' ', sent)

        parsed_sent = self.nlp(sent)
        parsed_sent = list(parsed_sent.sents)[0]
        target_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        phrase_in_words = []
        prev = ""
        for span in parsed_sent._.constituents:
            tag = span._.parse_string[1:3]
            item = str(span)
            ind_start = span.start
            ind_end = span.end
            len_cur_np = ind_end - ind_start
            for cur_target_tag in target_tags:
                if "VB" in tag and len_cur_np <= max_len_phrase:
                    if item not in prev:
                        phrase_in_words.append((item, (ind_start, ind_end - 1)))
                    if len(phrase_in_words) > 0:
                        prev = phrase_in_words[-1][0]
        return phrase_in_words

    def extract_np_with_len(self, sent, max_len_phrase=4, remove_duplicate_whitespace=True):
        if remove_duplicate_whitespace:
            sent = re.sub(" +", " ", sent)
        parsed_sent = self.nlp(sent)
        parsed_sent = list(parsed_sent.sents)[0]
        phrase_in_words = []
        prev = ""
        for span in parsed_sent._.constituents:
            tag = span._.parse_string[1:3]
            item = str(span)
            ind_start = span.start
            ind_end = span.end
            len_cur_np = ind_end - ind_start
            if "NP" in tag or "NN" in tag:
                if len_cur_np <= max_len_phrase:
                    if item not in prev:
                        phrase_in_words.append((item, (ind_start, ind_end - 1)))
                    if len(phrase_in_words) > 0:
                        prev = phrase_in_words[-1][0]
        return phrase_in_words

    def filter_speical_tokens(self, sent_ids):
        if "meta-llama/Llama-3" in self.encoder_weights:
            sent_ids = sent_ids[:, 1:]
        elif "llama" in self.encoder_weights or "opt" in self.encoder_weights:
            sent_ids = sent_ids[:, 1:]
        elif "flan-t5" in self.encoder_weights:
            sent_ids = sent_ids[:, :-1]
        elif "xlnet" in self.encoder_weights:
            sent_ids = sent_ids[:, :-2]
        elif "google/gemma-2" in self.encoder_weights:
            sent_ids = sent_ids[:, 1:]
        elif "google-bert/bert-" in self.encoder_weights:
            sent_ids = sent_ids[:, 1:-1]
        return sent_ids

    def get_np_index_subwords(self, cur_sent, is_add_vp=False):
        
        output_np_parser = {}
        if len(cur_sent) < 1:
            return None
        if cur_sent[0] in self.prefix_list:
            cur_sent = cur_sent[1:]
        if cur_sent[-1] in self.punctuation_list:
            cur_sent = cur_sent[:-1]
        sent_ids = self.tokenizer(cur_sent, return_tensors="pt").input_ids
        sent_ids = self.filter_speical_tokens(sent_ids)
        decoded_str = self.tokenizer.batch_decode(sent_ids)[0]
        i_ids = sent_ids[0].tolist()
        decoded_tokens = self.tokenizer.convert_ids_to_tokens(i_ids)
        np_in_words = self.extract_np_with_len(decoded_str, 4, remove_duplicate_whitespace=True)
        list_sbw_loc = self.track_sbw_loc(cur_sent, sent_ids, decoded_tokens, filter_punctation=True)
        list_np_sbw_loc = self.match_np_sbw(np_in_words, list_sbw_loc)
        is_np_parsed_all = len(np_in_words) == len(list_np_sbw_loc)
        if is_add_vp:
            vp_in_words = self.extract_vp_with_len(decoded_str, 4, remove_duplicate_whitespace=True)
            list_vp_sbw_loc = self.match_np_sbw(vp_in_words, list_sbw_loc)
            is_vp_parsed_all = len(vp_in_words) == len(list_vp_sbw_loc)
        output_np_parser["decoded_str"] = decoded_str
        output_np_parser["decoded_tokens"] = decoded_tokens
        output_np_parser["np_in_words"] = np_in_words
        output_np_parser["list_sbw_loc"] = list_sbw_loc
        output_np_parser["np_sbw_loc"] = list_np_sbw_loc
        output_np_parser["is_parsed_all"] = is_np_parsed_all
        if is_add_vp:
            output_np_parser["vp_in_words"] = vp_in_words
            output_np_parser["vp_sbw_loc"] = list_vp_sbw_loc
            output_np_parser["is_vp_parsed_all"] = is_vp_parsed_all

        return output_np_parser