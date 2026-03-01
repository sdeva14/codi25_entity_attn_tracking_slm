import stanza

# import datasets
import pandas as pd

import torch

import benepar, spacy

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
# from transformers import XLNetModel
# from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaModel

import re

'''
    NP parser based on looking back: 
      tokenize the whole input first, then concatenate tokens one by one to check whether it is entity
'''

class NP_Parser_BackT():
    def __init__(self, tokenizer, encoder_weights):
        #
        # self.stanza_pipeline = stanza.Pipeline('en', processors="tokenize, pos, constituency", use_gpu=True)
        self.nlp = spacy.load('en_core_web_md')
        self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

        #
        self.tokenizer = tokenizer
        self.encoder_weights = encoder_weights

        self.punctuation_list = [".", ",", "!", ":", ";" "-", "#", "'", "_", ")"]
        self.prefix_list = ["(", "#", "*"]

    ################

    def filter_two_or_more_dots(self, str_target):
        str_target = re.sub(r'\.{2,}', ' ', str_target)
        str_target = str_target.strip()

        return str_target

    def filter_comma(self, str_target):
        str_target = str_target.replace(",", " ")
        str_target = str_target.strip()

        return str_target

    def filter_sbw_str(self, sbw_str):
        # sbw_str = sbw_str.strip()

        # filter comma
        # sbw_str = self.filter_two_or_more_dots(sbw_str)
        sbw_str = self.filter_comma(sbw_str)

        # exit
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
        
        # elif len(sbw_str) > 1 and (sbw_str[-2:] == "\'s"):
        #     sbw_str = sbw_str[:-2]
        
        if len(sbw_str) > 1 and (sbw_str[-2:] == "\'m"):
            sbw_str = sbw_str[:-2]

        if len(sbw_str) > 2 and (sbw_str[-3:] == "\'ll" or sbw_str[-3:] == "\'ve"):
            sbw_str = sbw_str[:-3]

        return sbw_str

    def match_np_sbw(self, np_in_words, list_sbw_loc):

        ind_sbw = 0  # index used to point the current item the sbw list
        list_np_sbw = []
        # print("SBW Len:" + str(len(list_sbw_loc)))

        # filtering empty np string (consisting by only spaces, caused by linguistic parser)
        temp = []
        for cur_np_pair in np_in_words:
            np_str = cur_np_pair[0]
            split_np_str = np_str.split()
            if len(split_np_str) > 0:
                temp.append(cur_np_pair)
        np_in_words = temp

        # iterate for each NP
        for cur_np_pair in np_in_words:
            # print(cur_np_pair)
            np_str = cur_np_pair[0]  # string, e.g., "this weekend"
            # np_loc = cur_np_pair[1]  # (start, end) -> not used in this version

            ## filter comma
            # split_np_str = np_str.split()

            temp = []
            for curr in np_str.split():
                curr = self.filter_comma(curr)
                temp.append(curr)
            split_np_str = temp

            # exit for empty NP (consisting only spaces)
            if len(split_np_str) < 1:
                continue

            np_sbw_ind_start = -1
            np_sbw_ind_end = -1

            # print(len(list_sbw_loc))

            # print(f"{split_np_str} {ind_sbw}")
            # print(ind_sbw)

            while ind_sbw < len(list_sbw_loc):

                sbw_str = list_sbw_loc[ind_sbw][0]
                sbw_loc = list_sbw_loc[ind_sbw][1]  # (start, end)

                # preprocessing string
                # print(f"{sbw_str} {split_np_str[0]}")

                if self.filter_sbw_str(sbw_str) == split_np_str[0]:
                    match_ind = 0  # already one word is matched

                    np_sbw_ind_start = sbw_loc[0]
                    # print(f"{sbw_str} {ind_sbw}")

                    #################

                    def compare_ent_str(str_input, str_ent):

                        is_continue = False

                        filtered_str = self.filter_sbw_str(str_input)

                        # print(str_input, str_ent)

                        if filtered_str == str_ent:
                            is_continue = True
                        elif len(str_input) == 1:  # # if there is an empty space midde in the string, then skip this
                            if ord(str_input[0]) == 9601:
                                is_continue = True

                        return is_continue

                    # while self.filter_sbw_str(list_sbw_loc[ind_sbw][0]) == split_np_str[match_ind]:
                    while compare_ent_str(list_sbw_loc[ind_sbw][0], split_np_str[match_ind]):

                        # if there is an empty space midde in the string, then skip this
                        cur_token = list_sbw_loc[ind_sbw][0]
                        if len(cur_token) == 1 and ord(cur_token[0])==9601:
                            ind_sbw += 1
                            continue

                        match_ind += 1

                        if match_ind == len(split_np_str):
                        #     # np_sbw_ind_end = list_sbw_loc[ind_sbw][1][1]
                            # print("!!!")
                            break

                        if ind_sbw == len(list_sbw_loc)-1:
                            break

                        ind_sbw += 1

                    np_sbw_ind_end = list_sbw_loc[ind_sbw][1][1]

                    # print(f"{match_ind} {np_sbw_ind_start} {np_sbw_ind_end}")

                    if match_ind == len(split_np_str) and np_sbw_ind_end > -1:
                        list_np_sbw.append((np_str, (np_sbw_ind_start, np_sbw_ind_end)))
                        break
                
                ind_sbw += 1

        # print("Result:")
        # print(list_np_sbw)

        return list_np_sbw

    ###########################

    def map_subwords_type1(self, list_sbw, prev_token, cur_token, ind_start, i, word):
        ''' 
            xlnet,
            google/gemma-2-2b-it 
            microsoft/Phi-3.5-mini-instruct
        '''
        is_added = False
        if ord(cur_token[0]) == 9601 or ord(cur_token[0]) == 63:
        # if ord(cur_token[0]) == 9601 or ord(cur_token[0]) == 63 or \
                # ord(cur_token[0]) == 44:  # 9601: empty space, 63: question mark, 44: ","
            
            list_sbw.append((word, (ind_start, i-1)))
            
            word = ""
            ind_start = i

            is_added = True
                
        # if a new word (9601: whitespace)
        if ord(cur_token[0]) == 9601:
            cur_token = cur_token[1:]

        ###########

        ## exception handling: "," or double space
        if not is_added and prev_token is not None and len(prev_token)>0:
            # if ord(prev_token[-1]) == 44 or ord(prev_token[-1]) == 9601:
            if ord(prev_token[-1]) == 9601:
                list_sbw.append((word, (ind_start, i-1)))
            
                word = ""
                ind_start = i

        ##
        word += cur_token
        prev_token = cur_token

        return ind_start, word, prev_token

    def map_subwords_type2(self, list_sbw, prev_token, cur_token, ind_start, i, word):
        ''' 
            llama3, Qwen/Qwen2.5-1.5B-Instruct
            -> ["<|begin_of_text|>Is it better for oneself's developpement to have broad knowledge of many academic subjects than to specialize in one specific subject?"]      
            -> ['<|begin_of_text|>', 'Is', 'Ġit', 'Ġbetter', 'Ġfor', 'Ġoneself', "'s", 'Ġdevelop', 'p', 'ement', 'Ġto', 'Ġhave', 'Ġbroad', 'Ġknowledge', 'Ġof', 'Ġmany', 'Ġacademic', 'Ġsubjects', 'Ġthan', 'Ġto', 'Ġspecialize', 'Ġin', 'Ġone', 'Ġspecific', 'Ġsubject', '?'] 
        '''
        is_added = False
        if ord(cur_token[0]) == 288 or ord(cur_token[0]) == 63:  # 288: unicode of empty space, 63: question mark, 44: ","
            list_sbw.append((word, (ind_start, i-1)))
            
            word = ""
            ind_start = i

            is_added = False
        
        # if a new word (9601: whitespace)
        if ord(cur_token[0]) == 288:
            cur_token = cur_token[1:]
        
        ## exception handling: "," or double space
        if not is_added and prev_token is not None and len(prev_token)>0:
            if ord(prev_token[-1]) == 288:
                list_sbw.append((word, (ind_start, i-1)))
            
                word = ""
                ind_start = i

        word += cur_token
        prev_token = cur_token

        return ind_start, word, prev_token
    
    def map_subwords_googlebert(self, list_sbw, prev_token, cur_token, ind_start, i, word):
        ''' 
            google-bert
            -> ["[CLS] is it better for oneself ' s developpement to have broad knowledge of many academic subjects than to specialize in one specific subject? [SEP]"]         
            -> ['[CLS]', 'is', 'it', 'better', 'for', 'oneself', "'", 's', 'develop', '##pe', '##ment', 'to', 'have', 'broad', 'knowledge', 'of', 'many', 'academic', 'subjects', 'than', 'to', 'special', '##ize', 'in', 'one', 'specific', 'subject', '?', '[SEP]']
        '''

        is_added = True
        # if ord(cur_token[0]) == 9601 or ord(cur_token[0]) == 63:  # 9601 -> unicode of empty space, 63 -> question mark
        # if not (len(cur_token) > 2 and ord(cur_token[0]) == 35 and ord(cur_token[1]) == 35):  # 35 -> unicode of empty space, 63 -> question mark
        if not (len(cur_token) > 2 and ord(cur_token[0]) == 35 and ord(cur_token[1]) == 35):  # 35 -> unicode of empty space, 63 -> question mark
            list_sbw.append((word, (ind_start, i-1)))
            
            word = ""
            ind_start = i

            is_added = False
        
        # if a new word (9601: whitespace)
        if len(cur_token)>1 and ord(cur_token[0]) == 35 and ord(cur_token[0]) == 35:
            cur_token = cur_token[2:]
        
        ## exception handling: "," or double space
        if not is_added and prev_token is not None and len(prev_token)>0:
            if len(prev_token) > 1 and ord(prev_token[-2]) == 35 and ord(prev_token[-1]) == 35:
                word = ""
                ind_start = i

                is_added = False
        
        word += cur_token
        prev_token = cur_token

        return ind_start, word, prev_token

    ############################

    def track_sbw_loc(self, sent, sent_ids, decoded_tokens, filter_punctation=True):

        # # debugging codes
        # print("---decode test---")
        # sent_ids = self.tokenizer(sent, return_tensors="pt").input_ids
        # print(sent_ids)
        # decoded_word = self.tokenizer.batch_decode(sent_ids)
        # print(decoded_word)
        # print("-----")

        # sent_ids = self.tokenizer(sent, return_tensors="pt").input_ids
        # print(sent_ids)
        # print(sent_ids.shape)
        # for curr in sent_ids:
        #     print( self.tokenizer.batch_decode(curr))
        # print("---")

        ## 
        # ## filter punctation
        # if sent[0] in self.prefix_list:
        #     sent = sent[1:]
        # if sent[-1] in self.punctuation_list:
        #     sent = sent[:-1]

        # ## tokenization as a whole input
        # sent_ids = self.tokenizer(sent, return_tensors="pt").input_ids

        # ## filtering special tokens from subword tokenizers
        # if "llama" in self.encoder_weights or "opt" in self.encoder_weights:
        #     sent_ids = sent_ids[:, 1:] # <s> ...
        # elif "flan-t5" in self.encoder_weights:
        #     sent_ids = sent_ids[:, :-1]  # ... </s>
        # elif "xlnet" in self.encoder_weights:
        #     sent_ids = sent_ids[:, :-2]  # <sep><cls>
        # elif "google/gemma-2" in self.encoder_weights:
        #     sent_ids = sent_ids[:, 1:]  # <bos> ...
        # elif "google-bert/bert-" in self.encoder_weights:
        #     sent_ids = sent_ids[:, 1:-1]  # [cls] ... [sep]

        ## group tokens
        '''
            * google/gemma-2-2b-it
            ["<bos>Is it better for oneself's developpement to have broad knowledge of many academic subjects than to specialize in one specific subject?"]
            ['<bos>', 'Is', '▁it', '▁better', '▁for', '▁oneself', "'", 's', '▁develop', 'pement', '▁to', '▁have', '▁broad', '▁knowledge', '▁of', '▁many', '▁academic', '▁subjects', '▁than', '▁to', '▁specialize', '▁in', '▁one', '▁specific', '▁subject', '?']
        '''
        '''
            subwords: [('Is', (0, 0)), ('it', (1, 1)), ('better', (2, 2)), ('for', (3, 3)), ("oneself's", (4, 7)), ('developpement', (8, 9)), ('to', (10, 10)), ('have', (11,
11)), ('broad', (12, 12)), ('knowledge', (13, 13)), ('of', (14, 14)), ('many', (15, 15)), ('academic', (16, 16)), ('subjects', (17, 17)), ('than', (18, 18)), ('to
', (19, 19)), ('specialize', (20, 21)), ('in', (22, 22)), ('one', (23, 23)), ('specific', (24, 24)), ('subject?', (25, 26))]
        '''

        ## decode tokens first
        # i_ids = sent_ids[0].tolist()
        # decoded_tokens = self.tokenizer.convert_ids_to_tokens(i_ids)
        # print(f"Decoded tokens: {decoded_tokens}")
        
        list_sbw = []
        ind_start = 0

        # init by the first token
        word = decoded_tokens[0]
        if ord(word[0]) == 9601:
            word = word[1:]

        prev_token = None
        # iterate for all tokens to make a mapping
        for i in range(1, len(decoded_tokens)):
            cur_token = decoded_tokens[i]
            # next_token = decoded_tokens[i+1]

            if "google-bert/bert-base" in self.encoder_weights:
                '''
                    tested list:
                    - google-bert/bert-base-uncased
                '''
                ind_start, word, prev_token = self.map_subwords_googlebert(list_sbw, prev_token, cur_token, ind_start, i, word)
            elif "meta-llama/Llama-3" in self.encoder_weights or \
                    "Qwen/Qwen2.5" in self.encoder_weights:
                '''
                    tested list:
                    - meta-llama/Llama-3.2-1B
                    - Qwen/Qwen2.5-1.5B-Instruct
                '''
                ind_start, word, prev_token = self.map_subwords_type2(list_sbw, prev_token, cur_token, ind_start, i, word)
            else:
                '''
                    tested list: 
                    - xlnet-base-cased
                    - google/gemma-2
                    - microsoft/Phi-3.5-mini-instruct
                    - 
                '''
                #  if "google/gemma-2" in self.encoder_weights \
                #     or "xlnet-" in self.encoder_weights:
                ind_start, word, prev_token = self.map_subwords_type1(list_sbw, prev_token, cur_token, ind_start, i, word)

            # ## google/gemma-2-2b-it
            # if ord(cur_token[0]) == 9601 or ord(cur_token[0]) == 63:  # 9601 -> unicode of empty space, 63 -> question mark
            #     list_sbw.append((word, (ind_start, i-1)))
                
            #     word = ""
            #     ind_start = i
            
            #  # concat
            # if ord(cur_token[0]) == 9601:
            #     cur_token = cur_token[1:]
            # word += cur_token

        if len(word) > 0:
            list_sbw.append((word, (ind_start, len(decoded_tokens)-1)))


        return list_sbw


    def extract_vp_with_len(self, sent, max_len_phrase=4, remove_duplicate_whitespace=True):

        # remove duplicate whietspace -> it causes wrong restuls in linguistic parsers
        if remove_duplicate_whitespace:
            sent = re.sub(' +', ' ', sent)

        parsed_sent = self.nlp(sent)
        parsed_sent = list(parsed_sent.sents)[0]

        ##
        target_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        phrase_in_words = []
        prev = ""  # store the previous np to check whether it is included
        for span in parsed_sent._.constituents:  # pre-order traverse
            tag = span._.parse_string[1:3]
            item = str(span)
            # print(tag + " " + item)
            # print(str(span.start) + " " + str(span.end))

            ind_start = span.start
            ind_end = span.end
            len_cur_np = ind_end - ind_start

            ## filter NPs by the constraint NP length (the number of words)
            # if "NP" in tag or "NN" in tag:
            for cur_target_tag in target_tags:
                if "VB" in tag:
                # if cur_target_tag == tag:
                    if len_cur_np <= max_len_phrase:

                        ## check whether current NP is the sub-np of the previous one
                        if item not in prev:
                            phrase_in_words.append( (item, (ind_start, ind_end-1)) )

                        # prev = item
                        if len(phrase_in_words)>0:
                            prev = phrase_in_words[-1][0]
            
        # print(phrase_in_words)
        # print(welkfjwelf)

        return phrase_in_words

    def extract_np_with_len(self, sent, max_len_phrase=4, remove_duplicate_whitespace=True):

        # remove duplicate whietspace -> it causes wrong restuls in linguistic parsers
        if remove_duplicate_whitespace:
            sent = re.sub(' +', ' ', sent)

        parsed_sent = self.nlp(sent)
        parsed_sent = list(parsed_sent.sents)[0]

        ##
        target_tags = [""]
        phrase_in_words = []
        prev = ""  # store the previous np to check whether it is included
        for span in parsed_sent._.constituents:  # pre-order traverse
            tag = span._.parse_string[1:3]
            item = str(span)
            # print(tag + " " + item)
            # print(str(span.start) + " " + str(span.end))

            ind_start = span.start
            ind_end = span.end
            len_cur_np = ind_end - ind_start

            ## filter NPs by the constraint NP length (the number of words)
            if "NP" in tag or "NN" in tag:
                if len_cur_np <= max_len_phrase:

                    ## check whether current NP is the sub-np of the previous one
                    if item not in prev:
                        phrase_in_words.append( (item, (ind_start, ind_end-1)) )

                    # prev = item
                    if len(phrase_in_words)>0:
                        prev = phrase_in_words[-1][0]
        
        # print(np_in_words)
        # print(welkfjwelf)

        return phrase_in_words

    def filter_speical_tokens(self, sent_ids):
         ## filtering special tokens from subword tokenizers
        if "meta-llama/Llama-3" in self.encoder_weights:
            sent_ids = sent_ids[:, 1:] # <begin_of_text> ...
        elif "llama" in self.encoder_weights or "opt" in self.encoder_weights:
            sent_ids = sent_ids[:, 1:] # <s> ...
        elif "flan-t5" in self.encoder_weights:
            sent_ids = sent_ids[:, :-1]  # ... </s>
        elif "xlnet" in self.encoder_weights:
            sent_ids = sent_ids[:, :-2]  # <sep><cls>
        elif "google/gemma-2" in self.encoder_weights:
            sent_ids = sent_ids[:, 1:]  # <bos> ...
        elif "google-bert/bert-" in self.encoder_weights:
            sent_ids = sent_ids[:, 1:-1]  # [cls] ... [sep]
    
        return sent_ids

    #####

    def get_np_index_subwords(self, cur_sent, is_add_vp=False):
        
        output_np_parser = {}

        ## exit for empty strings
        if len(cur_sent) < 1:
            return None

        # ## filter multiple dots (disabled)
        # np_str = self.filter_two_or_more_dots(np_str)

        ## stage 0: pre-processing;
        # if cur_sent[-1] in list_punctuation:
        #     cur_sent = cur_sent[:-1]

        #### stage 0:decoding sentence by the target tokenizer first -> because spacing can be different
        ## filter punctation
        if cur_sent[0] in self.prefix_list:
            cur_sent = cur_sent[1:]
        if cur_sent[-1] in self.punctuation_list:
            cur_sent = cur_sent[:-1]

        ## tokenization as a whole input
        sent_ids = self.tokenizer(cur_sent, return_tensors="pt").input_ids
        sent_ids = self.filter_speical_tokens(sent_ids)

        ## decoding then extract sentence text -> each tokenizer can cause a different sentence due to different spacing or etc
        decoded_str = self.tokenizer.batch_decode(sent_ids)
        decoded_str = decoded_str[0]  # only deal with it one by one

        i_ids = sent_ids[0].tolist()
        decoded_tokens = self.tokenizer.convert_ids_to_tokens(i_ids)

        # print(cur_sent)

        #### stage 1: NP location
        # np_in_words = self.extract_np(cur_sent)
        np_in_words = self.extract_np_with_len(decoded_str, 4, remove_duplicate_whitespace=True)
        # print("NP in words: {}".format(np_in_words))

        #### stage 2: sbw location
        list_sbw_loc = self.track_sbw_loc(cur_sent, sent_ids, decoded_tokens, filter_punctation=True)
        # print("subwords: {}".format(list_sbw_loc))

        #### stage 3: matching NP to SBW location
        list_np_sbw_loc = self.match_np_sbw(np_in_words, list_sbw_loc)  # [("The important of education", (0, 3), ("our world of 21st centry", (15, 20)), ...]
        # print("NP in subwords: {}".format(list_np_sbw_loc))
        # print("")

        ## debugging
        is_np_parsed_all = True
        if len(np_in_words) != len(list_np_sbw_loc):
            # print(cur_sent)
            is_np_parsed_all = False
            # print(wlkfjlwejklwef)
        
        ## VP handling for optional
        if is_add_vp:
            vp_in_words = self.extract_vp_with_len(decoded_str, 4, remove_duplicate_whitespace=True)
            list_vp_sbw_loc = self.match_np_sbw(vp_in_words, list_sbw_loc)

            is_vp_parsed_all = True
            if len(vp_in_words) != len(list_vp_sbw_loc):
                # print(cur_sent)
                is_vp_parsed_all = False
            
            # print("VP in words: {}".format(vp_in_words))
            # print("VP in subwords: {}".format(list_vp_sbw_loc))

        #### output formatting
        output_np_parser["decoded_str"] = decoded_str
        output_np_parser["decoded_tokens"] = decoded_tokens
        output_np_parser["np_in_words"] = np_in_words
        output_np_parser["list_sbw_loc"] = list_sbw_loc
        output_np_parser["np_sbw_loc"] = list_np_sbw_loc
        output_np_parser["is_parsed_all"] = is_np_parsed_all

        #
        if is_add_vp:
            output_np_parser["vp_in_words"] = vp_in_words
            output_np_parser["vp_sbw_loc"] = list_vp_sbw_loc
            output_np_parser["is_vp_parsed_all"] = is_vp_parsed_all

        return output_np_parser