from .NER_Model import NER_Model
import ast
import json
import sys
import os

import nltk
from nltk.corpus import words
from nltk.corpus import stopwords
from spacy.lang.en import English
import pandas as pd
from tqdm import tqdm

class NER_Cleaning:
    def __init__(self, UMLS_loading=False):
        
        NER_MODEL_NAME = 'bert-base-cased'
        NER_MODEL_PATH = './NER_Model/' + NER_MODEL_NAME

        self.ner_model = NER_Model()
        
        model, tokenizer = self.ner_model.load_ner_model_from_directory(NER_MODEL_PATH=NER_MODEL_PATH, NER_MODEL_NAME=NER_MODEL_NAME)        
        self.model = model
        self.tokenizer = tokenizer
        # Pull stop words
        class CustomEnglishDefaults(English.Defaults):
            stop_words = set(["custom", "stop"])
        
        self.nlp = English()
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading nltk stopwords")
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            print("Downloading nltk words")
            nltk.download('words')         
                        
        dictionary_word_list = set(words.words())



    def load_and_ner(self, hadm_id, text, SPLITS_FOLDER, HADM_ID_PARQUET_LOCATIONS, window_size=0, split="window"):

        diag_codes_original = ['no_diagnoses_provided']    
        
        
        tokens, predicted_labels = self.ner_model.ner_disease_terms(model=self.model, tokenizer=self.tokenizer, text=text, split=split)
            
        #ner_labels, ner_text_spans, expanded_text_spans = te.expand_text_spans_claude(tokens, predicted_labels, self.tokenizer, window_size=window_size)

           
        return self.tokenizer, tokens, predicted_labels, text, diag_codes_original#ner_labels, ner_text_spans, expanded_text_spans, text, diag_codes_original


    def load_and_ner_for_color_print(self, hadm_id, text, SPLITS_FOLDER, HADM_ID_PARQUET_LOCATIONS, window_size=0, split="window"):

        diag_codes_original = ['no_diagnoses_provided']    
        
        
        tokens, predicted_labels = self.ner_model.ner_disease_terms_for_color_printing(model=self.model, tokenizer=self.tokenizer, text=text, split=split)
            
        #ner_labels, ner_text_spans, expanded_text_spans = te.expand_text_spans_claude(tokens, predicted_labels, self.tokenizer, window_size=window_size)

           
        return self.tokenizer, tokens, predicted_labels, text, diag_codes_original#ner_labels, ner_text_spans, expanded_text_spans, text, diag_codes_original


    
    

    def clean_stopwords(self, text):
        
        doc = self.nlp(text)
        
        # Filters out verbs and pronouns only, nothing else
        filtered = [token for token in doc if (token.pos_ != "VERB" and token.pos_ != "PRON")] 
        result = "".join(token.text_with_ws for token in filtered)
      
        return result   

        

    def clean_stopwords_keep_newlines(self, text):
        
        doc = self.nlp(text.lower())
        filtered_tokens = []
        
        for token in doc:
            if token.text == "\n":  # Keep newlines as is
                filtered_tokens.append("\n")
            elif not token.is_stop and (token.is_alpha or token.like_num):
                filtered_tokens.append(token.text)
        
        result = " ".join(filtered_tokens).replace(" \n ", "\n")
        return result

        
    
   
    def is_in_text(self, ner_text_span, text):
        doc = self.nlp(text.lower())
        filtered = [token.text for token in doc if not token.is_stop and token.is_alpha]
        
        return any(word.lower() in filtered for word in ner_text_span)
    
    def is_umls_term(self, ner_text_span):
        doc = nlp(ner_text_span.lower())
        filtered = [token.text for token in doc if not token.is_stop and token.is_alpha]
        
        return any(word.lower() in self.umls_word_list.lower() for word in filtered)
    
    def is_in_icd9_desc(self, ner_text_span, db_name):
        doc = self.nlp(ner_text_span.lower())
        filtered = [token.text for token in doc if not token.is_stop and token.is_alpha]

        return any(word.lower() in self.icd9_text[db_name].lower() for word in filtered)


