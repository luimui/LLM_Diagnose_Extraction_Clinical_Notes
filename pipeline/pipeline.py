import pandas as pd
from pipeline.ner.NER_Service import NER_Service
from pipeline.retrieval.Retrieval_Service import Retrieval_Service
from pipeline.llm.LLM_Service import LLM_Service
from pipeline.utils.Color_Tokens import color_tokens_html
import re

class Pipeline:
    def __init__(self):#, model_name="all-MiniLM-L6-v2", db_path="./VectorDB/all-MiniLM-L6-v2_icd9_full_LD_UMLS_SNOMED_cleaned__dashsplit_noVCodes/"):
        self.ner = NER_Service()
        self.retrieval = Retrieval_Service()
        self.llm = LLM_Service()
        
        self.df_icd9_desc = pd.read_csv("./data/ICD_9_LD_hierarchy.csv")

    def run(self, text: str, no_trashs: list, ks: list, score_thresholds: list, segmentations: list, window_sizes: list):        

        #----------------------NER
        expanded_text_spans = self.ner.NER_run(text=text, no_trashs=no_trashs, segmentations=segmentations, window_sizes=window_sizes)
       
        tokens, predicted_labels = self.ner.NER_run_for_color_print(text=text, no_trashs=no_trashs, segmentations=segmentations, window_sizes=window_sizes)
        
        html_colored_text = color_tokens_html(tokens, predicted_labels)

        html_colored_text = re.sub(r' (<span[^>]*>)(##)([^<]*</span>)', r'\1\3', html_colored_text)


        
        #---------------------Retrieval
        k_icd_codes_from_expaned_text_spans_full = self.retrieval.retrieve_from_NER(expanded_text_spans=expanded_text_spans, ks=ks, score_thresholds=score_thresholds, df_icd9_desc=self.df_icd9_desc)
       
        #------------------------LLM
        evidence_dicts_list_full = self.llm.llm_call(text=text, k_icd_codes_from_expaned_text_spans_full=k_icd_codes_from_expaned_text_spans_full)
        
        return k_icd_codes_from_expaned_text_spans_full, html_colored_text, evidence_dicts_list_full