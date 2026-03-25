from .NER_Cleaning import NER_Cleaning
from .NER_Text_Expansion import expand_text_spans 


class NER_Service:
    def __init__(self):
        self.ner_cleaning = NER_Cleaning()

    def NER_run(self, text: str, segmentations: list, no_trashs: list, window_sizes: list):
        
        SPLITS_FOLDER=""
        HADM_ID_PARQUET_LOCATIONS=""
        for segmentation in segmentations:
            tokenizer, tokens, predicted_labels, text, diag_codes_original = self.ner_cleaning.load_and_ner(    hadm_id=None
                                                                                                                , text=text
                                                                                                                , SPLITS_FOLDER=SPLITS_FOLDER
                                                                                                                , HADM_ID_PARQUET_LOCATIONS=HADM_ID_PARQUET_LOCATIONS
                                                                                                                , split="window")
        
                                                                                                 
            
            for no_trash in no_trashs:
        
                for window_size in window_sizes:
                    
                    if no_trash == 1:
                        if window_size != 0:
                            continue
        
                    ner_text_spans, expanded_text_spans = expand_text_spans(tokens, predicted_labels, tokenizer, window_size=window_size)
                    expanded_text_spans = [e for e in expanded_text_spans if len(e)>1]
                    
                    
        
        return expanded_text_spans

    def NER_run_for_color_print(self, text: str, segmentations: list, no_trashs: list, window_sizes: list):
        
        SPLITS_FOLDER=""
        HADM_ID_PARQUET_LOCATIONS=""
        for segmentation in segmentations:
            tokenizer, tokens, predicted_labels, text, diag_codes_original = self.ner_cleaning.load_and_ner_for_color_print(hadm_id=None
                                                                                                                            , text=text
                                                                                                                            , SPLITS_FOLDER=SPLITS_FOLDER
                                                                                                                            , HADM_ID_PARQUET_LOCATIONS=HADM_ID_PARQUET_LOCATIONS
                                                                                                                            , split="window")
                    
                                                                                                 
            
            for no_trash in no_trashs:
        
                for window_size in window_sizes:
                    
                    if no_trash == 1:
                        if window_size != 0:
                            continue
        
                    ner_text_spans, expanded_text_spans = expand_text_spans(tokens, predicted_labels, tokenizer, window_size=window_size)
                    expanded_text_spans = [e for e in expanded_text_spans if len(e)>1]
                    
                    
        
        return tokens, predicted_labels
                        
     