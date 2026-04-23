from ..vectordb.VectorDB import VectorDB
from operator import itemgetter
import pandas as pd

class Retrieval_Service:
    
    def __init__(self):
        self.vectordb = VectorDB()

        
    def retrieve_from_NER(self, expanded_text_spans: list, ks: list, score_thresholds: list, df_icd9_desc: pd.DataFrame, filter_json=None):

        doc_list_total = []
                            
        for expanded_text_span in expanded_text_spans:
            doc_list = self.vectordb.retrieve_with_score(query=expanded_text_span
                                                        , k=max(ks)
                                                        , filter_json=filter_json)
            doc_list = sorted(doc_list, key=itemgetter(1), reverse=True)
            
            doc_list = [(x, y, expanded_text_span) for (x, y) in doc_list]
            doc_list_total.append(doc_list)

                   
        for score_threshold in score_thresholds:

            
            for k in ks:     
                retrievals_k = {}
                for doc_list in doc_list_total:
                    doc_list_k = [(doc, score, expanded_text_span) for doc, score, expanded_text_span in doc_list if (1 - score) >= score_threshold]                        
                    doc_list_k = doc_list_k[:k]
                   
                    
                    retrievals_k.update({doc.metadata['ICD_CODE']: (doc.page_content, doc.metadata['source'], score, expanded_text_span) for (doc, score, expanded_text_span) in doc_list_k})


        
        k_icd_codes_from_expaned_text_spans_full = []

        for icd_code,value_tuple in retrievals_k.items():
            k_icd_codes_from_expaned_text_spans_full.append(
            {"retrieved_icd_code": icd_code
             , "Long Description": df_icd9_desc.loc[df_icd9_desc['DIAGNOSIS CODE'] == icd_code, 'LONG DESCRIPTION'].iloc[0]
             , "expanded_text_span":value_tuple[3]
             , "retrieved_text_span":value_tuple[0]
             , "retrieval_source": value_tuple[1]}
            )


        return k_icd_codes_from_expaned_text_spans_full