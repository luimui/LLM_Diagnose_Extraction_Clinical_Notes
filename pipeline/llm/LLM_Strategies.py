import json
from ..schemas.evidence_objects import DetailedEvidencesList
from .utils import enrich_detailed_evidence
from ..vectordb.VectorDB import VectorDB

class LLM_Strategies():


    def __init__(self):
        self.vectordb = VectorDB()
        
    
    def step10_1(self, k_retrievals, text):

        message_similar_text_evidence = [
                             {"role": "system", "content": """You are an expert fact checker. Your job is to fact check the diagnoses ICD 9 diagnoses list provided based on the clinical note given and return a list of python dictionaries. 
                                                             Find out if the proposed diagnoses does have any text evidence. If you find any text evidence that support this particular diagnosis or a somewhat similar one (it does not have to be very close), save it under the dictionary key "text_evidence", if no evidence is present or not mentioned leave it empty, also if you find contradictory evidence, leave "text_evidence" empty. If you found any evidence for this particular or a somewhat similar diagnosis save as "True" under key "deduction", otherwise save as "False" under key "deduction". Put the alternative similar diagnosis under the key "alternative_similar_diagnosis" if any evidence for it is present. 
                                                             Return a list of dictionaries, where each dictionary contains the ICD 9 diagnosis, the positive text evidence, the . Find the span of text (a one sentence) from the letter as an evidence. Do not add extra text.
                                                             Output Format Example: [
                                                                                     {"diagnosis":"...",  "text_evidence":"...", "alternative_similar_diagnosis":"...", "deduction":"..."},
                                                                                     {"diagnosis":"...", "text_evidence":"...", "alternative_similar_diagnosis":"...", "deduction":"..."}
                                                                                     {"diagnosis":"...", "text_evidence":"...", "alternative_similar_diagnosis":"...", "deduction":"..."}
                                                                                    ] 
                                                                             """},
                              {"role": "user", "content": "Here is the list of possible diagnoses codes: \n" + k_retrievals + "\n" + "Here is the clinical note: \n\n" + text},
                            ]
    
        return message_similar_text_evidence
    
    
    
    
    def strategy10(self, letter, k_icd_codes_from_expaned_text_spans_full, model_llm, ollamaClient):
    
        
        evidence_dicts_list_full = []

    
    
    
        k_retrieval_list = ["Diagnosis: " + k_retrieval["Long Description"] + " --> because of found text span: " +  k_retrieval["expanded_text_span"] for k_retrieval in k_icd_codes_from_expaned_text_spans_full]
    

        message_similar_text_evidence = self.step10_1(str(k_retrieval_list), letter)
    
        
        print(f'model_llm: {model_llm}')
        response_str = ollamaClient.ollama_request(model=model_llm, messages=message_similar_text_evidence)#, format=SimilarityEvidenceDictsList.model_json_schema())
        json_data = json.loads(response_str)

        try:
            detailed_evidence_list = list(DetailedEvidencesList.model_validate_json(json_data["message"]["content"]))[0][1]
            
            enriched_detailed_evidence_list = [enrich_detailed_evidence(detailed_evidence, k_retrieval) for (detailed_evidence, k_retrieval) in zip(detailed_evidence_list, k_icd_codes_from_expaned_text_spans_full)]

            for evidence in enriched_detailed_evidence_list:
                try:
                    icd_code = df_icd9_desc.loc[df_icd9_desc['LONG DESCRIPTION'] == evidence.diagnosis_LD, 'DIAGNOSIS CODE'].iloc[0]
                except:
                    print(f'evidence.diagnosis_LD {evidence.diagnosis_LD} not in df_icd9_desc["LONG DESCRIPTION"], retrieving from VectorDB')
                    icd_code = self.vectordb.similarity_search_with_score(evidence.diagnosis, k=1)[0][0].metadata['ICD_CODE']
                evidence.diagnosis_code = icd_code
                
                    
            evidence_dicts_list_full.extend(enriched_detailed_evidence_list)

        except Exception as e:
            print(f'validation failed: {json_data["message"]["content"]}')
       
            print(f"Error: {e}")
    
    
            
        return evidence_dicts_list_full 