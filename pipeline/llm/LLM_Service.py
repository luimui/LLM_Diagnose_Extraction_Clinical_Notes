from .Ollama_Client import Ollama_Client
from .LLM_Strategies import LLM_Strategies
from tqdm import tqdm
import inspect


class LLM_Service():

    def __init__(self):
        self.ollamaClient = Ollama_Client()
        self.llmStrategies = LLM_Strategies()
        self.strategies_dict = {}
        self.strategies_dict['strategy10'] = self.llmStrategies.strategy10


    def llm_call(self, text: str, k_icd_codes_from_expaned_text_spans_full: list):
    
        
              
        
        for strategy in list(self.strategies_dict.keys()):
            
            print(f'strategy: {strategy}')
            
            for model_llm in tqdm(["gpt-oss:120b"]):           
                
                letter = text

                evidence_dicts_list_full = self.strategies_dict[strategy](text, k_icd_codes_from_expaned_text_spans_full, model_llm, self.ollamaClient)                    

        
        return evidence_dicts_list_full