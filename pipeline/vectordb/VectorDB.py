from langchain_chroma import Chroma
from ..embeddings.Embeddings import EmbeddingFunctionGPU


class VectorDB:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = "../../VectorDB/all-MiniLM-L6-v2_icd9_full_LD_UMLS_SNOMED_cleaned__dashsplit_noVCodes/"):
        
        self.collection_name = self._clean_name(model_name)
        
        self.db_path = db_path
        
        self.embedding = self._get_embedding()

        self.vector_store = self._connect()

    
    def _clean_name(self, name: str) -> str:
        
        return name.replace("-", "_").replace("/", "_").replace(":","_")

    
    def _get_embedding(self):
        
        embeddingFunctionGPU = EmbeddingFunctionGPU()
        embeddingFunction = embeddingFunctionGPU.get_emb_gpu_model()
        return embeddingFunction

    
    def _connect(self):

        vector_store = Chroma(
                                collection_name=self.collection_name,
                                embedding_function=self.embedding,
                                persist_directory=self.db_path,
                                collection_metadata={"hnsw:space": "cosine"}
                            )

        return vector_store

    def similarity_search_with_score(self, query: str, k: int):
        
        return self.vector_store.similarity_search_with_score(query, k=k)

    def retrieve(self, query: str, k: int, score_threshold: float = 0.01, filter_json: dict = None):

        search_kwargs = {"k": k, "score_threshold": score_threshold}
        
        if filter_json:
            search_kwargs["filter"] = filter_json
        
        retriever = self.vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs=search_kwargs)
        doc_list = retriever.invoke(input_query)
        
        return doc_list
        


    def retrieve_with_score(self, query: str, k: int, filter_json: dict = None):

        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
       
        return docs_with_scores
        

    def get_all_text_snippets_by_icd_code(self, icd_code: str):

        results = self.vector_store._collection.get(where={"ICD_CODE": icd_code})
        
        return results["documents"]