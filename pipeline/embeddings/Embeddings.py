from langchain_core.embeddings import Embeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List


class AdapterEmbeddings(Embeddings):
    def __init__(self, embeddingFunction: SentenceTransformerEmbeddingFunction):
        self.embeddingFunction = embeddingFunction

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddingFunction(texts)

    def embed_query(self, text: str) -> List[float]:
        # embed_query can call ef([text]) and return the first embedding
        return self.embeddingFunction([text])[0]


class EmbeddingFunctionGPU():

    def __init__(self):

        self.EMB_MODEL_NAME = 'all-MiniLM-L6-v2'

        self.embedding_model = HuggingFaceEmbeddings(model=self.EMB_MODEL_NAME,model_kwargs={"device": "cuda"})
        
        self.gpu_embedding = SentenceTransformerEmbeddingFunction(model_name=self.EMB_MODEL_NAME, device="cuda")
        self.gpu_embedding = AdapterEmbeddings(self.gpu_embedding)

    def get_emb_gpu_model(self):
        return self.gpu_embedding