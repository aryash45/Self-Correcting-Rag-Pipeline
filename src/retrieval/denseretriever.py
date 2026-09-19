from sentence_transformers import SentenceTransformer
import numpy as np 
class DenseRetriever:
    def __init__(self, documents, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.doc_embeddings = self.model.encode(
            [d['content'] for d in documents],
            normalize_embeddings= True
        )
    def retrieve(self,query,k=5):
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )
        similarities = np.dot(
            self.doc_embeddings,
            query_embedding
        )
        top_k_indices = np.argsort(similarities)[::-1][:k]
        results = []
        for idx in top_k_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(similarities[idx])
            })
        return results