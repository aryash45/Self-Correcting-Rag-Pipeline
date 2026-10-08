from src.retrieval.denseretriever import DenseRetriever
from src.retrieval.bm25 import BM25Retriever


class HybridRetriever:
    def __init__(self, documents, bm25_weight=0.5, dense_weight=0.5):
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(documents)

        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def retrieve(self, query, k=5):
        bm25_results = self.bm25.retrieve(query, k=k)
        dense_results = self.dense.retrieve(query, k=k)

        bm25_scores = [
            result["score"] for result in bm25_results
        ]

        bm25_min = min(bm25_scores)
        bm25_max = max(bm25_scores)

        if bm25_max == bm25_min:
            bm25_norm = {
                result["document"]["id"]: 1.0
                for result in bm25_results
            }
        else:
            bm25_norm = {
                result["document"]["id"]:
                    (result["score"] - bm25_min) /
                    (bm25_max - bm25_min)
                for result in bm25_results
            }

        dense_scores = [
            result["score"] for result in dense_results
        ]

        dense_min = min(dense_scores)
        dense_max = max(dense_scores)

        if dense_max == dense_min:
            dense_norm = {
                result["document"]["id"]: 1.0
                for result in dense_results
            }
        else:
            dense_norm = {
                result["document"]["id"]:
                    (result["score"] - dense_min) /
                    (dense_max - dense_min)
                for result in dense_results
            }

        all_documents = {}

        for result in bm25_results:
            doc_id = result["document"]["id"]
            all_documents[doc_id] = result["document"]

        for result in dense_results:
            doc_id = result["document"]["id"]
            all_documents[doc_id] = result["document"]

        hybrid_scores = {}

        for doc_id in all_documents:
            bm25_score = bm25_norm.get(doc_id, 0.0)
            dense_score = dense_norm.get(doc_id, 0.0)

            hybrid_scores[doc_id] = (
                self.bm25_weight * bm25_score +
                self.dense_weight * dense_score
            )

        ranked_docs = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for doc_id, score in ranked_docs[:k]:
            results.append({
                "document": all_documents[doc_id],
                "score": score
            })

        return results