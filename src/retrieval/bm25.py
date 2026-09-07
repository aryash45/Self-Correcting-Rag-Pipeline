from rank_bm25 import BM25Okapi
import string 
class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        tokenized_docs = [self._tokenize(doc['content']) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.doc_ids = [doc['id'] for doc in documents]

    def _tokenize(self, text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def retrieve(self, query, k=5, topk=None):
        if topk is not None:
            k = topk
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_k_indices]



BM25Retriver = BM25Retriever

