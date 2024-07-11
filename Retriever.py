import faiss
import numpy as np

class CustomRetriever:
    _instance = None 
    
    def __new__(cls, index_filename, encoder):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.index_filename = index_filename
            cls._instance.encoder = encoder
        return cls._instance
    
    def retrieve_documents(self, splitted_text, query, top_k=3):
        # Implement your retrieval logic here
        index = faiss.read_index(self.index_filename)
        embeddings = self.encoder.compute_bert_embeddings(query)
        self.embeddings = np.array(embeddings, dtype=np.float32).reshape(1, -1)
        distances, indices = index.search(self.embeddings, top_k)
        reversed_indices = indices[0][::-1] 
        reversed_indices_2d = np.expand_dims(reversed_indices, axis=0)
        retrieved_documents = [splitted_text[i] for i in reversed_indices_2d[0]]
        return retrieved_documents
