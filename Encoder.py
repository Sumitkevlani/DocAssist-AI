import torch
import faiss
import pickle
import numpy as np

class Encoder:
    _instance = None
    
    def __new__(cls, tokenizer=None, model=None, device="cpu"):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.tokenizer = tokenizer
            cls._instance.model = model
            cls._instance.device = device
        return cls._instance
    
    def compute_bert_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        embeddings = torch.mean(last_hidden_state, dim=1).squeeze().numpy()
        return embeddings
    
    def store_embeddings_in_faiss(self, splitted_text, index_filename):
        embedding_dim = 768  # Bert base model output size
        index = faiss.IndexFlatL2(embedding_dim)
        
        docstore = {}
        index_to_docstore_id = {}
        idx = 0
        
        for chunk in splitted_text:
            text = chunk.page_content
            embeddings = self.compute_bert_embeddings(text)
            embeddings = np.array(embeddings, dtype=np.float32).reshape(1, -1)
            index.add(embeddings)
            
            # Assuming chunk has a unique identifier like doc_id
            doc_id = f"doc_{idx}"
            docstore[doc_id] = {
                'text': text,
                'source': chunk.metadata['source']  # Example metadata field
            }
            index_to_docstore_id[idx] = doc_id
            idx += 1
        
        # Save FAISS index
        faiss.write_index(index, index_filename + ".faiss")
        
        # Save associated metadata (docstore and index_to_docstore_id)
        metadata = {
            'docstore': docstore,
            'index_to_docstore_id': index_to_docstore_id
        }
        
        with open(index_filename + ".pkl", "wb") as f:
            pickle.dump(metadata, f)
