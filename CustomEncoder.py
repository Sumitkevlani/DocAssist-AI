from Encoder import Encoder 

class CustomEncoder(Encoder):
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            model_factory = kwargs.get('model_factory')
            model_dir = kwargs.get('model_dir')
            model_type = kwargs.get('model_type')
            
            tokenizer, model = model_factory.generate_model(model_dir, model_type)
            cls._instance = super().__new__(cls, tokenizer, model)
            cls._instance.model_factory = model_factory
            cls._instance.model_dir = model_dir
            cls._instance.model_type = model_type
        
        return cls._instance
    
    def compute_embeddings(self, text):
        return super().compute_bert_embeddings(text)
    
    def store_embeddings(self, splitted_text, index_filename):
        return super().store_embeddings_in_faiss(splitted_text=splitted_text, index_filename=index_filename)
