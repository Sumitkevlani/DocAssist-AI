from Generator import Generator 

class CustomGenerator(Generator):
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            model_factory = kwargs.get('model_factory')
            model_dir = kwargs.get('model_dir')
            model_type = kwargs.get('model_type')
            
            tokenizer, model = model_factory.generate_model(model_dir, model_type)
            
            # Initialize superclass (Generator) with tokenizer and model
            cls._instance = super().__new__(cls, tokenizer, model)
            cls._instance.model_factory = model_factory
            cls._instance.model_dir = model_dir
            cls._instance.model_type = model_type
        
        return cls._instance

    def generate_answer(self, retrieved_documents, query):
        return super().generate(retrieved_documents=retrieved_documents, query=query)
