from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering

class ModelFactory:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def generate_model(self, model_dir, model_type):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if model_type == 'encoder':
            model = AutoModel.from_pretrained(model_dir)
        elif model_type == 'generator':
            model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        else:
            raise ValueError("Invalid model type. Choose 'encoder' or 'generator'.")
        return tokenizer, model
