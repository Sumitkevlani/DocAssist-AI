import re
import torch

class Generator:
    _instance = None
    
    def __new__(cls, tokenizer=None, model=None, device="cpu"):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.tokenizer = tokenizer
            cls._instance.model = model
            cls._instance.device = device
        return cls._instance

    def generate(self, retrieved_documents, query):
        contexts = [doc.page_content for doc in retrieved_documents]
        combined_context = " ".join(contexts)

        combined_context = re.sub(r'\n\s*\n', '\n\n', combined_context).strip()

        question = query
        text = combined_context

        inputs = self.tokenizer(question, text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the predicted start and end indices for the answer
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        # Extract the tokens corresponding to the predicted answer span
        predicted_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

        # Decode the tokens to get the predicted answer text
        predicted_answer = self.tokenizer.decode(predicted_answer_tokens, skip_special_tokens=True)
        print("Predicted Answer:", predicted_answer)

        # Define the target answer indices (for loss calculation)
        target_start_index = torch.tensor([14])  # Modify as per your target answer start index
        target_end_index = torch.tensor([15])    # Modify as per your target answer end index

        # Compute the outputs including the loss
        outputs = self.model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
        loss = outputs.loss
        print("Loss:", round(loss.item(), 2))

        # For seeing the actual target text
        target_answer_tokens = inputs.input_ids[0, target_start_index : target_end_index + 1]
        target_answer = self.tokenizer.decode(target_answer_tokens, skip_special_tokens=True)
        print("Target Answer:", target_answer)

        return predicted_answer
