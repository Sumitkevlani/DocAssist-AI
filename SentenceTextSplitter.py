import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class SentenceTextSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-self.chunk_overlap:]
                current_length = sum(len(sent) for sent in current_chunk)
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
