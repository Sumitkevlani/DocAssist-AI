from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFChatbot:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    

    def __init__(self, encoder, retriever, generator):
        if not hasattr(self, 'initialized'):
            self.encoder = encoder
            self.retriever = retriever
            self.generator = generator
            self.splitted_text = None
            self.initialized = True
    
    def encode_pdf(self, path):
        loader = TextLoader(path)
        output = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=30,separators=["\n\n","\n", "."])
        splitted_text = text_splitter.split_documents(documents=output)
        self.splitted_text = splitted_text
        self.encoder.store_embeddings(splitted_text=splitted_text,index_filename="index")

    def resolve_query(self, query):
        if not self.splitted_text:
            raise ValueError("No PDF has been encoded yet. Call encode_pdf first.")
        
        documents = self.retriever.retrieve_documents(splitted_text=self.splitted_text,query=query)
        predicted_answer = self.generator.generate_answer(retrieved_documents=documents,query=query)
        return predicted_answer