from CustomEncoder import CustomEncoder
from CustomDecoder import CustomGenerator
from Retriever import CustomRetriever
from PDFChatbot import PDFChatbot
from ModelFactory import ModelFactory
import streamlit as st


st.title('Contract Analysis Chatbot')

# Initialize necessary objects
encoder_model_dir = "bert"
qna_model_dir = "bert-for-qna"
index_filename = "index.faiss"
model_factory = ModelFactory()

encoder = CustomEncoder(model_factory=model_factory, model_dir=encoder_model_dir, model_type='encoder')
generator = CustomGenerator(model_factory=model_factory, model_dir=qna_model_dir, model_type='generator')
retriever = CustomRetriever(index_filename=index_filename, encoder=encoder)
chatbot = PDFChatbot(encoder=encoder, retriever=retriever, generator=generator)

# Process the example file
example_file_path = 'contract.txt' 

st.header('Processing Example File')
st.write(f"Using PDF file: {example_file_path}")

# Encode the example file
chatbot.encode_pdf(example_file_path)


st.success("Example PDF successfully encoded!")

# Provide an option to ask questions
st.header('Ask Question')
query = st.text_area('Enter your question')
if st.button('Submit'):
    answer = chatbot.resolve_query(query)
    st.info(f"Answer: {answer}")

