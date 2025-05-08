import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Load API key from Streamlit Secrets
HUGGINGFACEHUB_API_TOKEN = os.getenv("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")

st.set_page_config(page_title="📄 PDF/Text Chatbot")
st.title("📄 Chat with your PDF/Text using Hugging Face 🤗")

# Upload the file
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
question = st.text_input("Ask a question about the file content:")

if uploaded_file and question:
    # Save uploaded file temporarily
    suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load file with appropriate loader
    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Use Hugging Face hosted embeddings (works with Streamlit Cloud)
    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    # Create Chroma vector store in a temp directory
    persist_directory = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    retriever = vectorstore.as_r_
