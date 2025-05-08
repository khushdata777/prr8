import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Load API key securely from environment
HUGGINGFACEHUB_API_TOKEN = os.getenv("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")

st.set_page_config(page_title="📄 Chat with PDF or TXT")
st.title("📄 Chat with your PDF or Text file (Free HF-based Chatbot)")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
user_query = st.text_input("Ask something about your document:")

if uploaded_file and user_query:
    # Save file temporarily
    suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load file
    loader = PyPDFLoader(file_path) if suffix == ".pdf" else TextLoader(file_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Create HuggingFace embeddings via endpoint
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    # Create Chroma vector DB (in memory)
    persist_directory = tempfile.mkdtemp()
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)

    # Load LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        model_kwargs={"temperature": 0.3, "max_length": 256}
    )

    # QA chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    response = qa.run(user_query)

    st.markdown("### 💬 Response:")
    st.write(response)
