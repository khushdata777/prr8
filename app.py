import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os
import tempfile

# Load Hugging Face API key from environment (set in Streamlit Cloud secrets)
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.title("📄 PDF/Text Chatbot with Hugging Face")

# Upload a PDF or Text file
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
user_question = st.text_input("Ask a question about the document:")

if uploaded_file and user_question:
    # Save uploaded file to a temporary file
    suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load the file content using appropriate loader
    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # Split content into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embed the chunks using Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # Set up the retriever
    retriever = vectorstore.as_retriever()

    # Initialize the Hugging Face model (flan-t5-base, for example)
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        model_kwargs={"temperature": 0.5, "max_length": 256},
    )

    # Create the QA chain using the retriever and model
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Get the answer to the user's question
    answer = qa.run(user_question)
    st.write("💬 Answer:", answer)
