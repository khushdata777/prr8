import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os
import tempfile

# Hugging Face API key setup
HUGGINGFACEHUB_API_TOKEN = os.getenv("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")

# Streamlit UI
st.title("📄 PDF/Text Chatbot using Hugging Face")
uploaded_file = st.file_uploader("https://github.com/khushdata777/prr8/blob/main/datasci.txt", type=["pdf", "txt"])
user_question = st.text_input("Ask a question about the document:")

if uploaded_file and user_question:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    documents = loader.load()

    # Split and embed text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever()

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 256},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    response = qa.run(user_question)
    st.write("💬 Answer:", response)
