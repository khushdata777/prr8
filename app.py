import os
import streamlit as st
import tempfile

# Disable telemetry and tokenizer parallelism warnings
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Hugging Face token from Streamlit secrets
HUGGINGFACEHUB_API_TOKEN = os.getenv("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")

st.set_page_config(page_title="📄 Chat with PDF/TXT")
st.title("📄 Chat with Your PDF or Text File")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
user_query = st.text_input("Ask a question about the document:")

if uploaded_file and user_query:
    # Save uploaded file to a temporary path
    suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load the document
    loader = PyPDFLoader(file_path) if suffix == ".pdf" else TextLoader(file_path)
    documents = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Use Hugging Face's API for embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    # Use Chroma with a temporary directory
    persist_directory = tempfile.mkdtemp()
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)

    # Use Hugging Face's FLAN-T5 for LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.3, "max_length": 256},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    # Retrieval-based QA
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    answer = qa_chain.run(user_query)

    st.markdown("### 💬 Answer:")
    st.write(answer)
