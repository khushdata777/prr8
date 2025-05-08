import tempfile
import time
import os
from torch import cuda, bfloat16
import transformers
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template



def get_pdfs(loaders):

    docs=[]
    temp_dir = tempfile.TemporaryDirectory()
    for loader in loaders:
        temp_filepath = os.path.join(temp_dir.name, loader.name)
        with open(temp_filepath, "wb") as f:
            f.write(loader.getvalue())
        pdf = PyPDFLoader(temp_filepath)

        docs.extend(pdf.load())

    return docs

def split_docs_into_chunks(docs_list):

    text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", "(?<=\. )" ," ", ""]
    )
    chunks_list=text_splitter.split_documents(docs_list)
    return chunks_list

def get_vector_indices(chunks):

    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    embed_model = HuggingFaceEmbeddings(model_name=embed_model_id)

    pinecone.init(
    api_key=os.environ['pinecone_key'],
    environment=os.environ['pinecone_env']
    )

    index_name = 'llama-2-rag'

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=384,
            metric='cosine'
            )
    # wait for index to finish initialization
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)

    index = pinecone.Index(index_name)

    index = Pinecone.from_documents(chunks, embed_model, index_name=index_name)
    text_field = "text"

    # switch back to normal index for langchain
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(
        index, embed_model.embed_query, text_field
        )

    return vectorstore

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def get_llm(model_id):


    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
        )

    # begin initializing HF items, need auth token for these

    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=os.environ['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']
        )

    llm = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=os.environ['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']
        )

    llm.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=os.environ['']
    )



    generate_text = transformers.pipeline(
    model=llm, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
                        )

    llm = HuggingFacePipeline(pipeline=generate_text)

    return llm



def get_conversation_chain(llm,vectorstore):




    memory = ConversationBufferMemory( memory_key='chat_history', return_messages=True )
    conversation_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=vectorstore.as_retriever(),
                        memory=memory
                        )

    return conversation_chain




def main():
    load_dotenv()
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    st.set_page_config(page_title="Langchain: Chat with multiple PDFs", page_icon=":books:")
    #st.write(css, unsafe_allow_html=True)


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents:")
        loaders = st.file_uploader("Upload your PDFs here and click on 'Upload'.", accept_multiple_files=True)

        if st.button('Upload'):
            with st.spinner('Work under Progress'):

                # get the text from the pdfs
                raw_docs=get_pdfs(loaders)

                # get the chunks of text
                chunks= split_docs_into_chunks(raw_docs)
                # st.write(chunks)

                #get vectorstore
                vectorstore = get_vector_indices(chunks)
                
                #Name of the model to be use from huggingface
                model_id='meta-llama/Llama-2-13b-chat-hf'

                #get LLM
                llm=get_llm(model_id)

                #get conversation
                st.session_state.conversation = get_conversation_chain(llm,
                    vectorstore)


if __name__ == '__main__':
    main()
