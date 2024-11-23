# streamlit_app.py
import os
import uuid
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
import RAG_Milvus as rm

# Environment Variables
os.environ['LANGCHAIN_TRACING_V2'] = 'True'
os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY"]
os.environ['LANGCHAIN_ENDPOINT'] = st.secrets['LANGCHAIN_ENDPOINT']
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ['AWS_DEFAULT_REGION'] = st.secrets["AWS_DEFAULT_REGION"]
os.environ['CHUNK_SIZE'] = st.secrets["CHUNK_SIZE"]
os.environ['MODEL_NAME'] = st.secrets["MODEL_NAME"]
os.environ['EMBEDDING_MODEL_NAME'] = st.secrets["EMBEDDING_MODEL_NAME"]
os.environ['BASE_FILE_PATH'] = st.secrets["BASE_FILE_PATH"]
os.environ['MILVUS_TOKEN'] = st.secrets["MILVUS_TOKEN"]
os.environ['MILVUS_URI'] = st.secrets["MILVUS_URI"]
os.environ['COLLECTION_NAME'] = st.secrets["COLLECTION_NAME"]

# RAG env
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
BASE_FILE_PATH = os.getenv("BASE_FILE_PATH")

# Milvus env
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_URI = os.getenv("MILVUS_URI")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_ARGS = {'uri': MILVUS_URI, 'token': MILVUS_TOKEN}

# Streamlit Settings
if 'initial' not in st.session_state:
    st.session_state['initial'] = True

if 'admin_status' not in st.session_state:
    st.session_state['admin_status'] = False

if 'start' not in st.session_state:
    st.session_state['start'] = False

if 'question' not in st.session_state:
    st.session_state['question'] = False

if 'answer' not in st.session_state:
    st.session_state['answer'] = False

if 'history' not in st.session_state:
    st.session_state['history'] = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "next_session" not in st.session_state:
    st.session_state["next_session"] = str(uuid.uuid4())

st.set_page_config(
    page_title="Murphy's Library",
    menu_items={'About': "https://www.murphybooks.me"}
)
st.title(f"[Murphy's Library](https://www.murphybooks.me)\n **Charles, the librarian, will help you find information in Murphy's library.\nHe'll give you general answers in the first conversation.** \n Use the special pattern 'file_path: ' to find more details")
st.success("If you're unsure what to ask,simply say, Could you recommend a few posts?", icon="✅")
st.info("Your memory session " + st.session_state['next_session'], icon="ℹ️")

model_name1 = os.environ['MODEL_NAME']
model_name2 = 'gpt-3.5-turbo-0125'
model_name3 = 'amazon.titan-text-express-v1'
prompt_template = hub.pull("murphy/librarian_guide")

with st.sidebar:
    st.header("Select Model")
    st.subheader("OpenAI")
    openai_choice = st.radio("OpenAI models", (model_name1, "none"))
    st.subheader("AWS Bedrock **(Not Supported)**")
    aws_bedrock_choice = st.radio("AWS Bedrock models", ["none"])

    pwd = st.text_input('Please enter your password to enable Admin tab', type='password')
    st.session_state['admin_button'] = pwd == st.secrets["PASSWORD"]

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar='50486329.png' if message["role"] != 'user' else None):
        st.markdown(message["content"])

@st.cache_resource
def load_llm(model_name):
    return ChatOpenAI(model_name=model_name, temperature=0.05)

@st.cache_resource
def load_embeddings(model_name):
    return OpenAIEmbeddings(model=model_name)

@st.cache_resource
def load_milvus(_embedding):
    return rm.vectorstore_milvus(_embedding)

if openai_choice != "none":
    llm = load_llm(model_name1)
    embeddings = load_embeddings(EMBEDDING_MODEL_NAME)
    milvus = load_milvus(embeddings)
else:
    st.error("OpenAI models are not selected")

def get_response(llm, vectorstore, embeddings, collection_name, query, session, file_path_session):
    return rm.Milvus_chain(query, llm, prompt_template, vectorstore, embeddings, collection_name, session, file_path_session)

# Ensure session state for continuing conversations
if "next_session" not in st.session_state:
    st.session_state["next_session"] = str(uuid.uuid4())

if prompt := st.chat_input("If you have any questions, can you write them here?"):
    memory_session = st.session_state['next_session']
    
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        info_file_path = rm.extract_path(prompt)
        history, question, answer, session = get_response(llm, milvus, embeddings, COLLECTION_NAME, prompt, memory_session, info_file_path)
        st.session_state['next_session'] = session

    with st.chat_message("assistant", avatar='50486329.png'):
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Define tabs
tab1, tab2 = st.tabs(["OpenAI", "Admin"])

with tab1:
    st.header("OpenAI")
    # OpenAI tab content

if st.session_state['admin_button']:
    with tab2:
        st.header("Admin activated")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header("Create VectorDB")
            create_button = st.button('Create VectorDB', type="primary")
            if create_button:
                with st.spinner("Embedding Started"):
                    rm.create_collection()
                st.success('Embedding Done')
        with col2:
            st.header("**Update** Entities")
            file_path = st.text_input("**Update** Entity: Write filePath", placeholder="Enter your filePath")
            upsert_button = st.button('Upsert Entity of DB', type="primary")
            if upsert_button:
                with st.spinner("Upsert Started"):
                    st.write(file_path)
                    rm.create_or_update_collection(file_path)
                st.success('Upsert Done')
        with col3:
            st.header("**Create** Entities")
            file_path = st.text_input("**Create** Entity: Write filePath like ../../100", placeholder="Enter your filePath")
            create_button = st.button('Create Entity of DB', type="primary")
            if create_button:
                with st.spinner("Create Started"):
                    st.write(file_path)
                    rm.create_or_update_collection(file_path)
                st.success('Create Done')
        with col4:
            st.header("Delete entity")
            pk_id = st.text_input("Input Auto_id")
            base_button = st.button('Delete entity', type="primary")
            if base_button:
                with st.spinner("Delete entity...."):
                    if pk_id:
                        try:
                            auto_id = int(pk_id)
                            rm.delete_entity(auto_id)
                            st.success('Delete entity succeeded ' + str(auto_id))
                        except ValueError:
                            st.error('Please input a valid Auto_id as a number.')
                    else:
                        st.error('Please input a valid Auto_id.')
