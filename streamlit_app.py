import os
import streamlit as st # Easy wen app framework

# Langchain for Dev

## LLM 
from langchain_openai import ChatOpenAI
from langchain_community.llms import Bedrock

# Langchain for Ops
from langchain import hub # Prompt managing from the langchainhub site


### Module
import RAG_Milvus as rm

############### Special variables for Stramlit

# Langchain streamlit env
os.environ['LANGCHAIN_TRACING_V2']= 'True'
os.environ['LANGCHAIN_API_KEY']= st.secrets["LANGCHAIN_API_KEY"]
os.environ['LANGCHAIN_ENDPOINT']= st.secrets['LANGCHAIN_ENDPOINT']

# LLM API streamlit env
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ['AWS_DEFAULT_REGION'] = st.secrets["AWS_DEFAULT_REGION"]

# RAG stramlit env
os.environ['CHUNK_SIZE'] = st.secrets["CHUNK_SIZE"]
os.environ['MODEL_NAME'] = st.secrets["MODEL_NAME"]
os.environ['EMBEDDING_MODEL_NAME'] = st.secrets["EMBEDDING_MODEL_NAME"]
os.environ['BASE_FILE_PATH'] = st.secrets["BASE_FILE_PATH"]

# Milvus stramlit env
os.environ['MILVUS_TOKEN'] = st.secrets["MILVUS_TOKEN"]
os.environ['MILVUS_URI'] = st.secrets["MILVUS_URI"]
os.environ['COLLECTION_NAME'] = st.secrets["COLLECTION_NAME"]

############### From env variables for using function


# RAG env
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
BASE_FILE_PATH= os.getenv("BASE_FILE_PATH")

# Milvus env
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_URI = os.getenv("MILVUS_URI")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_ARGS = {'uri': MILVUS_URI, 'token': MILVUS_TOKEN}





st.set_page_config(page_title="Murphy's Library")
st.title("Murphy's Library") #페이지 제목

model_name1 = 'gpt-3.5-turbo-0125'
model_name2 = 'gpt-4-0125-preview'
model_name3 = 'amazon.titan-text-express-v1'

llm_model_openai_gpt3_5 = ChatOpenAI(model_name=model_name1, temperature=0) 
llm_model_openai_gpt4 = ChatOpenAI(model_name=model_name2, temperature=0) 
llm_model_aws_bedrock = Bedrock(model_id = 'amazon.titan-text-express-v1', region_name="us-east-1")


if 'initial' not in st.session_state:
    st.session_state['initial'] = True


# Use sidebar for model selection
with st.sidebar:
    st.header("Select Model")
    
    st.subheader("OpenAI")
    openai_choice = st.radio ("OpenAI models", ("none",model_name1,model_name2))
    
    st.subheader("AWS Bedrock")
    aws_bedrock_choice = st.radio ("AWS Bedrock models", ["none"])
    
    # Enable and disable by password for Admin status
    pwd = st.text_input('Please enter your password to enable Admin tab', type='password')
    
    if pwd == st.secrets["PASSWORD"]:
        st.session_state['admin_button'] = False
    else:
        st.session_state['admin_button'] = True     

    admin_status = st.button("Admin",disabled=st.session_state.admin_button,type="primary")
    






# Main area: tabs for query input and results
tab1, tab2, tab3 = st.tabs(["OpenAI", "AWS Bedrock", "Admin"])

# Define checkboxes for user choices
query = st.text_input("**Give me a question!**" ,placeholder="Enter your question")
go_button = st.button("Go", type="primary")
prompt_template = hub.pull("murphy/librarian_guide")


if go_button:
    with st.spinner("Working..."):
        # Initialize your language model based on user choice
        if openai_choice:
            with tab1:
                st.header("OpenAI")
                if openai_choice == "none":
                    st.error("OpenAI models are not selected")
                else:
                    if openai_choice == model_name1:
                        llm = llm_model_openai_gpt3_5 
                    elif openai_choice == model_name2:
                        llm = llm_model_openai_gpt4
                    
                    history, query, answer, session = rm.Milvus_chain("What is number about EDA?",llm,prompt_template)

                    
                    st.write(answer)
                    
                with st.expander(label="Chat History", expanded=False):
                    st.write(st.session_state.initial)
                
        if aws_bedrock_choice:
            with tab2:
                st.header("AWS Bedrock")
                if aws_bedrock_choice == "none":
                    st.error("AWS Bedrock models are not Supported yet")
                
                else:
                    llm = llm_model_aws_bedrock
                    history, query, answer, session = rm.Milvus_chain("What is number about EDA?",llm,prompt_template)

                    st.write(answer)
                    
                with st.expander(label="Chat History", expanded=False):
                    st.write(st.session_state.initial)
if admin_status:
    with tab3:
        st.header("Admin")
        st.write("Admin active")
    