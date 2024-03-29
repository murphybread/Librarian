import os
import streamlit as st # Easy wen app framework

# Langchain for Dev

## LLM 
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.llms import Bedrock

# Langchain for Ops
from langchain import hub # Prompt managing from the langchainhub site


### Module
import RAG_Milvus as rm


############## Streamlit Dynamic function
def submit():
    st.session_state.manual_session_value = st.session_state.widget
    st.session_state.widget = ''


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



url = "https://www.murphybooks.me"
st.set_page_config(
    page_title="Murphy's Library",
    menu_items={
        'About': url
        
    })
st.title(f"[Murphy's Library]({url})") # Page title









model_name1 = 'gpt-3.5-turbo-0125'
model_name2 = 'gpt-4-0125-preview'
model_name3 = 'amazon.titan-text-express-v1'
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

llm_model_openai_gpt3_5 = ChatOpenAI(model_name=model_name1, temperature=0) 
llm_model_openai_gpt4 = ChatOpenAI(model_name=model_name2, temperature=0) 
llm_model_aws_bedrock = Bedrock(model_id = 'amazon.titan-text-express-v1', region_name="us-east-1")


if 'initial' not in st.session_state:
    st.session_state['initial'] = True

if 'admin_status' not in st.session_state:
    st.session_state['admin_status'] = False  # Initialize the admin status

if 'manual_session_value' not in st.session_state:
    st.session_state.manual_session_value = ''


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
        st.session_state['admin_button'] = True
    else:
        st.session_state['admin_button'] = False
    
    # record_button = st.button("Record", type="primary") 



# Main area: tabs for query input and results
tab1, tab2, tab3 = st.tabs(["OpenAI", "AWS Bedrock", "Admin"])





# Define checkboxes for user choices
question = st.text_input("**Give me a question!**" , placeholder="Enter your question")

manual_session_toggle = st.toggle("I have memory session")
if manual_session_toggle:
    manual_session = st.text_input("**If you have information about the last session, you can continue the previous conversation.**" ,placeholder="Enter your Session string",key='widget',on_change=submit)
go_button = st.button("Go", type="primary")


prompt_template = hub.pull("murphy/librarian_guide")








if go_button:
    with st.spinner("Working..."):

       # OpenAI models
        if openai_choice and openai_choice != "none":
            with tab1:
                
                llm = llm_model_openai_gpt3_5 if openai_choice == model_name1 else llm_model_openai_gpt4

                
                
            
                if 'next_session' in st.session_state:    
                    memory_session = st.session_state['next_session']
                    st.write("Memory Session: " + memory_session)
                elif manual_session:
                    memory_session = manual_session
                    st.wrtie("Manuall Memory Session " + manual_session)
                    manual_session_toggle = False
                else:
                    memory_session= ''
                    st.write("No Memory Session: " + memory_session)
                
                
                
                    
                    
                
                history, query, answer, session = rm.Milvus_chain(question, llm, prompt_template, memory_session)
                st.markdown(answer)
                st.session_state['next_session'] = session
                
                
                with st.expander(label="Chat History", expanded=False):
                    st.text(history)

        elif openai_choice:
            with tab1:
                st.error("OpenAI models are not selected")

        # AWS Bedrock models
        if aws_bedrock_choice and aws_bedrock_choice != "none":
            with tab2:
                st.header("AWS Bedrock")
                llm = llm_model_aws_bedrock
                history, query, answer, session = rm.Milvus_chain(query, llm, prompt_template)

                with st.expander(label="Chat History", expanded=False):
                    st.write(st.session_state.initial)
        elif aws_bedrock_choice:
            with tab2:
                st.error("AWS Bedrock models are not supported yet")

        
col1, col2, col3 = st.columns(3)

if st.session_state['admin_button']:
    
    with tab3:
        st.header("Admin activated")
        
        with col1:
            st.header("Create VectorDB")
    
            create_button = st.button('Create VectorDB', type="primary")
            
            if create_button:
                with st.spinner("Embedding Started"):
                    rm.create_collection()
                st.success('Embedding Done')
        with col2:
            st.header("Update single entity")
            
            file_path = st.text_input( " **Write pifePath**" , placeholder="Enter your question")
            upsert_button = st.button('Upsert Entity of DB', type="primary")
            if upsert_button:
                with st.spinner("Upsert Started"):
                    st.write(file_path)
                    entitiy_memory = rm.MilvusMemory(embeddings,uri=MILVUS_URI, token=MILVUS_TOKEN, collection_name=COLLECTION_NAME)
                    entitiy_memory.update_entity(file_path, entitiy_memory.vectorstore)
                st.success('Upsert Done')
        with col3:
            st.header("Update base template")
            
            
            base_button = st.button('Upsert base template to Vector DB', type="primary")
            if upsert_button:
                with st.spinner("Upsert base template"):
                    file_path = './base_template.md'
                    st.write(file_path)
                    entitiy_memory = rm.MilvusMemory(embeddings,uri=MILVUS_URI, token=MILVUS_TOKEN, collection_name=COLLECTION_NAME)
                    entitiy_memory.update_entity(file_path, entitiy_memory.vectorstore)
                st.success('Update Base temlplate Done')
            
            


    