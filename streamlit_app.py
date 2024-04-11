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
# def submit():
#     st.session_state.widget = ''


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




# Setting for statue session 
if 'initial' not in st.session_state:
    st.session_state['initial'] = True

if 'admin_status' not in st.session_state:
    st.session_state['admin_status'] = False  # Initialize the admin status

if 'manual' not in st.session_state:
    st.session_state['manual'] = ''

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
    st.session_state.next_session = ""


st.set_page_config(
    page_title="Murphy's Library",
    menu_items={
        'About': url
        
    })
st.title(f"[Murphy's Library]({url})\n **Charles, the librarian, will help you find information in Murphy's library.\nHe'll give you general answers in the first conversation.** \n Use the special pattern 'file_path: ' to find more details") # Page title

st.info("Your memory session " + st.session_state['next_session'], icon="ℹ️")







model_name1 = 'gpt-4-0125-preview'

model_name2 = 'gpt-3.5-turbo-0125'
model_name3 = 'amazon.titan-text-express-v1'
# embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)


# llm_model_openai_gpt3_5 = ChatOpenAI(model_name=model_name1, temperature=0) 
# llm_model_openai_gpt4 = ChatOpenAI(model_name=model_name2, temperature=0) 
llm_model_aws_bedrock = "S"#Bedrock(model_id = 'amazon.titan-text-express-v1', region_name="us-east-1")





# Use sidebar for model selection
with st.sidebar:
    st.header("Select Model")
    
    st.subheader("OpenAI")
    openai_choice = st.radio ("OpenAI models", (model_name1,"none"))
    
    st.subheader("AWS Bedrock **(Not Supported)**")
    aws_bedrock_choice = st.radio ("AWS Bedrock models", ["none"])
    
    # Enable and disable by password for Admin status
    pwd = st.text_input('Please enter your password to enable Admin tab', type='password')
    
    if pwd == st.secrets["PASSWORD"]:
        st.session_state['admin_button'] = True
    else:
        st.session_state['admin_button'] = False
    
    
    manual_session = st.text_input("If you have information about the last session, you can continue the previous conversation.", placeholder="Enter your Session string", key='widget')
    st.write(f'manual: {manual_session}')
    


# Main area: tabs for query input and results
tab1, tab2, tab3 = st.tabs(["OpenAI", "AWS Bedrock **(Not Supported)**", "Admin"])




prompt_template = hub.pull("murphy/librarian_guide")


for message in st.session_state.messages:    
    if message["role"] == 'user':
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"],avatar='50486329.png'):
            st.markdown(message["content"])


@st.cache_resource
def load_llm(model_name):
    return ChatOpenAI(model_name=model_name, temperature=0.05)

@st.cache_resource
def load_embeddings(model_name):
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_milvus(_embedding):
    milvus = rm.MilvusMemory(_embedding,MILVUS_URI,MILVUS_TOKEN,COLLECTION_NAME)
    return  milvus

# OpenAI models LLM and Embedding
if openai_choice != "none":
    llm = load_llm(model_name1)
    embeddings = load_embeddings(EMBEDDING_MODEL_NAME)
    milvus = load_milvus(embeddings)
elif openai_choice:
    st.error("OpenAI models are not selected")




def get_respone(llm,milvus,query,session, file_path_session):
    
        
    history, question, answer, session = milvus.Milvus_chain(query, llm, prompt_template, session,file_path_session)
    
    # st.markdown(history)
    # st.markdown(question)
    # st.markdown(answer)
    # st.markdown(session)
    
    # if 'Answer:' in answer:
    #     return answer.replace('Answer:', ''), session
    
    return answer,session






if prompt:= st.chat_input("If you have any questions, can you write them here?"):
    if st.session_state['next_session']:
        memory_session = st.session_state['next_session']
    else:
        memory_session = ''
    if len(manual_session) >= 35 :
        memory_session = manual_session
        
        
    
        
    
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        info_file_path =  rm.extract_path(prompt)
        
        
        response, session = get_respone(llm,milvus,prompt , memory_session, info_file_path)
        # st.write(f"memory_session: {memory_session}")
        # st.write(f"session: {session}")
        st.session_state['next_session'] = session
    
    with st.chat_message("assistant",avatar='50486329.png'):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    
                
# st.chat_message("Charles")

# if go_button:
#     with st.spinner("Working..."):
#         if 'next_session' in st.session_state:    
#             memory_session = st.session_state['next_session']
#             st.write("Memory Session: " + memory_session)
#         elif len(manual_session) > 2:
#             memory_session = manual_session
#             st.write("Manuall Memory Session " + memory_session)
#             manual_session =  ''                    
#         else:
#             memory_session= ''
#             st.write("No Memory Session: " + memory_session)    
        
#         file_path_session = rm.extract_path(query)
#         milvus_class = rm.MilvusMemory(embeddings,MILVUS_URI,MILVUS_TOKEN,COLLECTION_NAME)

#         st.write("File path Session: " + file_path_session)
         
        
#         history, question, answer, session = milvus_class.Milvus_chain(query, llm, prompt_template, memory_session,file_path_session)
        
        
#         st.session_state['question'] = question
#         st.session_state['answer'] = answer
#         st.session_state['history'] = history
#         st.session_state['next_session'] = session
        
        
#         st.text(st.session_state['answer'])
        


# if st.session_state['history']:
#     with st.expander(label="Chat History", expanded=False):
#         st.text(st.session_state['history'])

#     # # AWS Bedrock models
#     # if aws_bedrock_choice and aws_bedrock_choice != "none":
#     #     with tab2:
#     #         st.header("AWS Bedrock")
#     #         llm = llm_model_aws_bedrock
#     #         history, query, answer, session = rm.Milvus_chain(query, llm, prompt_template)


#     # elif aws_bedrock_choice:
#     #     with tab2:
#     #         st.error("AWS Bedrock models are not supported yet")

        
col1, col2, col3, col4 = st.columns(4)

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
            st.header("**Update** Entitys")
            
            file_path = st.text_input( "**Update** Entity: Write filePath" , placeholder="Enter your filePath")
            upsert_button = st.button('Upsert Entity of DB', type="primary")
            if upsert_button:
                with st.spinner("Upsert Started"):
                    st.write(file_path)
                    entitiy_memory = rm.MilvusMemory(embeddings,uri=MILVUS_URI, token=MILVUS_TOKEN, collection_name=COLLECTION_NAME)
                    entitiy_memory.update_entity(file_path, entitiy_memory.vectorstore)
                st.success('Upsert Done')
                
        with col3:
            st.header("**Create** Entitys")
            
            file_path = st.text_input( "**Create** Entity: Write fieePath" , placeholder="Enter your filePath")

            
            create_button = st.button('Create Entity of DB', type="primary")
            if create_button:
                with st.spinner("Create Started"):
                    st.write(file_path)
                    entitiy_memory = rm.MilvusMemory(embeddings,uri=MILVUS_URI, token=MILVUS_TOKEN, collection_name=COLLECTION_NAME)
                    entitiy_memory.create_or_update_collection(file_path)
                st.success('Create Done')
                
        with col4:
            st.header("Delete entity")
            
            
            base_button = st.button('Delete entity', type="primary")
            if base_button:
                with st.spinner("Delete entity...."):
                    pk_id = st.input("input opk")
                    rm.delete_entity(pk_id)
                    
                st.success('Delete entity succed '+ pk_id)
            


    