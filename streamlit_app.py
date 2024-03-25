import os
import streamlit as st #모든 streamlit 명령은 "st" 별칭을 통해 사용할 수 있습니다.
import RAG as rag #로컬 라이브러리 스크립트 참조
import hmac



# import Lanchaing_Milvus as LM


os.environ['LANGCHAIN_TRACING_V2']= 'True'
os.environ['LANGCHAIN_API_KEY']= st.secrets["LANGSMITH"]["LANGSMITH_API"]
os.environ['LANGCHAIN_ENDPOINT']= st.secrets["LANGSMITH"]['LANGCHAIN_ENDPOINT']

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI"]["OPENAI_API_KEY"]
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["AWS"]["AWS_ACCESS_KEY_ID"]
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["AWS"]["AWS_SECRET_ACCESS_KEY"]
os.environ['AWS_DEFAULT_REGION'] = st.secrets["AWS"]["AWS_DEFAULT_REGION"]

os.environ['MILVUS_TOKEN'] = st.secrets["MILVUS"]["MILVUS_TOKEN"]
os.environ['MILVUS_URI'] = st.secrets["MILVUS"]["MILVUS_URI"]

st.set_page_config(page_title="Murphy's Library")
st.title("Murphy's Library") #페이지 제목


model_name1 = 'gpt-3.5-turbo-0125'
model_name2 = 'gpt-4-0125-preview'
model_name3 = 'amazon.titan-text-express-v1'



llm_model_openai_gpt3_5 = rag.get_llm_openai(model_name1)
llm_model_openai_gpt4 = rag.get_llm_openai(model_name2)
llm_model_aws_bedrock = rag.get_llm_aws_bedrock(model_name3)


if 'initial' not in st.session_state:
    st.session_state['initial'] = True


# Use sidebar for model selection
with st.sidebar:
    st.header("Select Model")
    
    st.subheader("OpenAI")
    openai_choice = st.radio ("OpenAI models", ("none",model_name1,model_name2))
    
    st.subheader("AWS Bedrock")
    aws_bedrock_choice = st.radio ("AWS Bedrock models", ["none", model_name3])
    
    # Enable and disable by password for Admin status
    pwd = st.text_input('input your password', type='password')
    
    if pwd == st.secrets["STREAMLIT"]["PASSWORD"]:
        st.session_state['admin_button'] = False
    else:
        st.session_state['admin_button'] = True     

    admin_status = st.button("Admin",disabled=st.session_state.admin_button,type="primary")
    






# Main area: tabs for query input and results
tab1, tab2, tab3 = st.tabs(["OpenAI", "AWS Bedrock", "Admin"])

# Define checkboxes for user choices
query = st.text_input("**Give me a question!**" ,placeholder="Enter your question")
go_button = st.button("Go", type="primary")

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
                        llm_model = llm_model_openai_gpt3_5
                    elif openai_choice == model_name2:
                        llm_model = llm_model_openai_gpt4
                        
                    response_content = rag.get_text_response(query, llm_model)
                    st.write(response_content)
                    
                with st.expander(label="Chat History", expanded=False):
                    st.write(st.session_state.initial)
                
        if aws_bedrock_choice:
            with tab2:
                st.header("AWS Bedrock")
                if aws_bedrock_choice == "none":
                    st.error("AWS Bedrock models are not selected")
                
                else:
                    llm_model = llm_model_aws_bedrock
                    response_content = rag.get_text_response(query, llm_model)
                    st.write(response_content)
                    
                with st.expander(label="Chat History", expanded=False):
                    st.write(st.session_state.initial)
if admin_status:#pwd == st.secrets['STREAMLIT']['password']:
    with tab3:
        st.header("Admin")
        st.write("Admin active")
    