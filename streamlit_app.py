import os
import streamlit as st #모든 streamlit 명령은 "st" 별칭을 통해 사용할 수 있습니다.
import RAG as rag #로컬 라이브러리 스크립트 참조

import Embedding_Milvus as em


os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ['AWS_DEFAULT_REGION'] = st.secrets["AWS_DEFAULT_REGION"]

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
    


# Main area: tabs for query input and results
tab1, tab2 = st.tabs(["OpenAI", "AWS Bedrock"])

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
        if aws_bedrock_choice:
            with tab2:
                st.header("AWS Bedrock")
                if aws_bedrock_choice == "none":
                    st.error("AWS Bedrock models are not selected")
                
                else:
                    llm_model = llm_model_aws_bedrock
                    response_content = rag.get_text_response(query, llm_model)
                    st.write(response_content)