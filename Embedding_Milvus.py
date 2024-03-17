import os
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda


from langchain.docstore.document import Document




from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, VectorStoreRetrieverMemory
from langchain.chains import ConversationChain


import streamlit as st

import uuid


#custom package
from milvus_memory import MilvusMemory



MILVUS_TOKEN=st.secrets["MILVUS"]["MILVUS_TOKEN"]
MILVUS_URI=st.secrets["MILVUS"]["MILVUS_URI"]

COLLECTION_NAME = "Library"
CONNECTION_ARGS = { 'uri': MILVUS_URI, 'token': MILVUS_TOKEN }


os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI"]["OPENAI_API_KEY"]
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["AWS"]["AWS_ACCESS_KEY_ID"]
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["AWS"]["AWS_SECRET_ACCESS_KEY"]
os.environ['AWS_DEFAULT_REGION'] = st.secrets["AWS"]["AWS_DEFAULT_REGION"]

os.environ['MILVUS_TOKEN'] = st.secrets["MILVUS"]["MILVUS_TOKEN"]
os.environ['MILVUS_URI '] = st.secrets["MILVUS"]["MILVUS_URI"]


CHUNK_SIZE= 15000


DEFAULT_MILVUS_CONNECTION = {
    "host": "localhost",
    "port": "19530",
    "user": "",
    "password": "",
    "secure": False,
    'uri': "https://example.zillizcloud.com",
    'token': "token"
}

def Create_collection_from_docs(splits, embeddings,collection_name=COLLECTION_NAME, connection_args= CONNECTION_ARGS):
    
    print("----------Embedding started to Milvus----------")
    vector_store = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=collection_name,
    drop_old=True,
    auto_id=True,
    
    ).from_documents(
    splits,
    embedding=embeddings,
    collection_name=collection_name,
    connection_args=connection_args,
    )

    print("----------Embedding finished to Milvus----------")
    
def load_base_template(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            base_template = file.read()  # Read all data from file
        return base_template
    except FileNotFoundError:
        return ""  # Return an empty string if file not found


def langchain_template():
    template = """ 
You follow these instructions
0. Keep the answer as concise as possible and If you don't know the answer, just say that you don't know, don't try to make up an answer.
1. Response to question of HUMAN in the Current Conversation. Always leave a file_path and description with respone to question. If none, write None.
If you don't have enough information to answer, do step 2
2. Refer to the history conversation as information to answer.
If you don't have enough information to answer, do step 3
3. Refer to Library_base_knowledge. 


Library_base_knowledge
{Library_base_knowledge}

history_conversation
{history_conversation}
    
Current conversation:
Human: {input}
"""
    rag_prompt = ChatPromptTemplate.from_template(template)
    
    return rag_prompt
    



def split_mutiple_documents(current_path,chunk_size):
    documents = []

    for file in os.listdir():
        # if file.endswith('.pdf'):
        #     pdf_path = './docs/' + file
        #     loader = PyPDFLoader(pdf_path)
        #     documents.extend(loader.load())
        # elif file.endswith('.docx') or file.endswith('.doc'):
        #     doc_path = './docs/' + file
        #     loader = Docx2txtLoader(doc_path)
        #     documents.extend(loader.load())
        if file.endswith('.txt'):
            text_path = current_path + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())
        if file.endswith('.md'):
            text_path = current_path + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    all_splits = text_splitter.split_documents(documents)
    return all_splits


def vector_store_milvus(embeddings , connection_args=CONNECTION_ARGS, collection_name=COLLECTION_NAME ):
    
    vector_store = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=collection_name,
    drop_old=False,
    auto_id=True
    )
    
    return vector_store





def invoke_from_retriever(query, llm, prompt_template, uuid=''):
    
    
    expr = f"source == '{uuid}'"
    retrieverOptions = {"expr": expr , 'k' : 1}
    pks = vectorstore.get_pks(expr)
    retriever = vectorstore.as_retriever(search_kwargs=retrieverOptions)
    
    if pks:        
        history = retriever.get_relevant_documents(query)[0].page_content + "\n"
    else:
        history = ""
        
    
    base = load_base_template('./base_template.md')

    
    # Set up the components of the chain.
    setup_and_retrieval = RunnableParallel(
        Library_base_knowledge =  RunnableLambda(lambda _: base),
        history_conversation=RunnableLambda(lambda _: history),  # Use RunnableLambda for static content
        input=RunnablePassthrough()  # This can just pass the question as is
    )

    # Construct and invoke the chain
    rag_chain = setup_and_retrieval | prompt_template | llm
    
    
    return history, query, rag_chain.invoke(query).content

docs_splits = split_mutiple_documents('./', CHUNK_SIZE)


prompt_template = langchain_template()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0) 


# Create_collection_from_docs(docs_splits, embeddings)





vectorstore = vector_store_milvus(embeddings)

# expr = f"source == './base_template.md'"
# pks = vectorstore.get_pks(expr)
# retrieverOptions = {"expr": f"pk == {pks[0]}" , 'k' : 2}
        
# # Retrieve existing documents
# retriever = vectorstore.as_retriever(search_kwargs=retrieverOptions)


milvus_instance = MilvusMemory(uri=MILVUS_URI, token=MILVUS_TOKEN, collection_name=COLLECTION_NAME)


    

query = "what is for reusability especially by Python? "
def Milvus_chain(query , llm, template , session ='',embedding=''):
    
    if embedding == '':
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        
    history, question, answer = invoke_from_retriever(query, llm, template,session)
    session = milvus_instance.memory_insert(history + "HUMAN:"+question+"\nAI:" + answer, embedding)
    
    return session
f1 = Milvus_chain(query,llm,prompt_template)

query = "what is topic jusst before?? "
f2 = Milvus_chain(query,llm,prompt_template,f1)

query = "what is article for pip related items?"
f3 = Milvus_chain(query,llm,prompt_template,f2)












