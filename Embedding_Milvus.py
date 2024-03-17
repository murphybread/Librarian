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


MILVUS_TOKEN = st.secrets['MILVUS']['MILVUS_TOKEN']
MILVUS_URI = st.secrets['MILVUS']['MILVUS_URI']
COLLECTION_NAME = "Library"
connection_args = { 'uri': MILVUS_URI, 'token': MILVUS_TOKEN }

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

def Create_collection_from_docs(splits, embeddings,collection_name="default_collection_name", connection_args=DEFAULT_MILVUS_CONNECTION):
    
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
Answer using history as context
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Answer with description and file_path about file if you can.

history
{history}
    
Current conversation:
Human: {input}
AI assistant:"""
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


def vector_store_milvus(embeddings , connection_args, collection_name="default_collection_name" ):
    
    vector_store = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=collection_name,
    drop_old=False,
    auto_id=True
    )
    
    return vector_store





def invoke_from_retriever(query, vector_store, llm, prompt_template, uuid=' '):
    # Load the initial context
    
    expr = f"source == '{uuid}'"
    # pks = vectorstore.get_pks(expr)
    # retrieverOptions = {"expr": f"pk == {pks[0]}" , 'k' : 3}
    retrieverOptions = {"expr": expr , 'k' : 3}
    
            
    # Retrieve existing documents
    retriever = vectorstore.as_retriever(search_kwargs=retrieverOptions)
    

    
    # if no match files in vectordb, it will return [] about get_relevant_documents(query)
    retriever_result = retriever.get_relevant_documents(query)
    if retriever_result:
        history = retriever.get_relevant_documents(query)[0].page_content
    else:
        history = ""
    
     
    print(f'history : {history}')
    
    
    # Set up the components of the chain.
    setup_and_retrieval = RunnableParallel(
        history=RunnableLambda(lambda _: history),  # Use RunnableLambda for static content
        input=RunnablePassthrough()  # This can just pass the question as is
    )

    # Construct and invoke the chain
    rag_chain = setup_and_retrieval | prompt_template | llm
    return rag_chain.invoke(query).content



docs_splits = split_mutiple_documents('./', CHUNK_SIZE)


prompt_template = langchain_template()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0.03) 


# Create_collection_from_docs(docs_splits, embeddings ,COLLECTION_NAME,connection_args )





vectorstore = vector_store_milvus(embeddings, connection_args, COLLECTION_NAME )

expr = f"source == './base_template.md'"
pks = vectorstore.get_pks(expr)
retrieverOptions = {"expr": f"pk == {pks[0]}" , 'k' : 5}
        
# Retrieve existing documents
retriever = vectorstore.as_retriever(search_kwargs=retrieverOptions)




answer1 = invoke_from_retriever("what is article number about background for machine learning" , vectorstore, llm, prompt_template, './base_template.md')

print(answer1)

# Create an instance of MilvusMemory
milvus_instance = MilvusMemory(uri=MILVUS_URI, token=MILVUS_TOKEN, collection_name=COLLECTION_NAME)
first_session = milvus_instance.memory_insert(answer1, embeddings)
    

# context = vectorstore.as_retriever(search_kwargs=retrieverOptions)
answer2 = invoke_from_retriever("what is the topic we just talked before?" , vectorstore, llm, prompt_template, first_session)
print(answer2)





# result = chat_retriever_chain.invoke({"input": "what your name?", "history": "aa"})
# print(result)

# for r in result:
#     print(r)
#memory = VectorStoreRetrieverMemory(retriever=retriever, ai_prefix="AI Assistant", memory_key="history")


# --------------

# memory=ConversationBufferWindowMemory(k=10,memory_key="history",ai_prefix="AI Assistant")



# conversation_with_summary = ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=True,
#     prompt=prompt_template,
# )                                                                                                                                                                                                                             

 
# print(conversation_with_summary.predict(input ="제 이름이 무엇이었는지 기억하세요?"))




                                                                                                                                                                                        



# print(conversation_with_summary.invoke("What your name?").output_schema.schema())

#print(conversation_with_summary.invoke(""))



# output = conversation_with_summary.predict(input="Hi, my name is Perry, what's up?")

# print(output['response'])

# output = conversation_with_summary.invoke("What about we talekd??")
# print(output['response'])

# output = conversation_with_summary.invoke("What is path about Langchain?")
# print(output['response'])

# output = conversation_with_summary.invoke("What about we talekd??")
# print(output['response'])


#print (conversation_with_summary.predict(input="Hello, write postfix number of our chat whenerver we talked for example your answer ... 1"))








