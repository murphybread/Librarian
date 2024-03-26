import os
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda


from langchain.docstore.document import Document




from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, VectorStoreRetrieverMemory
from langchain.chains import ConversationChain





from langchain import hub

from dotenv import load_dotenv
# Langchain

#Milvus
load_dotenv()
CHUNK_SIZE= int(os.getenv("CHUNK_SIZE"))
MILVUS_TOKEN=os.getenv("MILVUS_TOKEN")
MILVUS_URI=os.getenv("MILVUS_URI")
CONNECTION_ARGS = { 'uri': MILVUS_URI, 'token': MILVUS_TOKEN }


COLLECTION_NAME = "Library"


DEFAULT_MILVUS_CONNECTION = {
    "host": "localhost",
    "port": "19530",
    "user": "",
    "password": "",
    "secure": False,
    'uri': "https://example.zillizcloud.com",
    'token': "token"
}

def create_collection(splits, embeddings,collection_name=COLLECTION_NAME, connection_args= CONNECTION_ARGS):
    print(f"----------Embedding started from {splits} to Milvus----------")
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
    print(f"----------Embedding finished from {splits} to Milvus----------")
    return None
    




from pymilvus import Collection,connections
import uuid


class MilvusMemory:
    def __init__(self, uri, token, collection_name):
        connections.connect("default", uri=uri, token=token)
        self.collection = Collection(name=collection_name)

    def memory_insert(self, query, embedding,session=""):
        
        vector = embedding.embed_query(query)
        if not session:
            session = str(uuid.uuid1())
        self.collection.insert([[session], [query], [vector]])
        
        
        return session
    
    def update_entity(self, file_path, vector_store):
        print("-----------upsert start-----------")
        expr = f"source == '{file_path}'"
        pks = vector_store.get_pks(expr)

        # Load new documents to be inserted
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        if pks:
            # Prepare retrieval options
            retrieverOptions = {"expr": f"pk == {pks[0]}"}
            # Retrieve existing documents
            retriever = vector_store.as_retriever(search_kwargs=retrieverOptions)
            existing_docs = retriever.get_relevant_documents(expr)
            print(existing_docs)

            # Check if documents exist and print information
            if existing_docs:
                print(f'existing_docs : {existing_docs}')
                existing_doc = existing_docs[0]
                print(f"upsert before: {existing_doc.page_content}")
            else:
                print("No existing text content found.")

            # Delete the outdated entity
            vector_store.delete(pks)

            print(f'docs : {docs}, docs_type: {type(docs)}')
            # Add the new documents to the vector store after deletion
            vector_store.add_documents(docs)

            # Fetch the primary keys for new documents based on the same expression
            new_pks = vector_store.get_pks(expr)

            # Print the information about deletion and creation
            print(f"Entity with pk={pks} deleted and new entity created with pk={new_pks}.")
        else:
            print(f"No entity found for {file_path}. Creating entity...")
            vector_store.add_documents(docs)
            print("New entity created.")

        print("-----------Upsert finished-----------")



# From get strings file
def load_base_template(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            base_template = file.read()  # Read all data from file
        return base_template
    except FileNotFoundError:
        return ""  # Return an empty string if file not found

from pathlib import Path

def load_base_template(file_path):
    path = Path(file_path)
    try:
        return path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return ""


def langchain_template():
    template = """ 
You are as a librarian, introducte to the knowledge in Murphy's library.
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

    obj = hub.pull("murphy/librarian_guide")
    # print(type(obj))
    # print(obj)

    rag_prompt = ChatPromptTemplate.from_template(template)
    # print(type(rag_prompt))
    # print(rag_prompt)
    
    return obj
    
langchain_template()


def split_mutiple_documents(current_path,chunk_size: int):
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
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


    


def Milvus_chain(query , llm, template , session ='',embedding=''):
    
    if embedding == '':
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        
    history, question, answer = invoke_from_retriever(query, llm, template,session)
    session = milvus_instance.memory_insert(history + "HUMAN:"+question+"\nAI:" + answer, embedding)
    
    return session
f1 = Milvus_chain("hello what your name?",llm,prompt_template)


f2 = Milvus_chain("what was firstg question?",llm,prompt_template,f1)

# query = "what is article for pip related items?"
# f3 = Milvus_chain(query,llm,prompt_template,f2)











