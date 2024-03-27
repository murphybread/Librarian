# Langchain for Dev
import os

## Data
from langchain_community.document_loaders import TextLoader,PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

## DB
from langchain_community.vectorstores import Milvus

## LLM
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

## Memory
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, VectorStoreRetrieverMemory

## Chain
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.chains import ConversationChain



# Langchain for Ops
from langchain import hub # Prompt managing from the langchainhub site
from dotenv import load_dotenv # Lading environment variables from a file
from pathlib import Path # Intuitive path management regardless of OS

# Milvus
from pymilvus import Collection,connections
import uuid


# Variables
load_dotenv()


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
BASE_FILE_PATH= os.getenv("BASE_FILE_PATH")

MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_URI = os.getenv("MILVUS_URI")
CONNECTION_ARGS = {'uri': MILVUS_URI, 'token': MILVUS_TOKEN}
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


# Class 
class MilvusMemory:
    def __init__(self, embeddings,uri, token, collection_name,connection_args=CONNECTION_ARGS):
        connections.connect("default", uri=uri, token=token)
        self.collection = Collection(name=collection_name)
        self.vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args=connection_args,
            collection_name=collection_name,
            drop_old=False,
            auto_id=True
        )

    def memory_insert(self, query, embedding,session=""):
        
        vector = embedding.embed_query(query)
        if not session:
            session = str(uuid.uuid1())
        self.collection.insert([[session], [query], [vector]])
        return session
    
    def update_entity(self, file_path, vectorstore):
        print("-----------upsert start-----------")
        expr = f"source == '{file_path}'"
        pks = vectorstore.get_pks(expr)

        # Load new documents to be inserted
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        if pks:
            # Prepare retrieval options
            retrieverOptions = {"expr": f"pk == {pks[0]}"}
            # Retrieve existing documents
            retriever = vectorstore.as_retriever(search_kwargs=retrieverOptions)
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
            vectorstore.delete(pks)

            print(f'docs : {docs}, docs_type: {type(docs)}')
            # Add the new documents to the vector store after deletion
            vectorstore.add_documents(docs)

            # Fetch the primary keys for new documents based on the same expression
            new_pks = vectorstore.get_pks(expr)

            # Print the information about deletion and creation
            print(f"Entity with pk={pks} deleted and new entity created with pk={new_pks}.")
        else:
            print(f"No entity found for {file_path}. Creating entity...")
            vectorstore.add_documents(docs)
            print("New entity created.")

        print("-----------Upsert finished-----------")
        return None



    


def load_base_template(file_path):
    try:
        return Path(file_path).read_text(encoding='utf-8')
    except FileNotFoundError:
        return ""


def split_mutiple_documents(current_path,chunk_size: int):
    documents = []

    for file in os.listdir():
        if file.endswith('.pdf'):
            pdf_path = './docs/' + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = './docs/' + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
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



def create_collection(collection_name=COLLECTION_NAME, connection_args= CONNECTION_ARGS, embeddings= '',splits =''):

    if splits == '':
        splits = split_mutiple_documents('./', CHUNK_SIZE)

    if embeddings == '':
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    print(f"----------Embedding Starting from ALL files to Milvus----------")
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
    print(f"----------Embedding has Finished ALL files intto Milvus----------")
    return None


def vectorstore_milvus(embeddings , connection_args=CONNECTION_ARGS, collection_name=COLLECTION_NAME ):
    
    vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=collection_name,
    drop_old=False,
    auto_id=True
    )
    
    return vectorstore


def invoke_from_retriever(query, llm, prompt_template, vectorstore , uuid=''):    
    expr = f"source == '{uuid}'"
    retrieverOptions = {"expr": expr , 'k' : 1}
    pks = vectorstore.get_pks(expr)
    retriever = vectorstore.as_retriever(search_kwargs=retrieverOptions)
    
    if pks:        
        history = retriever.get_relevant_documents(query)[0].page_content + "\n"
    else:
        history = ""
            
    # Set up the components of the chain.
    setup_and_retrieval = RunnableParallel(
        Library_base_knowledge =  RunnableLambda(lambda _: load_base_template(BASE_FILE_PATH)),
        history_conversation=RunnableLambda(lambda _: history),  # Use RunnableLambda for static content
        input=RunnablePassthrough()  # This can just pass the question as is
    )

    # Construct and invoke the chain
    rag_chain = setup_and_retrieval | prompt_template | llm
    answer = rag_chain.invoke(query).content.rstrip("\nNone")
    
    return history, query, answer


def Milvus_chain(query, llm, prompt_template, session='', embeddings=''):
    if embeddings == '':
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    
    milvus_memory = MilvusMemory(embeddings,uri=MILVUS_URI, token=MILVUS_TOKEN, collection_name=COLLECTION_NAME)
    # print(milvus_memory, milvus_memory.vectorstore)
    history, question, answer = invoke_from_retriever(query, llm, prompt_template, milvus_memory.vectorstore, session)    
    session = milvus_memory.memory_insert(history + "HUMAN:" + question + "\nAI:" + answer, embeddings)
    
    
    
    return history, question, answer,session

# Implement


docs_splits = split_mutiple_documents('./', CHUNK_SIZE)
prompt_template = hub.pull("murphy/librarian_guide")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0) 
vectorstore = vectorstore_milvus(embeddings)







# history1, question1, answer1,session1 = Milvus_chain("What is number about EDA?",llm,prompt_template)
# print(history1, question1, answer1,session1)
# # f1 = Milvus_chain("What is description about EDA?",llm,prompt_template)

