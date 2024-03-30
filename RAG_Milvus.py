# Langchain for Dev
import os

## Data
from langchain_community.document_loaders import TextLoader,PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document


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
from pymilvus import Collection,connections,MilvusClient
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
        #connections.connect("default", uri=uri, token=token)
        self.collection = MilvusClient(uri = uri, token= token)
        self.vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args=connection_args,
            collection_name=collection_name,
            drop_old=False,
            auto_id=True
        )
        self.embeddings = embeddings

    def memory_insert(self, query, session=""):

        if isinstance(query, str):
            text_to_embed = query
        else:
            text_to_embed = query.page_content


        vector = self.embeddings.embed_query(text_to_embed)
        expr = f"source == '{session}'"
        pks = self.vectorstore.get_pks(expr)

        if not session:
            session = str(uuid.uuid1())
        else:
            expr = f"source == '{session}'"
            pks = self.vectorstore.get_pks(expr)
            if pks:
                self.collection.delete(collection_name=COLLECTION_NAME, ids=pks)

                
        
        data = {"source": session, "text": text_to_embed ,"vector": vector}
        
        self.collection.insert(collection_name= COLLECTION_NAME, data=data)
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
    
    def create_or_update_collection(self, splits_path='./', chunk_size=CHUNK_SIZE):
        # Walk through all directories and files starting at splits_path
        for root, _, files in os.walk(splits_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Assuming TextLoader and RecursiveCharacterTextSplitter have methods to process single files
                # Adjust loader as per your file processing requirement
                loader = TextLoader(file_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
                splits = text_splitter.split_documents(documents)

                # For each split, update the collection
                for split in splits:
                    # Generate a unique session or use file path as identifier
                    session = file_path  # or any unique identifier as per your logic
                    # Update collection for each document split
                    self.memory_insert(query=split, session=session)

        print(f"Collection updated with documents from {splits_path}.")

    def Milvus_chain(self, query, llm, prompt_template, session=''):
        # Example: Assuming llm and prompt_template are used in a method like invoke_from_retriever
        history, question, answer = invoke_from_retriever(query, llm, prompt_template, self.vectorstore, session)    
        session = self.memory_insert(history + "\nHUMAN:" + question + "\nAI:" + answer, session=session)
        return history, question, answer, session



def load_base_template(file_path):
    try:
        return Path(file_path).read_text(encoding='utf-8')
    except FileNotFoundError:
        return ""


def split_multiple_documents(current_path, chunk_size: int):
    documents = []

    # Walk through all directories and files starting at current_path
    for root, dirs, files in os.walk(current_path):
        for file in files:
            if file.endswith('.txt') or file.endswith('.md'):
                text_path = os.path.join(root, file)  # Correctly join the path to the file
                loader = TextLoader(text_path)
                documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    all_splits = text_splitter.split_documents(documents)
    return all_splits




def create_collection(collection_name=COLLECTION_NAME, connection_args= CONNECTION_ARGS, embeddings= '',splits_path ='./'):
    
    splits = split_multiple_documents(splits_path, CHUNK_SIZE) if splits_path else './'

    if embeddings == '':
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    print(f"----------Embedding Starting from {splits_path} to Milvus----------")
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
    print(f"----------Embedding has Finished {splits_path} files into Milvus----------")
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
    retrieverOptions = {"expr": expr , 'k' : 2}
    pks = vectorstore.get_pks(expr)
    retriever = vectorstore.as_retriever(search_kwargs=retrieverOptions)
    
    print(f'pks: {pks}')
    print(f'retriever : {retriever}')
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



# def Milvus_chain(query, llm, prompt_template, session='', embeddings=''):
#     if embeddings == '':    
#         embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    
#     milvus_memory = MilvusMemory(embeddings,uri=MILVUS_URI, token=MILVUS_TOKEN, collection_name=COLLECTION_NAME)
#     # print(milvus_memory, milvus_memory.vectorstore)
#     history, question, answer = invoke_from_retriever(query, llm, prompt_template, milvus_memory.vectorstore, session)    
#     session = milvus_memory.memory_insert(history + "\nHUMAN:" + question + "\nAI:" + answer, embeddings, session)

    
#     return history, question, answer, session

# Implement


docs_splits = split_multiple_documents('./', CHUNK_SIZE)
prompt_template = hub.pull("murphy/librarian_guide")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0) 
vectorstore = vectorstore_milvus(embeddings)

history , query , answer = invoke_from_retriever("HUMAN:Hello we talk about gitlab AI:Based on the information available in Murphy's library, here is a relevant file: - **File Path**: 200/210/210.20/210.20 a.md - **Description**: The solution about Gitlab. GitLab is one devsecops solution a" +"What is about detailed?", llm, prompt_template, vectorstore , uuid='../../200/210/210.20/210.20 a.md')
print(f'history: {history}')
print(f'query : {query}')
print(f'answer :{answer}')


def extract_pattern(string):
    prefix = "../../"
    if string.startswith(prefix):
        return string[len(prefix):]
    else:
        return string


# create_collection(splits_path='../../200')


