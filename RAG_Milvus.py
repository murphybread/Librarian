# Langchain for Dev
import os
import re


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


# Class Milvus
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
            text_for_embed = query
        else:
            text_for_embed = query.page_content
            
        print(f'in memory_insert query : {text_for_embed}')
            
        if len(session) < 1:
            session = str(uuid.uuid1())
    
        session = Path(session).as_posix()
        expr = f"source == '{session}'"
        expr = expr.encode('utf-8', 'ignore').decode('utf-8')
        pks = self.vectorstore.get_pks(expr)
        print(f'Check exist memory session pks:{pks} , session:{session}, session type: {type(session)}')
        if pks:
            self.collection.delete(collection_name=COLLECTION_NAME, ids=pks)


        vector = self.embeddings.embed_query(text_for_embed)                
        data = {"source": session, "text": text_for_embed ,"vector": vector}
        self.collection.insert(collection_name= COLLECTION_NAME, data=data)
        print(f'session: {session} is inserted')
        
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
        book_count = 0
        # Transform splits_path to Path object 
        splits_path = Path(splits_path)
        
        pattern = re.compile('.*[a-zA-Z]+.*\.md$')
        
        # Use Path objects to navigate directories and files
        for file_path in splits_path.rglob('*.md'):
            
            if pattern.match(file_path.name):
                # Transform POSIX style string to file path 
                session = file_path.as_posix()
                
                print(f'after as_posix session: {session}')
                # Using TextLoader and RecursiveCharacterTextSplitter to Process Files
                loader = TextLoader(session,encoding='utf-8')
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
                splits = text_splitter.split_documents(documents)

                # Update the collection for each split
                for split in splits:
                    self.memory_insert(query=split, session=session)
                    book_count +=1
        print(f"splits_path : {splits_path} book_count : {book_count}")

    def Milvus_chain(self, query, llm, prompt_template, session='',file_path_session=''):
        
        print(f"befre invoke from milvus_chain session : {session} ")
        print(f"befre invoke from milvus_chain file_path_session : {file_path_session} ")
        if len(file_path_session) > 2:
            file_history, file_question, file_answer = invoke_from_retriever(query, llm, prompt_template, self.vectorstore, file_path_session)
            file_info = f"\nInformation: {file_path_session}\n{file_history}"
            
            # print(f'file_history : {file_history}')
            # print(f'file_question : {file_question}')
            # print(f'file_answer : {file_answer}')
            
                                 
            print(f"before file_path query : {query}")
            query = query + file_info
            print(f"after file_path query : {query}")
        
        history, question, answer = invoke_from_retriever(query, llm, prompt_template, self.vectorstore, session)
        print(f"after invoke milvus_chain question : {question} ")
        print(f"after invoke milvus_chain answer : {answer} ")
        session = self.memory_insert(history + "\nHUMAN:" + question + "\nAI:" + answer, session=session)
        return history, question, answer, session


def invoke_from_retriever(query, llm, prompt_template, vectorstore , uuid=''):    
    expr = f"source == '{uuid}'"
    retrieverOptions = {"expr": expr , 'k' : 1}
    pks = vectorstore.get_pks(expr)
    
    
    print(f'pks: {pks}')
    
    if pks:        
        retriever = vectorstore.as_retriever(search_kwargs=retrieverOptions)
        history = retriever.get_relevant_documents(query)[0].page_content + "\n"
    else:
        history = ""
    
    
    # print(f'type history : {type(history)}')
    # print(f'history : {history}')

    
    knowledge = get_content_from_path(BASE_FILE_PATH)
    # print("knowledge\n" + knowledge)
    
    # Set up the components of the chain.
    setup_and_retrieval = RunnableParallel(
        Library_base_knowledge =  RunnableLambda(lambda _ : knowledge),
        history_conversation=RunnableLambda(lambda _: history),  # Use RunnableLambda for static content
        input=RunnablePassthrough()  # This can just pass the question as is
    )

    # Construct and invoke the chain
    rag_chain = setup_and_retrieval | prompt_template | llm
    answer = rag_chain.invoke(query).content.rstrip("\nNone")
    
    return history, query, answer


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


def extract_path(query, keyword='file_path: '):    
    # 'file_path: ' Find path by ketword        
    if keyword not in query:
        return ''
    start_index = query.find(keyword)
    prefix = '../../'
     
    
    if start_index != -1:
        # 'file_path: ' 
        start_index += len(keyword)
        temp_extract = query[start_index:]

        # Find postion '.md' from temp_extract string
        end_index = temp_extract.find('.md')

        if end_index != -1:
            # Extract final path that include '.md'
            extracted_path = temp_extract[:end_index + len('.md')]
            return prefix + extracted_path
        else:
            print("'.md' not found.")
    

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

def get_content_from_path(file_path):
    # Step 1: Create a Path object for the file
    content_path = Path(file_path)

# Step 2: Check if the file exists
    if content_path.exists():
        # Step 3: Open the file and read its contents into a string
        with open(content_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    else:
        print(f"The file {content_path} does not exist.")
        return ""

# docs_splits = split_multiple_documents('./', CHUNK_SIZE)
# prompt_template = hub.pull("murphy/librarian_guide")
# embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
# llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0) 
# vectorstore = vectorstore_milvus(embeddings)

# history , query , answer = invoke_from_retriever("HUMAN:Hello we talk about gitlab AI:Based on the information available in Murphy's library, here is a relevant file: - **File Path**: 200/210/210.20/210.20 a.md - **Description**: The solution about Gitlab. GitLab is one devsecops solution a" +"What is about detailed?", llm, prompt_template, vectorstore , uuid='../../200/210/210.20/210.20 a.md')
# print(f'history: {history}')
# print(f'query : {query}')
# print(f'answer :{answer}')


# def extract_pattern(string):
#     prefix = "../../"
#     if string.startswith(prefix):
#         return string[len(prefix):]
#     else:
#         return string


# create_collection(splits_path='../../200')

