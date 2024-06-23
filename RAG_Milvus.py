import os
import re
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from pymilvus import Collection, connections, MilvusClient
import uuid

# Variables
load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
BASE_FILE_PATH = os.getenv("BASE_FILE_PATH")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_URI = os.getenv("MILVUS_URI")
CONNECTION_ARGS = {'uri': MILVUS_URI, 'token': MILVUS_TOKEN}
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


def create_or_update_collection(splits_path='./', chunk_size=CHUNK_SIZE):
    splits_path = Path(splits_path)
    pattern = re.compile(r'.*[a-zA-Z]+.*\.md$')
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args=CONNECTION_ARGS,
        collection_name=COLLECTION_NAME,
        drop_old=False,
        auto_id=True
    )

    connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    collection = Collection(COLLECTION_NAME)
    collection.load()  # Ensure collection is loaded into memory

    for file_path in splits_path.rglob('*.md'):
        if pattern.match(file_path.name):
            loader = TextLoader(file_path.as_posix(), encoding='utf-8')
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
            splits = text_splitter.split_documents(documents)
            source = file_path.as_posix()
            print(f'Source: {source}')
            expr = f'source == "{source}"'
            print(f'Query expression: {expr}')

            try:
                res = collection.query(expr=expr, output_fields=["Auto_id"])
                pks = [item["Auto_id"] for item in res]
                print(f'pks: {pks}, pkstype: {type(pks)}')

                if pks:
                    expr_delete = f"Auto_id in {pks}"
                    print(f"Deleting documents with expr: {expr_delete}")
                    delete_result = collection.delete(expr_delete)
                    print(f"Deleted outdated documents from {source}. Delete result: {delete_result}")
                else:
                    print(f"No valid primary keys found for {source}, skipping deletion.")
            except Exception as e:
                print(f"Failed to delete existing documents for {source}: {e}")

            new_docs = [Document(page_content=doc.page_content, metadata={"source": source}) for doc in splits]

            try:
                vectorstore.add_documents(new_docs)
                print(f"Inserted documents from {source}.")
            except Exception as e:
                print(f"Failed to insert documents for {source}: {e}")



def invoke_from_retriever(query, llm, prompt_template, vectorstore, uuid=''):    
    expr = f"source == '{uuid}'"
    retrieverOptions = {"expr": expr , 'k' : 1}
    pks = vectorstore.get_pks(expr)
    
    print(f'pks: {pks}')
    
    if pks:        
        retriever = vectorstore.as_retriever(search_kwargs=retrieverOptions)
        history = retriever.get_relevant_documents(query)[0].page_content + "\n"
    else:
        history = ""
    
    knowledge = get_content_from_path(BASE_FILE_PATH)
    
    setup_and_retrieval = RunnableParallel(
        Library_base_knowledge=RunnableLambda(lambda _: knowledge),
        history_conversation=RunnableLambda(lambda _: history),
        input=RunnablePassthrough()
    )

    rag_chain = setup_and_retrieval | prompt_template | llm
    answer = rag_chain.invoke(query).content.rstrip("\nNone")
    
    return history, query, answer

def get_content_from_path(file_path):
    content_path = Path(file_path)

    if content_path.exists():
        with open(content_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    else:
        print(f"The file {content_path} does not exist.")
        return ""

def delete_entity(auto_id):
    # Connect to Milvus
    connections.connect(
        uri=MILVUS_URI,
        token=MILVUS_TOKEN,
    )
    # Retrieve the collection
    collection = Collection(COLLECTION_NAME)
    # Check if the entity exists
    expr_check = f"Auto_id == {auto_id}"
    results = collection.query(expr_check)
    if results:
        # Delete the entity if it exists
        expr_delete = f"Auto_id in [{auto_id}]"
        try:
            delete_result = collection.delete(expr_delete)
            print(f"Delete result: {delete_result}")
        except Exception as e:
            print(f"Failed to delete entity with Auto_id {auto_id}: {e}")
    else:
        print(f"No entity found with Auto_id: {auto_id}")
    return results

def extract_path(query):
    keywords = 'file_path: '
    if keywords not in query:
        return ''
    start_index = query.find(keywords)
    prefix = '../../'
     
    if start_index != -1:
        start_index += len(keywords)
        temp_extract = query[start_index:]
        end_index = temp_extract.find('.md')
        if end_index != -1:
            extracted_path = temp_extract[:end_index + len('.md')]
            print(f'extracted_path {extracted_path}')
            return prefix + extracted_path
        else:
            print("'.md' not found.")
    return ''

def create_collection(collection_name=COLLECTION_NAME, connection_args=CONNECTION_ARGS, embeddings='', splits_path='./'):
    splits = split_multiple_documents(splits_path, CHUNK_SIZE) if splits_path else './'

    if embeddings == '':
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    print(f"----------Embedding Starting from {splits_path} to Milvus----------")
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name=collection_name,
        drop_old=True,
        auto_id=True
    ).from_documents(
        splits,
        embedding=embeddings,
        collection_name=collection_name,
        connection_args=connection_args,
    )
    print(f"----------Embedding has Finished {splits_path} files into Milvus----------")
    return None

def split_multiple_documents(current_path, chunk_size: int):
    documents = []

    for root, dirs, files in os.walk(current_path):
        for file in files:
            if file.endswith('.txt') or file.endswith('.md'):
                text_path = os.path.join(root, file)
                loader = TextLoader(text_path)
                documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    all_splits = text_splitter.split_documents(documents)
    return all_splits

def vectorstore_milvus(embeddings, connection_args=CONNECTION_ARGS, collection_name=COLLECTION_NAME):
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args=connection_args,
        collection_name=collection_name,
        drop_old=False,
        auto_id=True
    )
    return vectorstore
