import os
import re
from pathlib import Path
from langchain_community.document_loaders import TextLoader
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

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

# Ensure connection to Milvus is established
connections.connect("default", uri=MILVUS_URI, token=MILVUS_TOKEN)

def create_or_update_collection(splits_path='./', chunk_size=CHUNK_SIZE):
    splits_path = Path(splits_path)
    pattern = re.compile(r'.*[a-zA-Z]+.*\.md$')

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args=CONNECTION_ARGS,
        collection_name=COLLECTION_NAME,
        drop_old=False,
        auto_id=True
    )

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
                    delete_result = collection.delete(expr=expr_delete)
                    print(f"Deleted outdated documents from {source}. Delete result: {delete_result}")
                else:
                    print(f"No valid primary keys found for {source}, skipping deletion.")
            except Exception as e:
                print(f"Failed to delete existing documents for {source}: {e}")

            new_docs = []
            for doc in splits:
                text_content = doc.page_content
                vector = embeddings.embed_query(text_content)
                data = {"source": source, "text": text_content, "vector": vector}
                new_docs.append(data)

            try:
                collection.insert(new_docs)
                print(f"Inserted documents from {source}.")
            except Exception as e:
                print(f"Failed to insert documents for {source}: {e}")

def memory_insert(vectorstore, embeddings, collection_name, query, session=""):
    collection = Collection(collection_name)

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
    pks = vectorstore.get_pks(expr)
    print(f'Check exist memory session pks:{pks} , session:{session}, session type: {type(session)}')
    if pks:
        collection.delete(expr=f"source == '{session}'")

    vector = embeddings.embed_query(text_for_embed)
    data = {"source": session, "text": text_for_embed, "vector": vector}
    collection.insert([data])  # Ensure data is inserted as a list of dictionaries
    print(f'session: {session} is inserted')
    
    return session

def Milvus_chain(query, llm, prompt_template, vectorstore, embeddings, collection_name, session='', file_path_session=''):
    print(f"before invoke from milvus_chain session : {session} ")
    print(f"before invoke from milvus_chain file_path_session : {file_path_session} ")
    
    if len(file_path_session) > 2:
        file_history, file_question, file_answer = invoke_from_retriever(query, llm, prompt_template, vectorstore, file_path_session)
        file_info = f"\nInformation: {file_path_session}\n{file_history}"
        
        print(f"before file_path query : {query}")
        query = query + file_info
        print(f"after file_path query : {query}")
    
    history, question, answer = invoke_from_retriever(query, llm, prompt_template, vectorstore, session)
    print(f"after invoke milvus_chain question : {question} ")
    print(f"after invoke milvus_chain answer : {answer} ")
    
    # Use the memory_insert function to update the session
    session = memory_insert(vectorstore, embeddings, collection_name, history + "\nHUMAN:" + question + "\nAI:" + answer, session)
    
    return history, question, answer, session

def invoke_from_retriever(query, llm, prompt_template, vectorstore, uuid=''):
    expr = f"source == '{uuid}'"
    retrieverOptions = {"expr": expr, 'k': 1}
    pks = vectorstore.get_pks(expr)
    
    print(f'pks: {pks}')
    
    if pks and pks[0] is not None:
        retriever = vectorstore.as_retriever(search_kwargs=retrieverOptions)
        relevant_docs = retriever.get_relevant_documents(query)
        if relevant_docs and relevant_docs[0].metadata.get('text'):
            history = relevant_docs[0].metadata['text'] + "\n"
        else:
            history = ""
    else:
        history = ""
        vector = embeddings.embed_query(query)
        new_docs = [{"source": uuid, "text": query, "vector": vector}]
        try:
            collection = Collection(COLLECTION_NAME)
            collection.insert(new_docs)
            print(f"Inserted new entity for session {uuid}.")
        except Exception as e:
            print(f"Failed to insert new entity for session {uuid}: {e}")

    knowledge = get_content_from_path(BASE_FILE_PATH)
    print(f'knowledge: {knowledge[:50]}')  # Correct string interpolation
    print(f'BASE_FILE_PATH : {BASE_FILE_PATH}')           # Correct direct string output
    
    setup_and_retrieval = RunnableParallel(
        Library_base_knowledge=RunnableLambda(lambda _: knowledge),
        history_conversation=RunnableLambda(lambda _: history),
        input=RunnablePassthrough()
    )

    rag_chain = setup_and_retrieval | prompt_template | llm
    print(f"setup_and_retrieval being sent to RAG chain: {setup_and_retrieval}")
    print(f"query being sent to RAG chain: {query}")
    answer = rag_chain.invoke(query).content.rstrip("\nNone")
    print(f"Answer generated by RAG chain: {answer}")

    
    return history, uuid, answer

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
    collection = Collection(COLLECTION_NAME)
    expr_check = f"Auto_id == {auto_id}"
    results = collection.query(expr_check)
    if results:
        expr_delete = f"Auto_id in [{auto_id}]"
        try:
            delete_result = collection.delete(expr=expr_delete)
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
