import os
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.schema.runnable import RunnablePassthrough


from langchain.docstore.document import Document

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
        text_path = './' + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())

# loader = TextLoader("./example.txt")
# docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_splits = text_splitter.split_documents(documents)



embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
TOKEN = os.environ['MILVUS']
COLLECTION_NAME = "cc"
connection_args = { 'uri': "https://in03-881134e550fc1b4.api.gcp-us-west1.zillizcloud.com", 'token': TOKEN }

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
rag_prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) 



vector_store_E = Milvus(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME,
    drop_old=False,
    auto_id=True
    
)



query = "What do yo favorite number is?"
docs = vector_store_E.similarity_search(query)


retriever_E = vector_store_E.as_retriever()

rag_chain2 = (
    {"context": retriever_E, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
)


print(rag_chain2.invoke(query))



def update_entity(file_path, vector_store):
    print("-----------upsert start-----------")
    expr = f"source == '{file_path}'"
    pks = vector_store.get_pks(expr)
    
    # Load new documents to be inserted
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)  # Define 'docs' outside the if block

    if pks:
        # Prepare retrieval options
        retrieverOptions = {"expr": f"pk == {pks[0]}"}
        
        # Retrieve existing documents
        retriever = vector_store.as_retriever(search_kwargs=retrieverOptions)
        existing_docs = retriever.get_relevant_documents(expr)
        print(existing_docs)
        
        # Check if documents exist and print information
        if existing_docs:
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
        # This is where 'docs' is now available to use because it's been defined outside the 'if' condition
        vector_store.add_documents(docs)
        print("New entity created.")

    print("-----------Upsert finished-----------")

# Call the function
update_entity('./test.txt', vector_store_E)


DEFAULT_MILVUS_CONNECTION = {
    "host": "localhost",
    "port": "19530",
    "user": "",
    "password": "",
    "secure": False,
    'uri': "https://example.zillizcloud.com",
    'token': "token"
}

def Create_collection_from_docs(embedding,splits,collection_name="default_collection_name", connection_args=DEFAULT_MILVUS_CONNECTION):
    
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

# Create_collection_from_docs(embeddings,all_splits,COLLECTION_NAME,connection_args )


# retriever = vector_store.as_retriever()

# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | rag_prompt
#     | llm
# )


# print(rag_chain.invoke(query))

