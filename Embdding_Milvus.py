import os
from langchain_community.document_loaders import TextLoader
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
COLLECTION_NAME = "aa"
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

print("-----------upsert start-----------")

def update_entity(file_path, vector_store):
    expr = "source == '" + file_path + "'"
    pks = vector_store.get_pks(expr)
    
    if pks:
        pk = pks  # Assuming there's only one matching entity
        
        
        # Get the existing document
        
        retrieverOptions = {
            "expr": f'pk == {pk[0]}'
        }
        
        retriever = vector_store.as_retriever(search_kwargs=retrieverOptions)
        print(f'retriever: {retriever}')
        
        existing_docs = retriever.get_relevant_documents(expr)
        print(existing_docs)
        print("++++++++++++++++++++++++++++++\n\n\n")
        if existing_docs:
            existing_doc = existing_docs[0]
            print(f"upsert before: {existing_doc.page_content}")
        else:
            print("No existing text content found.")
        
        # loader = TextLoader(file_path)
        # documents = loader.load()
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # docs = text_splitter.split_documents(documents)
        # new_text = docs[0].page_content  # Assuming a single document
        # new_vector = embeddings.embed_query(new_text)

        # # Prepare the data for update
        # data = [[pk], [], [new_text], [new_vector]]
        # data = Document(page_content=new_text,
        #     metatdata={
        #         "source": file_path,
        #         "pk": pk,
        #         "vector": new_vector
        #     }
        # )


        # # Update the entity
        # vector_store.upsert(pk,data)

        # retriever = vector_store.as_retriever(search_kwargs={'filter': {'source': file_path}})
        # existing_docs = retriever.get_relevant_documents()
        # print(existing_docs)
        # print("++++++++++++++++++++++++++++++\n\n\n")
        # if existing_docs:
        #     existing_doc = existing_docs[0]
        #     print(f"upsert after: {existing_doc.page_content}")
        # else:
        #     print("No existing text content found.")

        print(f"Entity with pk={pk} updated successfully.")
    else:
        print(f"No entity found for {file_path}")

update_entity('./number.txt',vector_store_E)



# print("----------md file embedded start to Milvus----------")
# vector_store = Milvus(
#     embedding_function=embeddings,
#     connection_args=connection_args,
#     collection_name=COLLECTION_NAME,
#     drop_old=True,
# ).from_documents(
#     all_splits,
#     embedding=embeddings,
#     collection_name=COLLECTION_NAME,
#     connection_args=connection_args,
# )
# print("----------md file embedded end to Milvus----------")



# retriever = vector_store.as_retriever()




# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | rag_prompt
#     | llm
# )


# print(rag_chain.invoke(query))
# pks = vector_store.get_pks(expr)
# print(pks)


