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




MILVUS_TOKEN = os.environ['MILVUS_TOKEN']
MILVUS_URI = os.environ['MILVUS_URI']
COLLECTION_NAME = "Library"
connection_args = { 'uri': MILVUS_URI, 'token': MILVUS_TOKEN }


DEFAULT_MILVUS_CONNECTION = {
    "host": "localhost",
    "port": "19530",
    "user": "",
    "password": "",
    "secure": False,
    'uri': "https://example.zillizcloud.com",
    'token': "token"
}
    
def load_base_template(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            base_template = file.read()  # Read all data from file
        return base_template
    except FileNotFoundError:
        return ""  # Return an empty string if file not found




def langchain_template():
    template = """ Answer using history as context
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    
    Always answer with description and file_path about file.
    history
    {history}
    
    Current conversation:
    Human: {input}
    AI assistant:
    """
    rag_prompt = ChatPromptTemplate.from_template(template)
    
    return rag_prompt
    





def split_mutiple_documents(current_path):
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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



def invoke_from_retriever(query, vector_store, llm, prompt_template, base_template_path=''):
    # Load the initial context
    initial_context_content = load_base_template(base_template_path)

    # Set up the components of the chain.
    setup_and_retrieval = RunnableParallel(
        history=RunnableLambda(lambda _: initial_context_content),  # Use RunnableLambda for static content
        # context=vector_store.as_retriever(),
        input=RunnablePassthrough()  # This can just pass the question as is
    )

    # Construct and invoke the chain
    rag_chain = setup_and_retrieval | prompt_template | llm
    return rag_chain.invoke(query)  


# initial_context = './base_template.md'
# docs_splits = split_mutiple_documents('./')
# prompt_template = langchain_template()
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.03) 

# vector_store = vector_store_milvus(embeddings, connection_args, COLLECTION_NAME )

# answer1 = invoke_from_retriever("what is article number about Industry background?" , vector_store, llm, prompt_template)
# answer2 = invoke_from_retriever("what is article number about Industry background?" , vector_store, llm, prompt_template, initial_context)


# print(answer1)
# print(answer2)




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

# # Call the function
# update_entity('./010.00 b.md', vector_store_exist)




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






# Create_collection_from_docs(docs_splits, embeddings ,COLLECTION_NAME,connection_args )





from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain











# memory.save_context({"input": "hi"}, {"output": "whats up"})
# memory.save_context({"input": "not much you"}, {"output": "not much"})


# print(memory,memory.save_context)
# print(memory.load_memory_variables({}))

####
    # setup_and_retrieval = RunnableParallel(
    #     history_chat=RunnableLambda(lambda _: initial_context_content),  # Use RunnableLambda for static content
    #     context=vector_store.as_retriever(),
    #     question=RunnablePassthrough()  # This can just pass the question as is
    # )

####
prompt = langchain_template()
llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0.03)
memory=ConversationBufferWindowMemory(k=5,memory_key="history",ai_prefix="AI Assistant")

# print(memory)
# print(memory.load_memory_variables({}))


template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

{history}

Current conversation:
Human: {input}
AI Assistant:"""
PROMPT = ChatPromptTemplate.from_template(template=template)

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
    prompt=PROMPT
)
output = conversation_with_summary.invoke("blue or red which color has high energy?")
print(output['response'])


print (conversation_with_summary.predict(input="Hello, write postfix number of our chat whenerver we talked for example AI Assistatnt :~~~~~ 1"))

print(conversation_with_summary.predict(input="What's the question just before?"))

print(conversation_with_summary.predict(input="What's climate is in east Asia"))

print(conversation_with_summary.predict(input="What's the weather today??"))


