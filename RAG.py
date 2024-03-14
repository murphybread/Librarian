import os


from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader


from langchain.schema import HumanMessage, SystemMessage


from langchain_community.llms import Bedrock






def get_llm_aws_bedrock (model="amazon.titan-text-express-v1"):
    
    # model_kwargs =  { #Anthropic 모델
    # "max_tokens_to_sample": 1024,
    # "temperature": 0, 
    # "top_k": 250, 
    # "top_p": 0.5, 
    # "stop_sequences": ["\n\nHuman:"] 
        
    # }
    
    llm = Bedrock(model_id = model, region_name="us-east-1")
    return llm
    

def get_llm_openai (model="gpt-3.5-turbo-0125"):
    
    
    llm = ChatOpenAI(model_name=model)
    
    return llm
    

    

def get_text_response (query,llm_model):
    
    #Step 1: Loading and splitting documents: 
    loader = TextLoader("./example.txt")
    documents = loader.load()
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    docs = text_splitter.split_documents(documents)
    
    #Step 2: Embedding generation and FAISS database creation:
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    
    
    #Step 3: Query the FAISS db to retrieve relevant document content
    query=query
    retriever = db.as_retriever()
    retrieved_docs = retriever.invoke(query)
    retrieved_context = retrieved_docs[0].page_content  # Assume using first doc's content
    
    # Step 4: Combine the retrieved document content with the original query
    combined_query = f"Query: {query} Context: {retrieved_context}\n"
    
    # Step 5: Use the combined query as a prompt for the ChatOpenAI model
    chat = llm_model  # Ensure the ChatOpenAI instance is correctly initialized
    response = chat.invoke([
        SystemMessage(content="Please use the following information to assist."),
        HumanMessage(content=combined_query)
    ])
    
    if hasattr(response, 'content'):
        return [response.content, retrieved_context]
    else:
        return [response,retrieved_context]  # Return the response directly if it doesn't have a 'content' attribute
    
def get_response_attribute (query,llm_model):
    
    #Step 1: Loading and splitting documents: 
    loader = TextLoader("./example.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    #Step 2: Embedding generation and FAISS database creation:
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    
    
    #Step 3: Query the FAISS db to retrieve relevant document content
    query=query
    retriever = db.as_retriever()
    retrieved_docs = retriever.invoke(query)
    retrieved_context = retrieved_docs[0].page_content  # Assume using first doc's content
    
    # Step 4: Combine the retrieved document content with the original query
    combined_query = f"Query: {query} \n Context: {retrieved_context}\n"
    
    # Step 5: Use the combined query as a prompt for the ChatOpenAI model
    chat = llm_model  # Ensure the ChatOpenAI instance is correctly initialized
    response = chat.invoke([
        SystemMessage(content="Please use the following information to assist."),
        HumanMessage(content=combined_query)
    ])
    
    
    print(f"Response type: {type(response)}")
    if hasattr(response, '__dict__'):
        print(f"Response attributes: {response.__dict__.keys()}")
        
    return print(f"respone of {llm_model}")
    


# q = "What is your LLM model name?"

# llm_b = get_llm_aws_bedrock()
# llm_o = get_llm_openai()
# llm_4 = get_llm_openai("gpt-4-0125-preview")


# print(get_text_response(q, llm_o))
# print(get_text_response(q, llm_b))
# print(get_text_response(q, llm_4))
