from pathlib import Path

# s = "can you tell me more detailed? file_path: 400/440/440.00/440.00.md and else thing?"
# s1 = '.'

# print(Path(s1).is_file())

# def extract_path(q, keyword='file_path: '):    
#     # 'file_path: ' Find path by ketword
    
    
#     start_index = q.find(keyword)
    
#     if keyword not in q:
#         return q

#     if start_index != -1:
#         # 'file_path: ' 
#         start_index += len(keyword)
#         temp_extract = s[start_index:]

#         # Find postion '.md' from temp_extract string
#         end_index = temp_extract.find('.md')

#         if end_index != -1:
#             # Extract final path that include '.md'
#             extracted_path = temp_extract[:end_index + len('.md')]
#             print(extracted_path)
#         else:
#             print("'.md' not found.")
#     else:
#         print("'file_path: ' pattern not found.")

query = "AAA"
file_session = "XYZ123"  # This should be your session identifier or relevant information
file_info = f"Information: {file_session}\n"  # Using an f-string for dynamic insertion

query = file_info + query  # Combining the strings
# print(query)

from langchain_community.vectorstores import Milvus
from pymilvus import Collection,connections,MilvusClient
import os

from langchain_openai import OpenAIEmbeddings

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
BASE_FILE_PATH= os.getenv("BASE_FILE_PATH")

MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_URI = os.getenv("MILVUS_URI")
CONNECTION_ARGS = {'uri': MILVUS_URI, 'token': MILVUS_TOKEN}
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

token = MILVUS_TOKEN
uri = MILVUS_URI


embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

vectorstore =  Milvus(
            embedding_function=embeddings,
            connection_args=CONNECTION_ARGS,
            collection_name=COLLECTION_NAME,
            drop_old=False,
            auto_id=True
        )
expr = f"source == './base_template.md'"
pks = vectorstore.get_pks(expr)

print(pks, type(pks))


from pymilvus import Collection

pks = [448986445593458408]  # Provide a list of entity IDs to delete

client = MilvusClient(uri = uri, token= token)
res =  client.delete( collection_name=COLLECTION_NAME, ids=pks)
print(res)