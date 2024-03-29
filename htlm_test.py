import os
from pymilvus import MilvusClient, DataType

from langchain_community.vectorstores import Milvus

MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_URI = os.getenv("MILVUS_URI")
CONNECTION_ARGS = {'uri': MILVUS_URI, 'token': MILVUS_TOKEN}
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# 1. Set up a Milvus client
client = MilvusClient(
    uri = MILVUS_URI,
    token= MILVUS_TOKEN
)
res = client.describe_collection(
    collection_name=COLLECTION_NAME
)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args=connection_args,
            collection_name=collection_name,
            drop_old=False,
            auto_id=True
        )
print
print(client.delete(collection_name=COLLECTION_NAME, ids='448709686912353712'))

# print (client._get_connection()_get_row)
# print(res)

# res = client.upsert(
#     collection_name=COLLECTION_NAME,
#     data=data
# )

# print(res)
