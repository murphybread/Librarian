import os
import re
from pymilvus import Collection, connections, MilvusClient
from dotenv import load_dotenv
from langchain_community.vectorstores import Milvus


# Load environment variables
load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
BASE_FILE_PATH = os.getenv("BASE_FILE_PATH")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_URI = os.getenv("MILVUS_URI")
CONNECTION_ARGS = {'uri': MILVUS_URI, 'token': MILVUS_TOKEN}
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Connect to Milvus


# Retrieve the collection
vdb = Milvus(
            embedding_function=EMBEDDING_MODEL_NAME,
            connection_args=CONNECTION_ARGS,
            collection_name=COLLECTION_NAME,
            drop_old=False,
            auto_id=True
        )


# Define the auto_id
auto_id = "450644945527654610"

# Check if the entity exists
expr = f"Auto_id == {auto_id}"
pks = vdb.get_pks(expr)

if pks:
    print(f"Entity found: {pks}")
else:
    print(f"No entity found with auto_id: {auto_id}")


print(vdb.delete(pks))