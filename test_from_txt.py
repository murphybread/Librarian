import os
import re
from pymilvus import Collection, connections, MilvusClient
from dotenv import load_dotenv

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
connections.connect(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN
)

# Retrieve the collection
collection = Collection(COLLECTION_NAME)

# Define the auto_id
auto_id = 450644945527654614

# Check if the entity exists
expr_check = f"Auto_id == {auto_id}"
results = collection.query(expr_check)
print(collection.schema)

if results:
    print(f"Entity found: {results}")
else:
    print(f"No entity found with Auto_id: {auto_id}")

# Delete the entity if it exists
expr_delete = f"Auto_id in [{auto_id}]"
try:
    delete_result = collection.delete(expr_delete)
    print(f"Delete result: {delete_result}")
except Exception as e:
    print(f"Failed to delete entity with Auto_id {auto_id}: {e}")
