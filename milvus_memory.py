from pymilvus import Collection,connections
import uuid


class MilvusMemory:
    def __init__(self, uri, token, collection_name):
        connections.connect("default", uri=uri, token=token)
        self.collection = Collection(name=collection_name)

    def memory_insert(self, query, embedding):
        print("---------------collection insert entity start-------------")
        vector = embedding.embed_query(query)
        session = str(uuid.uuid1())
        self.collection.insert([[session], [query], [vector]])
        print("-----------------collection insert entity end------------")