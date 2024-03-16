from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
# update_entity('./010.00 b.md', vectorstore)