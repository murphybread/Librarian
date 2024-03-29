from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

s = "HUMAN:we start talked about gitlab\nAI:Based on the information available in Murphy's library, GitLab is mentioned as a DevSecOps solution that can be installed for enterprise use. This suggests that GitLab offers features and functionalities tailored towards integrating development, security, and operations, making it suitable for enterprise-level applications.\n\nfile_path: 200\\210\\210.20\\210.20 a.md\ndescription: The solution about Gitlab. GitLab is one devsecops solution and can be installed for enterprise."
doc =  Document(page_content=s)
dc=  text_splitter.split_documents([doc])

print(dc)


from langchain_community.document_loaders import TextLoader

loader = TextLoader('./test.txt', encoding='utf-8')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print(docs)