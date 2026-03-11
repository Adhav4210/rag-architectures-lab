from langchain_community.document_loaders import WebBaseLoader

# Website URL
loader = WebBaseLoader("https://studyreadyfuture.com/")

documents = loader.load()

for doc in documents:
    print(doc.page_content)