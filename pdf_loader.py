from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("THE SECRET IN ENGLISH.pdf")

documents = loader.load()

for doc in documents:
    print(doc.page_content)