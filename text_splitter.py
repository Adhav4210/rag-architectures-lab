from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("THE SECRET IN ENGLISH.pdf")
documents = loader.load()

# Text Splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

print("Total chunks:", len(docs))

for i, doc in enumerate(docs[:3]):
    print(f"\nChunk {i+1}")
    print(doc.page_content)