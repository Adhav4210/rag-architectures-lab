from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Step 1: Load PDF
loader = PyPDFLoader("THE SECRET IN ENGLISH.pdf")
documents = loader.load()

# Step 2: Create Recursive Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# Step 3: Split Documents
docs = text_splitter.split_documents(documents)

# Step 4: Print Results
print("Total Chunks:", len(docs))

for i, doc in enumerate(docs[:3]):
    print(f"\nChunk {i+1}")
    print(doc.page_content)