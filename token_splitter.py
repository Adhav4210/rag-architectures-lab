from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Step 1: Load PDF
loader = PyPDFLoader("THE SECRET IN ENGLISH.pdf")
documents = loader.load()

# Step 2: Create Token Text Splitter
text_splitter = TokenTextSplitter(
    chunk_size=200,      # tokens per chunk
    chunk_overlap=50     # overlap tokens
)

# Step 3: Split Documents
docs = text_splitter.split_documents(documents)

# Step 4: Print Results
print("Total Chunks:", len(docs))

for i, doc in enumerate(docs[:3]):
    print(f"\nChunk {i+1}")
    print(doc.page_content)