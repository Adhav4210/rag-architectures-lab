from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1: Load PDF
loader = PyPDFLoader("THE SECRET IN ENGLISH.pdf")
documents = loader.load()

# Step 2: Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# Step 3: Local Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Store in Chroma DB
vector_db = Chroma.from_documents(
    docs,
    embedding,
    persist_directory="chroma_db"
)

print("Embeddings created successfully!")