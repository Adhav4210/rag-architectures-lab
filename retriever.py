from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma vector DB
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

# Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# User question
query = "What is the law of attraction?"

# Retrieve relevant chunks
results = retriever.invoke(query)

# Print results
for i, doc in enumerate(results):
    print(f"\nResult {i+1}")
    print(doc.page_content)