from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma DB
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = db.as_retriever()

# Load Ollama LLM
llm = Ollama(model="llama3")

# User Question
query = input("Ask question: ")

# Condition check
if "secret" not in query.lower():
    print("\n⚠ This question is not related to the document.\n")
    exit()

# Retrieve documents
docs = retriever.invoke(query)

if not docs:
    print("❌ No information found.")
    exit()

# Combine context
context = "\n\n".join([doc.page_content for doc in docs[:3]])

# Prompt Template
prompt = PromptTemplate(
    template="""
You are an AI assistant.

Use ONLY the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

# Format prompt
final_prompt = prompt.format(context=context, question=query)

# Generate answer
response = llm.invoke(final_prompt)

print("\n🤖 AI Answer:\n")
print(response)