from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# -------------------------
# Load LLM
# -------------------------

print("Loading LLM...")

llm = Ollama(model="llama3")

print("LLM Loaded ✅")

# -------------------------
# Load Embedding
# -------------------------

print("Loading Embedding Model...")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Loaded ✅")

# -------------------------
# Load Vector DB
# -------------------------

print("Loading Vector Database...")

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = db.as_retriever(search_kwargs={"k":3})

print("Vector DB Loaded ✅")

# -------------------------
# Ask Question
# -------------------------

user_query = input("\nAsk question: ")

# -------------------------
# Retrieval
# -------------------------

docs = retriever.invoke(user_query)

retrieved_content = ""

for doc in docs:
    retrieved_content += doc.page_content + "\n"

# -------------------------
# CRAG Prompt
# -------------------------

prompt = f"""
You are a Corrective RAG (CRAG) AI system responsible for evaluating retrieved knowledge before generating an answer.

USER QUERY:
{user_query}

RETRIEVED CONTENT:
{retrieved_content}

STEP 1: CONTENT EVALUATION

Evaluate the retrieved content using:

Relevance Score (0–1)
Completeness Score (0–1)
Accuracy Score (0–1)
Specificity Score (0–1)

Determine overall content quality:
Excellent
Good
Fair
Poor

STEP 2: CORRECTIVE LOGIC

If content quality is FAIR or POOR:
Do NOT generate the answer.

Instead:
Suggest a refined query and explain why retrieval failed.

STEP 3: ANSWER GENERATION

If content quality is GOOD or EXCELLENT:
Generate the final answer using retrieved content.

Confidence Levels:
High
Medium
Low

STEP 4: RESPONSE FORMAT

Content Quality:
Relevance Score:
Completeness Score:
Accuracy Score:
Specificity Score:

Retrieval Level Used:
P

Confidence Level:

If Fair or Poor:
Refined Query:
Reason for Correction:

If Good or Excellent:
Answer:
Information Coverage:
"""

# -------------------------
# LLM Evaluation
# -------------------------

response = llm.invoke(prompt)

print("\nCRAG Evaluation Result\n")
print(response)

print("\nCorrective RAG Pipeline Finished 🚀")