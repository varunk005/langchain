from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_demo.llm import get_llm
import uuid


# ============================================================
# 1. CREATE DOCUMENTS
# ============================================================

document_1 = Document(
    page_content=(
        "Python was created by Guido van Rossum. "
        "It was first released in 1991."
    ),
    metadata={
        "topic": "python",
        "source": "programming_notes"
    }
)


document_2 = Document(
    page_content=(
        "Java was created at Sun Microsystems. "
        "It was released publicly in 1995."
    ),
    metadata={
        "topic": "java",
        "source": "programming_notes"
    }
)


document_3 = Document(
    page_content=(
        "JavaScript was created by Brendan Eich "
        "and first appeared in 1995."
    ),
    metadata={
        "topic": "javascript",
        "source": "web_notes"
    }
)


documents = [
    document_1,
    document_2,
    document_3
]


# ============================================================
# 2. CREATE AN EMBEDDING MODEL
#
# IMPORTANT:
#
# Claude is NOT being used here.
#
# This model converts text into vectors.
# ============================================================
print("Loading embedding model (first run may download)...") 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 3. CREATE IN-MEMORY VECTOR STORE
#
# The vector store needs an embedding model.
# ============================================================

vector_store = InMemoryVectorStore(
    embedding=embeddings
)


# ============================================================
# 4. CREATE UNIQUE IDS
#
# Each document gets its own ID.
# ============================================================

ids = [
    str(uuid.uuid4())
    for _ in documents
]


# ============================================================
# 5. ADD DOCUMENTS
#
# When add_documents() runs:
#
# document text
#     ↓
# embedding model
#     ↓
# vectors
#     ↓
# vector store
# ============================================================

vector_store.add_documents(
    documents=documents,
    ids=ids
)


print("Documents added successfully!")


# ============================================================
# 6. SEARCH THE VECTOR STORE
#
# User asks a normal English question.
#
# The QUERY is also converted into an embedding.
# Then LangChain compares that query vector
# with the stored document vectors.
# ============================================================

query = "Who created Python?"


results = vector_store.similarity_search(
    query,
    k=2
)


# ============================================================
# 7. PRINT RETRIEVED DOCUMENTS
# ============================================================

print("\nMost relevant documents:\n")

for document in results:

    print("Content:")
    print(document.page_content)

    print("Metadata:")
    print(document.metadata)

    print("-" * 50)


# ============================================================
# 8. NOW CREATE CLAUDE
#
# Claude has a DIFFERENT job.
#
# Embedding model:
#     finds relevant information
#
# Claude:
#     reads that information and generates an answer
# ============================================================

llm = get_llm()


# ============================================================
# 9. COMBINE RETRIEVED DOCUMENTS
# ============================================================

context = "\n\n".join(
    document.page_content
    for document in results
)


# ============================================================
# 10. SEND CONTEXT + QUESTION TO CLAUDE
# ============================================================

prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}
"""


response = llm.invoke(prompt)


print("\nClaude answer:\n")
print(response.content)


# ============================================================
#                       NOTES
# ============================================================


# ------------------------------------------------------------
# DOCUMENT
# ------------------------------------------------------------

# A LangChain Document normally contains:
#
# page_content
#     = actual text
#
# metadata
#     = extra information about that text
#
#
# Example:
#
# Document(
#     page_content="Python was created by Guido...",
#
#     metadata={
#         "topic": "python",
#         "source": "notes"
#     }
# )


# ------------------------------------------------------------
# WHAT DOES AN EMBEDDING MODEL DO?
# ------------------------------------------------------------

# It converts text into numbers.
#
#
# Example conceptually:
#
# "Python programming"
#
#       ↓ embedding model
#
# [0.12, -0.83, 0.42, ...]
#
#
# These numbers are called a VECTOR.


# ------------------------------------------------------------
# WHY DO WE NEED VECTORS?
# ------------------------------------------------------------

# Computers can compare vectors mathematically.
#
# Similar meanings usually produce vectors
# that are close together.
#
#
# Example:
#
# "Who invented Python?"
#
# and
#
# "Python was created by Guido van Rossum."
#
# have different words,
# but similar meaning.
#
# Their embeddings should therefore be relatively close.


# ------------------------------------------------------------
# WHAT HAPPENS DURING add_documents()?
# ------------------------------------------------------------

# vector_store.add_documents(documents)
#
#
# Internally:
#
# Document text
#       ↓
# embedding model
#       ↓
# vector
#       ↓
# store:
#
# vector
# document
# metadata
# ID


# ------------------------------------------------------------
# WHAT HAPPENS DURING similarity_search()?
# ------------------------------------------------------------

# Query:
#
# "Who created Python?"
#
#       ↓
# embedding model
#
# query vector
#
#       ↓
# compare with stored vectors
#
#       ↓
# closest documents returned


# ------------------------------------------------------------
# DOES CLAUDE CREATE THE EMBEDDINGS?
# ------------------------------------------------------------

# In this example:
#
# NO.
#
#
# HuggingFaceEmbeddings
#     ↓
# creates vectors
#
#
# Claude
#     ↓
# generates the final natural-language answer


# ============================================================
#                  COMPLETE FLOW
# ============================================================

#                INDEXING / STORAGE
#
#
# Document 1 ─┐
# Document 2 ─┼──> Embedding Model
# Document 3 ─┘
#                       ↓
#
#                    vectors
#                       ↓
#
#              InMemoryVectorStore
#
#
#
#                 SEARCH TIME
#
#
# User:
# "Who created Python?"
#
#         ↓
#
# Embedding Model
#
#         ↓
#
# Query Vector
#
#         ↓
#
# InMemoryVectorStore
#
#         ↓
#
# Compare vectors
#
#         ↓
#
# Relevant Document:
#
# "Python was created by Guido van Rossum..."
#
#         ↓
#
# Claude
#
#         ↓
#
# "Python was created by Guido van Rossum."


# ============================================================
#             MOST IMPORTANT THING TO REMEMBER
# ============================================================

# LLM:
#
# Claude
#     =
# reads / reasons / writes answers
#
#
# EMBEDDING MODEL:
#
# HuggingFaceEmbeddings
#     =
# text -> vectors
#
#
# VECTOR STORE:
#
# InMemoryVectorStore
#     =
# stores vectors and finds similar vectors