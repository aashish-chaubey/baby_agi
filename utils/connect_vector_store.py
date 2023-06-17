"""
Connect to Vector Store
"""

import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
EMBEDDING_SIZE = 1536
index = faiss.IndexFlatL2(EMBEDDING_SIZE)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
