import os
from src.reader import load_document
from src.chunker import chunk_text
from src.embedder import Embedder
from src.vectordb import VectorDB
from src.retriever import retrieve
from src.answer import ask_llm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

doc_path ="data/raw/pdf-sample.pdf"
# 1. Reading the document
text = load_document(doc_path)

# 2. Chunking the text
chunks = chunk_text(text)

# 3. Embedding the chunks
embedder = Embedder()
embeddings = embedder.encode(chunks)

# 4. Preparing metadata
metadata = [{"text": chunks[i], "chunk_id": i} for i in range(len(chunks))]

# 5. Adding to FAISS vector database
vectordb = VectorDB(dim=embeddings.shape[1])
vectordb.add(embeddings, metadata)
vectordb.save()

# 6. Asking a question
query = "What is the main purpose of the document?"
context = retrieve(query, embedder, vectordb)

# 7. Getting the answer
answer = ask_llm(query, context, model="llama3.2")

print("\nFINAL ANSWER:\n", answer)