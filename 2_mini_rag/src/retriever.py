def retrieve(query: str, embedder, vectordb, k=3):
    q_emb = embedder.encode([query])
    results = vectordb.search(q_emb, k)
    context = "\n\n".join([r["text"] for r in results])
    return context
