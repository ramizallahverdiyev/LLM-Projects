def chunk_text(text: str, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # overlap tətbiqi

        if start < 0:
            start = 0

    return chunks