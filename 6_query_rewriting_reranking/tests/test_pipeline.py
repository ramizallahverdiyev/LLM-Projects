import unittest
from src.query_rewriting.hyde_rewriter import HyDERewriter
from src.embeddings.build_index import FAISSRetriever
from src.reranking.bge_reranker import BGEReranker
from src.context_selection.select_context import ContextSelector
from sentence_transformers import SentenceTransformer
import json
import os

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load documents
        with open("data/documents.json", "r", encoding="utf-8") as f:
            cls.documents = json.load(f)
        
        # Initialize modules
        cls.rewriter = HyDERewriter()
        cls.retriever = FAISSRetriever(nlist=10)  # small nlist for test
        cls.retriever.documents = cls.documents
        cls.reranker = BGEReranker()
        cls.selector = ContextSelector(max_context_tokens=100)

    def test_hyde_rewriter(self):
        query = "Best AI universities in Azerbaijan"
        rewritten = self.rewriter.rewrite(query, max_length=50)
        self.assertIsInstance(rewritten, str)
        self.assertTrue(len(rewritten) > 0)

    def test_faiss_retriever(self):
        self.retriever.build_index()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(["Test query"], convert_to_numpy=True)
        top_docs = self.retriever.search(query_embedding, top_k=3)
        self.assertEqual(len(top_docs), 3)

    def test_reranker(self):
        sample_docs = self.documents[:3]
        sorted_docs = self.reranker.rerank("Sample query", sample_docs)
        self.assertEqual(len(sorted_docs), 3)
        # Check if score field exists
        for doc in sorted_docs:
            self.assertIn("score", doc)

    def test_context_selector(self):
        sample_docs = [{"title": "Doc1", "text": "This is a sample document text."}]
        contexts = self.selector.select(sample_docs)
        self.assertTrue(len(contexts) > 0)
        self.assertIn("title", contexts[0])
        self.assertIn("text", contexts[0])

if __name__ == "__main__":
    unittest.main()
