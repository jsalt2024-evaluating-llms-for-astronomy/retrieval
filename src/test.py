import unittest
from unittest.mock import patch, MagicMock
from hyde import HydeRetrievalSystem
from hyde_reranking import HydeCohereRetrievalSystem
from vector_store import EmbeddingClient, Document, DocumentLoader

class TestHydeRetrievalSystem(unittest.TestCase):
    def setUp(self):
        self.retrieval_system = HydeRetrievalSystem()

    def test_init(self):
        self.assertEqual(self.retrieval_system.generation_model, "claude-3-haiku-20240307")
        self.assertEqual(self.retrieval_system.embedding_model, "text-embedding-3-small")
        self.assertEqual(self.retrieval_system.temperature, 0.5)
        self.assertEqual(self.retrieval_system.max_doclen, 500)
        self.assertEqual(self.retrieval_system.generate_n, 1)
        self.assertEqual(self.retrieval_system.embed_query, True)
    
    def test_retrieve(self):
        query = "What is the stellar mass of the Milky Way?"
        arxiv_id = None
        top_k = 10
        return_scores = False
        
        self.retrieval_system.retrieve(query, arxiv_id, top_k, return_scores)

        self.retrieval_system.weight_citation = True
        self.retrieval_system.weight_keywords = True

        self.retrieval_system.retrieve(query, arxiv_id, top_k, return_scores)

class TestHydeCohereRetrievalSystem(unittest.TestCase):
    def setUp(self):
        self.retrieval_system = HydeCohereRetrievalSystem()
    
    def test_retrieve(self):
        query = "What is the stellar mass of the Milky Way?"
        arxiv_id = None
        top_k = 10
        return_scores = False
        
        self.retrieval_system.weight_citation = True
        self.retrieval_system.weight_keywords = True

        self.retrieval_system.retrieve(query, arxiv_id, top_k, return_scores)
        self.retrieval_system.retrieve(query, arxiv_id, top_k, return_scores, reweight = True)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
