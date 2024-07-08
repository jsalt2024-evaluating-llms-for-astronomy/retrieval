import unittest
from unittest.mock import patch, MagicMock
from hyde import HydeRetrievalSystem

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

if __name__ == '__main__':
    unittest.main()