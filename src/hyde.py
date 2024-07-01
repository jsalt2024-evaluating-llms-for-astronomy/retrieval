import anthropic
from datasets import load_dataset
import sys
import requests
import numpy as np
sys.path.append('../evaluation')
from typing import List, Dict, Tuple
from evaluate import RetrievalSystem, main as evaluate_main
from semantic_search import EmbeddingRetrievalSystem
import yaml
import json
import transformers
from sklearn.metrics.pairwise import cosine_similarity
from vector_store import EmbeddingClient, Document, DocumentLoader
import concurrent.futures

# use generate_n = 0, embed_query = True to do basic vector search (no generation)
class HydeRetrievalSystem(EmbeddingRetrievalSystem):
    def __init__(self, config_path: str, dataset_path: str, embeddings_path: str = "../data/vector_store/embeddings_matrix.npy", 
                 documents_path: str = "../data/vector_store/documents.pkl", index_mapping_path: str = "../data/vector_store/index_mapping.pkl", 
                 generation_model: str = "claude-3-haiku-20240307", embedding_model: str = "text-embedding-3-small", 
                 temperature: float = 0.5, max_doclen: int = 500, generate_n: int = 1, embed_query = True):
        
        super().__init__(dataset_path = dataset_path, embeddings_path = embeddings_path, documents_path = documents_path, index_mapping_path = index_mapping_path)

        if max_doclen * generate_n > 8191:
            raise ValueError("Too many tokens. Please reduce max_doclen or generate_n.")
        
        self.embedding_model = embedding_model
        self.generation_model = generation_model

        # HYPERPARAMETERS
        self.temperature = temperature
        self.max_doclen = max_doclen
        self.generate_n = generate_n
        self.embed_query = embed_query

        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
            self.anthropic_key = config['anthropic_api_key']
            # self.openai_key = config['openai_api_key']
        
        self.generation_client = anthropic.Anthropic(api_key = self.anthropic_key)

    def retrieve(self, query: str, arxiv_id: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        docs = self.generate_docs(query)
        doc_embeddings = self.embed_docs(docs)

        if self.embed_query: 
            query_emb = self.embed_docs([query])[0]
            doc_embeddings.append(query_emb)
        
        embedding = np.mean(np.array(doc_embeddings), axis = 0)
        query_date = self.parse_date(arxiv_id)

        top_results = self.rank_and_filter(embedding, query_date = query_date, top_k = top_k)
        
        return top_results

    def generate_doc(self, query: str):
        message = self.generation_client.messages.create(
                model = self.generation_model,
                max_tokens = self.max_doclen,
                temperature = self.temperature,
                system = """You are an expert astronomer. Given a scientific query, generate the abstract and conclusion of an expert-level research paper
                            that answers the question. Stick to a maximum length of {} tokens and return just the text of the abstract and conclusion.
                            Do not include labels for any section. Use research-specific jargon.""".format(self.max_doclen),
                
                messages=[{ "role": "user",
                        "content": [{"type": "text", "text": query,}] }]
            )

        return message.content[0].text
    
    def generate_docs(self, query: str):
        docs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_query = {executor.submit(self.generate_doc, query): query for i in range(self.generate_n)}
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    data = future.result()
                    docs.append(data)
                except Exception as exc:
                    print(f'Query {query} generated an exception: {exc}')
        return docs

    def embed_docs(self, docs: List[str]):
        vecs = self.client.embed_batch(docs)
        return vecs

def main():
    retrieval_system = HydeRetrievalSystem(dataset_path = "charlieoneill/jsalt-astroph-dataset",
                         embeddings_path = "/users/christineye/retrieval/data/vector_store/embeddings_matrix.npy",
                         documents_path = "/users/christineye/retrieval/data/vector_store/documents.pkl",
                         index_mapping_path = "/users/christineye/retrieval/data/vector_store/index_mapping.pkl", config_path = "/users/christineye/retrieval/config.yaml", 
                                     generate_n = 1, embed_query = False, max_doclen = 300)
    evaluate_main(retrieval_system, "BaseHyDE")

if __name__ == "__main__":
    main()