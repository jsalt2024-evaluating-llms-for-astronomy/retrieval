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
import cohere
from openai import OpenAI
from hyde import HydeRetrievalSystem

class HydeCohereRetrievalSystem(HydeRetrievalSystem):
    def __init__(self, config_path: str = '../config.yaml', embeddings_path: str = "../data/vector_store/embeddings_matrix.npy", metadata_path = "../data/vector_store/metadata.json",
                 documents_path: str = "../data/vector_store/documents.pkl", index_mapping_path: str = "../data/vector_store/index_mapping.pkl", 
                 generation_model: str = "claude-3-haiku-20240307", embedding_model: str = "text-embedding-3-small", 
                 temperature: float = 0.5, max_doclen: int = 500, generate_n: int = 1, embed_query = True, 
                 weight_citation = False, weight_date = False, weight_keywords = False):
        
        super().__init__(config_path=config_path, embeddings_path=embeddings_path, documents_path=documents_path, index_mapping_path=index_mapping_path,
                         metadata_path=metadata_path, generation_model=generation_model, embedding_model=embedding_model,
                         temperature=temperature, max_doclen=max_doclen, generate_n=generate_n, embed_query=embed_query, 
                         weight_citation=weight_citation, weight_date=weight_date, weight_keywords=weight_keywords)
        
        self.cohere_client = cohere.Client(self.cohere_key)

    def retrieve(self, query: str, arxiv_id: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        top_results = super().retrieve(query, arxiv_id, top_k = 250) # should that be hard coded
        
        doc_texts = self.get_document_texts(top_results)
        
        docs_for_rerank = [f"Abstract: {doc['abstract']}\nConclusions: {doc['conclusions']}" for doc in doc_texts]
        
        reranked_results = self.cohere_client.rerank(
            query=query,
            documents=docs_for_rerank,
            model='rerank-english-v2.0',
            top_n=top_k
        )
        
        final_results = []
        for result in reranked_results.results:
            doc_id = top_results[result.index]
            doc_text = docs_for_rerank[result.index]
            score = float(result.relevance_score)
            #final_results.append((doc_id, doc_text, score))
            final_results.append(doc_id)
        
        return final_results

    def embed_docs(self, docs: List[str]):
        return self.client.embed_batch(docs)

def main():
    retrieval_system = HydeCohereRetrievalSystem(embeddings_path="../data/vector_store/embeddings_matrix.npy",
                         documents_path="../data/vector_store/documents.pkl",
                         index_mapping_path="../data/vector_store/index_mapping.pkl", 
                         generate_n=1, embed_query=False, max_doclen=300, weight_citation=True)
    evaluate_main(retrieval_system, "Rerank + Citations")

if __name__ == "__main__":
    main()