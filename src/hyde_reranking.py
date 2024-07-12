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
from temporal import analyze_temporal_query

class HydeCohereRetrievalSystem(HydeRetrievalSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cohere_client = cohere.Client(self.config['cohere_api_key'])

    def retrieve(self, query: str, arxiv_id: str = None, top_k: int = 10, return_scores = False, reweight = False) -> List[Tuple[str, str, float]]:
        time_result, time_taken = analyze_temporal_query(query, self.anthropic_client)
        
        top_results = super().retrieve(query, arxiv_id, top_k = 250, time_result = time_result)
        
        doc_texts = self.get_document_texts(top_results)
        
        docs_for_rerank = [f"Abstract: {doc['abstract']}\nConclusions: {doc['conclusions']}" for doc in doc_texts]
        
        if len(docs_for_rerank) == 0:
            return []
        
        reranked_results = self.cohere_client.rerank(
            query=query,
            documents=docs_for_rerank,
            model='rerank-english-v3.0',
            top_n=top_k
        )
        
        final_results = []
        for result in reranked_results.results:
            doc_id = top_results[result.index]
            doc_text = docs_for_rerank[result.index]
            score = float(result.relevance_score)
            final_results.append([doc_id, "", score])

        if reweight:
            if time_result['has_temporal_aspect']:
                final_results = self.date_filter.filter(final_results, time_score = time_result['expected_recency_weight'])
            
            if self.weight_citation: self.citation_filter.filter(final_results)
        
        if return_scores:
            return {result[0]: result[2] for result in final_results}

        return [doc[0] for doc in final_results]

    def embed_docs(self, docs: List[str]):
        return self.client.embed_batch(docs)

def main():
    retrieval_system = HydeCohereRetrievalSystem(embeddings_path="../data/vector_store/embeddings_matrix.npy",
                         documents_path="../data/vector_store/documents.pkl",
                         index_mapping_path="../data/vector_store/index_mapping.pkl", 
                         generate_n=1, embed_query=False, max_doclen=500, weight_citation=False, weight_keywords = True)
    evaluate_main(retrieval_system, "Rerank + Keywords")
    # query = "What is the stellar mass of the Milky Way?"
    # arxiv_id = "2301.00001"
    # top_k = 10
    # results = retrieval_system.retrieve(query, arxiv_id, top_k)
    # print(f"Retrieved documents: {results}")

if __name__ == "__main__":
    main()
