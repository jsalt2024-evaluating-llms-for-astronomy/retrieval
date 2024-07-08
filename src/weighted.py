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
from hyde import HydeRetrievalSystem
from bag_of_words import BagOfWordsRetrievalSystem

class WeightedRetrievalSystem():
    def __init__(self, bow_weight: float = 0.5):
        self.bow_weight = 0.5
        self.make_embed()
        self.make_bow()
    
    def make_embed(self, dataset_path = "charlieoneill/jsalt-astroph-dataset",
                         embeddings_path = "/users/christineye/retrieval/data/vector_store/embeddings_matrix.npy",
                         documents_path = "/users/christineye/retrieval/data/vector_store/documents.pkl",
                         index_mapping_path = "/users/christineye/retrieval/data/vector_store/index_mapping.pkl"):
        
        self.embed = EmbeddingRetrievalSystem(embeddings_path = embeddings_path, 
                                         documents_path = documents_path, index_mapping_path = index_mapping_path)
        
        
    def make_bow(self, dataset_path: str = "charlieoneill/jsalt-astroph-dataset", index_path: str = "../data/bow_index.pkl", remove_capitals: bool = True):
        self.bow = BagOfWordsRetrievalSystem(dataset_path=dataset_path, index_path=index_path, remove_capitals=remove_capitals)

    def retrieve(self, query: str, arxiv_id: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        bow_scores = self.bow.retrieve(query, arxiv_id, return_scores = True, top_k = 500)
        # print('Computed bag of words scores.')
        
        embed_scores = self.embed.retrieve(query, arxiv_id, return_scores = True, top_k = 500)
        # print('Computed embedding scores.')
        
        candidate_ids = set(bow_scores.keys()) & set(embed_scores.keys())
        # print('{} candidate documents.'.format(len(candidate_ids)))

        weighted_scores = {doc_id: bow_scores[doc_id] * self.bow_weight + embed_scores[doc_id] * (1 - self.bow_weight) for doc_id in candidate_ids}
        
        sorted_scores = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        sorted_doc_ids = [doc_id for doc_id, _ in sorted_scores]
        
        return sorted_doc_ids
        

def main():
    retrieval_system = WeightedRetrievalSystem()
    evaluate_main(retrieval_system, "BoW + Embedding")

if __name__ == "__main__":
    main()