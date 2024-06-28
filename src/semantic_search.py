import json
from typing import List, Dict, Tuple
from collections import Counter
from datasets import load_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from tqdm import tqdm
from datetime import datetime
import yaml
from openai import OpenAI

import sys
sys.path.append('../evaluation')

from evaluate import RetrievalSystem, main as evaluate_main
from vector_store import EmbeddingClient, Document, DocumentLoader

class EmbeddingRetrievalSystem(RetrievalSystem):
    def __init__(self, dataset_path: str, embeddings_path: str = "../data/embeddings_matrix.npy", 
                 documents_path: str = "../data/documents.pkl", 
                 index_mapping_path: str = "../data/index_mapping.pkl"):
        self.dataset_path = dataset_path
        self.embeddings_path = embeddings_path
        self.documents_path = documents_path
        self.index_mapping_path = index_mapping_path
        
        self.embeddings = None
        self.documents = None
        self.index_mapping = None
        self.document_dates = []
        
        self.load_data()

        config = yaml.safe_load(open('../config.yaml', 'r'))
        self.client = EmbeddingClient(OpenAI(api_key=config['openai_api_key']))

    def parse_date(self, arxiv_id: str) -> datetime:
        try:
            year = int("20" + arxiv_id[:2])
            month = int(arxiv_id[2:4])
        except:
            year = 2023
            month = 1
        return datetime(year, month, 1)

    def load_data(self):
        print("Loading embeddings...")
        self.embeddings = np.load(self.embeddings_path)
        
        print("Loading documents...")
        with open(self.documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        print("Loading index mapping...")
        with open(self.index_mapping_path, 'rb') as f:
            self.index_mapping = pickle.load(f)
        
        print("Processing document dates...")
        self.document_dates = {doc.id: self.parse_date(doc.arxiv_id) for doc in self.documents}
        
        print("Data loaded successfully.")

    def retrieve(self, query: str, arxiv_id: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        query_date = self.parse_date(arxiv_id)
        
        # Get the query embedding
        query_embedding = self.get_query_embedding(query)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Filter and rank results
        filtered_results = []
        for doc_id, mappings in self.index_mapping.items():
            doc_date = self.document_dates[doc_id]
            if doc_date <= query_date:
                abstract_sim = similarities[mappings['abstract']] if 'abstract' in mappings else -np.inf
                conclusions_sim = similarities[mappings['conclusions']] if 'conclusions' in mappings else -np.inf
                if abstract_sim > conclusions_sim:
                    filtered_results.append((doc_id, 'abstract', abstract_sim))
                else:
                    filtered_results.append((doc_id, 'conclusions', conclusions_sim))
        
        # Sort and get top-k results
        top_results = sorted(filtered_results, key=lambda x: x[2], reverse=True)[:top_k]
        
        return top_results

    def get_query_embedding(self, query: str) -> np.ndarray:
        embedding = self.client.embed(query)
        return np.array(embedding, dtype=np.float32)
    
    def get_document_texts(self, doc_ids: List[str]) -> List[Dict[str, str]]:
        results = []
        for doc_id in doc_ids:
            doc = next((d for d in self.documents if d.id == doc_id), None)
            if doc:
                results.append({
                    'id': doc.id,
                    'abstract': doc.abstract,
                    'conclusions': doc.conclusions
                })
            else:
                print(f"Warning: Document with ID {doc_id} not found.")
        return results

def main():
    retrieval_system = EmbeddingRetrievalSystem("charlieoneill/jsalt-astroph-dataset")
    query = "What is the stellar mass of the Milky Way?"
    arxiv_id = "2301.00001"
    
    # Retrieve
    retrieved_docs = retrieval_system.retrieve(query, arxiv_id)
    print("Retrieved documents:")
    for doc_id, match_type, similarity in retrieved_docs:
        print(f"Document ID: {doc_id}, Matched on: {match_type}, Similarity: {similarity:.4f}")
    print()

    # Get document texts
    document_texts = retrieval_system.get_document_texts([doc_id for doc_id, _, _ in retrieved_docs])
    
    # Go through and print the texts
    for doc, (_, match_type, _) in zip(document_texts, retrieved_docs):
        print(f"Document ID: {doc['id']}")
        print(f"Matched on: {match_type}")
        if match_type == 'abstract':
            print(f"Abstract: {doc['abstract'][:500]}...")
        else:
            print(f"Conclusion: {doc['conclusions'][:500]}...")
        print()

if __name__ == "__main__":
    main()