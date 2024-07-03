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
from filters import CitationFilter, DateFilter, KeywordFilter

import sys
sys.path.append('../evaluation')

from evaluate import RetrievalSystem, main as evaluate_main
from vector_store import EmbeddingClient, Document, DocumentLoader

class EmbeddingRetrievalSystem(RetrievalSystem):
    def __init__(self, embeddings_path: str = "../data/vector_store/embeddings_matrix.npy", 
                 documents_path: str = "../data/vector_store/documents.pkl", 
                 index_mapping_path: str = "../data/vector_store/index_mapping.pkl",
                 metadata_path: str = "../data/vector_store/metadata.json", weight_citation = False, weight_date = False, weight_keywords = False):
        
        self.embeddings_path = embeddings_path
        self.documents_path = documents_path
        self.index_mapping_path = index_mapping_path
        self.metadata_path = metadata_path
        self.weight_citation = weight_citation
        self.weight_date = weight_date
        self.weight_keywords = weight_keywords

        self.embeddings = None
        self.documents = None
        self.index_mapping = None
        self.metadata = None
        self.document_dates = []
        
        self.load_data()
        self.init_filters()

        config = yaml.safe_load(open('../config.yaml', 'r'))
        self.client = EmbeddingClient(OpenAI(api_key=config['openai_api_key']))

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
        
        if os.path.exists(self.metadata_path):
            print("Loading metadata...")
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            print("Could not find metadata.")
        
        print("Data loaded successfully.")
    
    def init_filters(self):
        print("Loading filters...")
        if self.weight_citation: 
            self.citation_filter = CitationFilter(metadata = self.metadata)
        
        if self.weight_date:
            self.date_filter = DateFilter(document_dates = self.document_dates)
        
        if self.weight_keywords:
            self.keyword_filter = KeywordFilter(index_path = "../data/vector_store/keyword_index.json", metadata = self.metadata, remove_capitals = True)

    def retrieve(self, query: str, arxiv_id: str, top_k: int = 10, return_scores = False, ) -> List[Tuple[str, str, float]]:
        query_date = self.parse_date(arxiv_id)
        
        # Get the query embedding
        query_embedding = self.get_query_embedding(query)
        
        top_results = self.rank_and_filter(query_embedding, query_date, top_k, return_scores = return_scores)
        
        return top_results
    
    def rank_and_filter(self, query_embedding: np.ndarray, query_date, top_k: int = 10, return_scores = False) -> List[Tuple[str, str, float]]:
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)  #cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Filter and rank results
        filtered_results = []
        for doc_id, mappings in self.index_mapping.items():
            doc_date = self.document_dates[doc_id]
            
            if doc_date <= query_date: # this can stay here
                abstract_sim = similarities[mappings['abstract']] if 'abstract' in mappings else -np.inf
                conclusions_sim = similarities[mappings['conclusions']] if 'conclusions' in mappings else -np.inf
                
                if abstract_sim > conclusions_sim: 
                    filtered_results.append([doc_id, "abstract", abstract_sim])
                else: 
                    filtered_results.append([doc_id, "conclusions", conclusions_sim])
                
        
        # Sort and weight and get top-k results
        if self.weight_citation: self.citation_filter.filter(filtered_results)
            
        top_results = sorted(filtered_results, key=lambda x: x[2], reverse=True)[:top_k]

        if return_scores:
            return {doc_id: doc[2] for doc in top_results}

        # Only keep the document IDs
        top_results = [doc[0] for doc in top_results]
        return top_results

    def get_query_embedding(self, query: str) -> np.ndarray:
        embedding = self.client.embed(query)
        return np.array(embedding, dtype = np.float32)
    
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
    retrieval_system = EmbeddingRetrievalSystem()
    #evaluate_main(retrieval_system, "BaseSemanticSearch")
    query = "What is the stellar mass of the Milky Way?"
    arxiv_id = "2301.00001"
    top_k = 10
    results = retrieval_system.retrieve(query, arxiv_id, top_k)
    print(f"Retrieved documents: {results}")

if __name__ == "__main__":
    main()