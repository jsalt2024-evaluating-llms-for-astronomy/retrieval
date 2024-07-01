import json
from typing import List, Dict, Tuple
from collections import Counter
from datasets import load_dataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
from datetime import datetime

import sys
sys.path.append('../evaluation')

from evaluate import RetrievalSystem, main as evaluate_main

class BagOfWordsRetrievalSystem(RetrievalSystem):
    def __init__(self, dataset_path: str, index_path: str = "../data/bow_index.pkl", remove_capitals: bool = True):
        self.dataset_path = dataset_path
        self.index_path = index_path
        self.remove_capitals = remove_capitals
        self.documents = []
        self.document_ids = []
        self.document_dates = []
        self.tfidf_matrix = None
        self.vectorizer = None
        
        nltk.download('stopwords', quiet=True)
        self.stopwords = set(stopwords.words('english'))
        
        self.load_or_build_index()

    def preprocess_text(self, text: str) -> str:
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        if self.remove_capitals:
            text = text.lower()
        return ' '.join(word for word in text.split() if word.lower() not in self.stopwords)

    def parse_date(self, arxiv_id: str) -> datetime:
        try:
            year = int("20" + arxiv_id[:2])
            month = int(arxiv_id[2:4])
        except:
            year = 2023
            month = 1
        return datetime(year, month, 1)

    def build_index(self):
        print("Building new index...")
        dataset = load_dataset(self.dataset_path, split="train")

        for paper in tqdm(dataset, desc="Processing documents", unit="doc"):
            if paper['introduction'] and paper['conclusions']:
                combined_text = self.preprocess_text(paper['introduction'] + ' ' + paper['conclusions'])
                self.documents.append(combined_text)
                self.document_ids.append(paper['arxiv_id'])
                self.document_dates.append(self.parse_date(paper['arxiv_id']))

        print("Creating TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer(
            lowercase=self.remove_capitals,
            token_pattern=r'\b\w+\b',
            stop_words=None,
            max_df=0.95,
            min_df=2
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

        print("Saving index...")
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'document_ids': self.document_ids,
                'document_dates': self.document_dates,
                'tfidf_matrix': self.tfidf_matrix,
                'vectorizer': self.vectorizer,
                'remove_capitals': self.remove_capitals
            }, f)
        print("Index built and saved successfully.")

    def load_index(self):
        print("Loading existing index...")
        with open(self.index_path, 'rb') as f:
            index_data = pickle.load(f)
            self.document_ids = index_data['document_ids']
            self.document_dates = index_data['document_dates']
            self.tfidf_matrix = index_data['tfidf_matrix']
            self.vectorizer = index_data['vectorizer']
        print("Index loaded successfully.")

    def load_or_build_index(self):
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.build_index()

    def retrieve(self, query: str, arxiv_id: str, top_k: int = 10) -> List[str]:
        query_date = self.parse_date(arxiv_id)
        processed_query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])

        # Filter documents by date
        valid_indices = [i for i, date in enumerate(self.document_dates) if date <= query_date]
        filtered_tfidf_matrix = self.tfidf_matrix[valid_indices]

        similarities = cosine_similarity(query_vector, filtered_tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [self.document_ids[valid_indices[i]] for i in top_indices]

def main():
    retrieval_system = BagOfWordsRetrievalSystem("charlieoneill/jsalt-astroph-dataset", remove_capitals=True)
    evaluate_main(retrieval_system, "BagOfWordsDateFiltered")

if __name__ == "__main__":
    main()