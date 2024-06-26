import json
from typing import List, Dict
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

import sys
sys.path.append('../evaluation')

from evaluate import RetrievalSystem, main as evaluate_main

class BagOfWordsRetrievalSystem(RetrievalSystem):
    def __init__(self, dataset_path: str, index_path: str = "bow_index.pkl"):
        self.dataset_path = dataset_path
        self.index_path = index_path
        self.documents = []
        self.document_ids = []
        self.tfidf_matrix = None
        self.vectorizer = None
        
        # Download NLTK stopwords
        nltk.download('stopwords', quiet=True)
        self.stopwords = set(stopwords.words('english'))
        
        self.load_or_build_index()

    def preprocess_text(self, text: str) -> str:
        # Basic preprocessing: lowercase and remove punctuation
        text = ''.join(char.lower() for char in text if char.isalnum() or char.isspace())
        # Remove stopwords
        return ' '.join(word for word in text.split() if word not in self.stopwords)

    def build_index(self):
        print("Building new index...")
        # Load the dataset
        dataset = load_dataset(self.dataset_path, split="train")

        # Preprocess and store documents with progress bar
        for paper in tqdm(dataset, desc="Processing documents", unit="doc"):
            if paper['introduction'] and paper['conclusions']:
                combined_text = self.preprocess_text(paper['introduction'] + ' ' + paper['conclusions'])
                self.documents.append(combined_text)
                self.document_ids.append(f"{paper['subfolder']}/{paper['filename']}")

        # Create TF-IDF matrix
        print("Creating TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer(
            lowercase=False,  # We've already lowercased in preprocess_text
            token_pattern=r'\b\w+\b',  # Match any word character
            stop_words=None,  # We've already removed stopwords in preprocess_text
            max_df=0.95,  # Remove terms that appear in more than 95% of documents
            min_df=2  # Remove terms that appear in less than 2 documents
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

        # Save the index
        print("Saving index...")
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'document_ids': self.document_ids,
                'tfidf_matrix': self.tfidf_matrix,
                'vectorizer': self.vectorizer
            }, f)
        print("Index built and saved successfully.")

    def load_index(self):
        print("Loading existing index...")
        with open(self.index_path, 'rb') as f:
            index_data = pickle.load(f)
            self.document_ids = index_data['document_ids']
            self.tfidf_matrix = index_data['tfidf_matrix']
            self.vectorizer = index_data['vectorizer']
        print("Index loaded successfully.")

    def load_or_build_index(self):
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.build_index()

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        # Preprocess the query
        processed_query = self.preprocess_text(query)

        # Transform the query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top-k document indices
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Return top-k document IDs
        return [self.document_ids[i] for i in top_indices]

def main():
    # Initialize and prepare the retrieval system
    retrieval_system = BagOfWordsRetrievalSystem("charlieoneill/jsalt-astroph-dataset")

    # Run the evaluation
    evaluate_main(retrieval_system)

if __name__ == "__main__":
    main()