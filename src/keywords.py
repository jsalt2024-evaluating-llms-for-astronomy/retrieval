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
from datetime import datetime
import sys
sys.path.append('../evaluation')
from evaluate import RetrievalSystem, main as evaluate_main

import spacy
from collections import Counter
from string import punctuation
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)
stopwords = set(stopwords.words('english')) 
nlp = spacy.load("en_core_web_sm")
spacy.cli.download('en_core_web_sm')
nlp.add_pipe("textrank")


class KeywordRetrievalSystem(RetrievalSystem):
    def __init__(self, index_path: str = "../data/vector_store/keyword_index.json", metadata_path: str = "../data/vector_store/metadata.json",
                 remove_capitals: bool = True):
        
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.remove_capitals = remove_capitals
        
        nltk.download('stopwords', quiet=True)
        self.stopwords = set(stopwords.words('english')) 

        self.load_or_build_index()

    def preprocess_text(self, text: str) -> str:
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        if self.remove_capitals:
            text = text.lower()
        return ' '.join(word for word in text.split() if word.lower() not in self.stopwords)

    def build_index(self):
        self.index = {}
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)


        for i, index in tqdm(enumerate(self.metadata)):
            paper = self.metadata[index]
            for keyword in paper['keyword_search']:
                term = ' '.join(word for word in keyword.lower().split() if word.lower() not in stopwords)
                if term not in self.index:
                    self.index[term] = []
                
                self.index[term].append(paper['arxiv_id'])

    def load_index(self):
        print("Loading existing index...")
        with open(self.index_path, 'rb') as f:
            self.index = json.load(f)
        
        print("Index loaded successfully.")

    def load_or_build_index(self):
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.build_index()

    def segment_keywords(self):
        pass

    def retrieve(self, query: str, arxiv_id: str, top_k: int = 10, return_scores = False):
        query_date = self.parse_date(arxiv_id)
        processed_query = self.preprocess_text(query)

        keyword = "dark matter"
        
        return self.index[keyword]

def main():
    retrieval_system = KeywordRetrievalSystem(remove_capitals = True)
    evaluate_main(retrieval_system, "ADSKeywords")

if __name__ == "__main__":
    main()