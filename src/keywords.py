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
import pytextrank
nltk.download('stopwords', quiet=True) 
nlp = spacy.load("en_core_web_sm")
#spacy.cli.download('en_core_web_sm')
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

    def parse_doc(self, text, nret = 10):
        #text = ' '.join(word for word in text.split() if word.lower() not in self.stopwords)
        local_kws = []
        doc = nlp(text)
        # examine the top-ranked phrases in the document
        for phrase in doc._.phrases[:nret]:
            # print(phrase.text)
            local_kws.append(phrase.text.lower())
        
        return [self.preprocess_text(word) for word in local_kws]

    def get_propn(self, text):
        result = []
        doc = nlp(text) 

        working_str = ''
        for token in doc:
            if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
                if working_str != '':
                    result.append(working_str.strip())
                    working_str = ''

            if(token.pos_ == "PROPN"):
                working_str += token.text
                working_str += ' '

        if working_str != '': result.append(working_str.strip())
        
        return [self.preprocess_text(word) for word in result]

    def keyword_filter(self, query: str, verbose = False, ne_only = True):
        query_keywords = self.parse_doc(query)
        nouns = self.get_propn(query)
        if verbose: print('keywords:', query_keywords)
        if verbose: print('proper nouns:', nouns)

        filtered = set()
        if len(query_keywords) > 0 and not ne_only:
            for keyword in query_keywords:
                if keyword != '' and keyword in self.index.keys(): filtered |= set(self.index[keyword])
        
        if len(nouns) > 0:
            ne_results = set()
            for noun in nouns:
                if noun in self.index.keys(): ne_results |= set(self.index[noun])
            if ne_only: return ne_results
            
            filtered &= ne_results
        return filtered
    
    def date_filter(self, results, arxiv_id):
        query_date = self.parse_date(arxiv_id)
        filtered = set()
        for doc in results:
            if self.parse_date(doc) >= query_date:
                filtered.add(doc)
        
        return filtered


    def retrieve(self, query: str, arxiv_id: str, top_k: int = 10, verbose = True, ne_only = False) -> List[str]:
        filtered = self.date_filter(self.keyword_filter(query, verbose, ne_only = ne_only), arxiv_id)
        if verbose: print('Retrieved documents:', len(filtered))
        return list(filtered)[:top_k]

def main():
    retrieval_system = KeywordRetrievalSystem(remove_capitals = True)
    evaluate_main(retrieval_system, "ADSKeywords")

if __name__ == "__main__":
    main()