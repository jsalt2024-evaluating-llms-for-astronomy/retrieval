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

class Filter():    
    # we can also build the weighting system directly into this
    def filter(self, query: str, arxiv_id: str) -> List[str]:
        pass

class CitationFilter(Filter): # can do it with all metadata

    def __init__(self, metadata):
        self.metadata = metadata
        self.citation_counts = {doc_id: self.metadata[doc_id]['citation_count'] for doc_id in self.metadata}
    
    def citation_weight(self, x, shift, scale):
        return 1 / (1 + np.exp(-1 * (x - shift) / scale)) # sigmoid function
    
    def filter(self, doc_scores, weight = 0.2): # additive weighting
        citation_count = np.array([self.citation_counts[doc[0]] for doc in doc_scores])
        cmean, cstd = np.mean(citation_count), np.std(citation_count)
        citation_score = self.citation_weight(citation_count, cmean, cstd)

        for i, doc in enumerate(doc_scores):
            doc_scores[i][2] *= weight * citation_score[i]

class DateFilter(Filter): # include time weighting eventually
    def __init__(self, document_dates):
        self.document_dates = document_dates

    def parse_date(self, arxiv_id: str) -> datetime:
        try:
            year = int("20" + arxiv_id[:2])
            month = int(arxiv_id[2:4])
        except:
            year = 2023
            month = 1
        return datetime(year, month, 1)
    
    def filter(self, query: str, arxiv_id: str, doc_ids: List[str]):
        query_date = self.parse_date(arxiv_id)
        filtered = set()
        for doc in doc_ids:
            if self.document_dates[doc] >= query_date:
                filtered.add(doc)
        
        return filtered

class KeywordFilter(Filter):
    def __init__(self, index_path: str = "../data/vector_store/keyword_index.json", metadata_path: str = "../data/vector_store/metadata.json",
                 remove_capitals: bool = True, metadata = None,  ne_only = True):
        
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.remove_capitals = remove_capitals
        self.ne_only = ne_only
        self.stopwords = set(stopwords.words('english')) 

        if metadata is None:
            self.load_or_build_index()
        else: self.metadata = metadata

    def preprocess_text(self, text: str) -> str:
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        if self.remove_capitals: text = text.lower()
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

    def parse_doc(self, text):
        local_kws = []
        doc = nlp(text)
        
        for phrase in doc._.phrases:
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
                working_str += token.text + ' '

        if working_str != '': result.append(working_str.strip())
        
        return [self.preprocess_text(word) for word in result]

    def filter(self, query: str, arxiv_id: str, doc_ids = None, verbose = False):
        query_keywords = self.parse_doc(query)
        nouns = self.get_propn(query)
        if verbose: print('keywords:', query_keywords)
        if verbose: print('proper nouns:', nouns)

        filtered = set()
        if len(query_keywords) > 0 and not self.ne_only:
            for keyword in query_keywords:
                if keyword != '' and keyword in self.index.keys(): filtered |= set(self.index[keyword])
        
        if len(nouns) > 0:
            ne_results = set()
            for noun in nouns:
                if noun in self.index.keys(): ne_results |= set(self.index[noun])
            
            if self.ne_only: filtered = ne_results # keep only named entity results
            else: filtered &= ne_results # take the intersection
        
        if doc_ids is not None: filtered &= doc_ids # apply filter to results
        return filtered