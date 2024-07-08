import json
from typing import List, Dict, Tuple
from collections import Counter
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime, date
import sys
sys.path.append('../evaluation')
import spacy
from collections import Counter
from string import punctuation
from nltk.corpus import stopwords
import nltk
import math
import pytextrank
import ast
nltk.download('stopwords', quiet=True) 
nlp = spacy.load("en_core_web_sm")
#spacy.cli.download('en_core_web_sm')
nlp.add_pipe("textrank")

class Filter():    
    def filter(self, query: str, arxiv_id: str) -> List[str]:
        pass

class CitationFilter(Filter): # can do it with all metadata
    def __init__(self, metadata):
        self.metadata = metadata
        self.citation_counts = {doc_id: self.metadata[doc_id]['citation_count'] for doc_id in self.metadata}
    
    def citation_weight(self, x, shift, scale):
        return 1 / (1 + np.exp(-1 * (x - shift) / scale)) # sigmoid function
    
    def filter(self, doc_scores, weight = 0.1): # additive weighting
        citation_count = np.array([self.citation_counts[doc[0]] for doc in doc_scores])
        cmean, cstd = np.median(citation_count), np.std(citation_count)
        citation_score = self.citation_weight(citation_count, cmean, cstd)

        for i, doc in enumerate(doc_scores):
            doc_scores[i][2] += weight * citation_score[i]

class DateFilter(Filter): # include time weighting eventually
    def __init__(self, document_dates):
        self.document_dates = document_dates

    def parse_date(self, arxiv_id: str) -> datetime: # only for documents
        if arxiv_id.startswith('astro-ph'):
            arxiv_id = arxiv_id.split('astro-ph')[1].split('_arXiv')[0]
        try:
            year = int("20" + arxiv_id[:2])
            month = int(arxiv_id[2:4])
        except:
            year = 2023
            month = 1
        return date(year, month, 1)
    
    def weight(self, time, shift, scale):
        return 1 / (1 + math.exp((time - shift) / scale))

    def evaluate_filter(self, year, filter_string):
        try:
            # Use ast.literal_eval to safely evaluate the expression
            result = eval(filter_string, {"__builtins__": None}, {"year": year})
            return result
        except Exception as e:
            print(f"Error evaluating filter: {e}")
            return False
                
    def filter(self, docs, boolean_date = None, min_date = None, max_date = None, time_score = 0):
        filtered = []

        if boolean_date is not None:
            boolean_date = boolean_date.replace("AND", "and").replace("OR", "or")
            for doc in docs:
                if self.evaluate_filter(self.document_dates[doc[0]].year, boolean_date):
                    filtered.append(doc)
        
        else:
            if min_date == None: min_date = date(1990, 1, 1)
            if max_date == None: max_date = date(2024, 7, 3)

            for doc in docs:
                if self.document_dates[doc[0]] >= min_date and self.document_dates[doc[0]] <= max_date:
                    filtered.append(doc)
        
        if time_score is not None: # apply time weighting
            for i, item in enumerate(filtered):
                time_diff = (max_date - self.document_dates[filtered[i][0]]).days / 365
                filtered[i][2] += time_score * 0.1 * self.weight(time_diff, 5, 5)

        return filtered

class KeywordFilter(Filter):
    def __init__(self, index_path: str = "../data/vector_store/keyword_index.json", 
                 remove_capitals: bool = True, metadata = None, ne_only = True, verbose = False):
        
        self.index_path = index_path
        self.metadata = metadata
        self.remove_capitals = remove_capitals
        self.ne_only = ne_only
        self.stopwords = set(stopwords.words('english')) 
        self.verbose = verbose
        self.index = None

        self.load_or_build_index()

    def preprocess_text(self, text: str) -> str:
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        if self.remove_capitals: text = text.lower()
        return ' '.join(word for word in text.split() if word.lower() not in self.stopwords)

    def build_index(self): # include the title in the index
        print("Building index...")
        self.index = {}

        for i, index in tqdm(enumerate(self.metadata)):
            paper = self.metadata[index]
            title = paper['title'][0]
            title_keywords = set() #set(self.parse_doc(title) + self.get_propn(title))
            for keyword in set(paper['keyword_search']) | title_keywords:
                term = ' '.join(word for word in keyword.lower().split() if word.lower() not in self.stopwords)
                if term not in self.index:
                    self.index[term] = []
                
                self.index[term].append(paper['arxiv_id'])
        
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f)

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

    def parse_doc(self, doc):
        local_kws = []
        
        for phrase in doc._.phrases:
            local_kws.append(phrase.text.lower())
        
        return [self.preprocess_text(word) for word in local_kws]

    def get_propn(self, doc):
        result = []

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

    def filter(self, query: str, doc_ids = None):
        doc = nlp(query)
        query_keywords = self.parse_doc(doc)
        nouns = self.get_propn(doc)
        if self.verbose: print('keywords:', query_keywords)
        if self.verbose: print('proper nouns:', nouns)

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