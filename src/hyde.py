import anthropic
from datasets import load_dataset
import sys
import requests
import numpy as np
sys.path.append('../evaluation')
from typing import List
from evaluate import RetrievalSystem, main as evaluate_main
import yaml
import json
import transformers
from sklearn.metrics.pairwise import cosine_similarity

class HydeRetrievalSystem(RetrievalSystem):
    def __init__(self, config_path: str, db_path: str, generation_model: str = "claude-3-5-sonnet-20240620",
                 embedding_model: str = "text-embedding-3-small", 
                 temperature: float = 0.5, max_doclen: int = 500, generate_n: int = 1, embed_query = True):
        
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.embeddings = None
        self.document_ids = []

        # HYPERPARAMETERS
        self.temperature = temperature
        self.max_doclen = max_doclen
        self.generate_n = generate_n
        self.embed_query = embed_query

        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
            self.anthropic_key = config['anthropic_api_key']
            self.openai_key = config['openai_api_key']
        
        self.client = anthropic.Anthropic(api_key = self.anthropic_key)
        
        self.load_embeddings()
    
    def load_embeddings(self):

        pass

    def retrieve(self, query: str, top_k: int = 10):
        docs = self.generate_docs(query)
        
        doc_embeddings = self.embed_docs(docs)
        if self.embed_query: 
            query_emb = self.embed_docs([query])[0]
            doc_embeddings.append(query_emb)
        
        embedding = np.mean(np.array(doc_embeddings), axis = 0).reshape(1, -1)

        similarities = cosine_similarity(embedding, self.embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [self.document_ids[i] for i in top_indices]

    def generate_docs(self, query: str):
        docs = []
        for i in range(self.generate_n):
            message = self.client.messages.create(
                model = self.generation_model,
                max_tokens = self.max_doclen,
                temperature = self.temperature,
                system = """You are an expert astronomer. Given a scientific query, generate the abstract and conclusion of an expert-level research paper
                            that answers the question. Stick to a maximum length of {} tokens and return just the text of the abstract and conclusion.
                            Do not include labels for any section. Use research-specific jargon.""".format(self.max_doclen),
                
                messages=[{ "role": "user",
                        "content": [{"type": "text", "text": query,}] }]
            )
            docs.append(message.content[0].text)

        return docs

    def embed_docs(self, docs: List[str]):
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(self.openai_key)
        }

        vecs = []
        for i in range(len(docs)):
            data = {
                "input": docs[i],
                "model": self.embedding_model,
            }
            response = requests.post(url, headers=headers, data=json.dumps(data))
            vecs.append(response.json()['data'][0]['embedding'])
        
        return vecs

def main():
    retrieval_system = HydeRetrievalSystem()
    evaluate_main(retrieval_system)

if __name__ == "__main__":
    main()