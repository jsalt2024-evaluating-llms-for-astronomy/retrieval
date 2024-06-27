import os
import anthropic
from tqdm import tqdm
from datasets import load_dataset
import sys
import requests
sys.path.append('../evaluation')
from evaluate import RetrievalSystem, main as evaluate_main
import yaml
import json
import transformers
from sklearn.metrics.pairwise import cosine_similarity

class HydeRetrievalSystem(RetrievalSystem):
    def __init__(self, config_path: str, db_path: str, model: str = "text-embedding-3-small"):
        self.db_path = db_path
        self.model = model
        self.embeddings = None
        self.document_ids = []

        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
            self.anthropic_key = config['anthropic_api_key']
            self.openai_key = config['openai_api_key']
        
        self.client = anthropic.Anthropic(api_key = self.anthropic_key)
        
        self.load_embeddings()
    
    def load_embeddings(self):

        pass

    def retrieve(self, query: str, top_k: int = 10):
        doc = self.generate_doc(query)
        
        similarities = cosine_similarity(doc, self.embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [self.document_ids[i] for i in top_indices]

    def generate_doc(self, query: str, max_doclen: int = 500):
        
        message = self.client.messages.create(
            model = "claude-3-5-sonnet-20240620",
            max_tokens = max_doclen,
            temperature = 0,
            system = """You are an expert astronomer. Given a scientific query, generate the abstract and conclusion of an expert-level research paper
                        that answers the question. Stick to a maximum length of {} and return just the text of the abstract and conclusion.
                        Do not include labels for any section. Use research-specific jargon.""".format(max_doclen),
            
            messages=[{"role": "user",
                    "content": [{"type": "text", "text": query,}] }]
        )

        return message.content[0].text

    def embed_doc(self, doc: str):
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(self.openai_key)
        }
        data = {
            "input": doc,
            "model": self.model,
        }

        response = requests.post(url, headers = headers, data = json.dumps(data))
        return response.json()['data'][0]['embedding']

def main():
    retrieval_system = HydeRetrievalSystem()
    evaluate_main(retrieval_system)

if __name__ == "__main__":
    main()