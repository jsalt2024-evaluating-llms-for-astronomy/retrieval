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
import concurrent.futures

# use generate_n = 0, embed_query = True to do basic vector search (no generation)
class HydeRetrievalSystem(RetrievalSystem):
    def __init__(self, config_path: str, db_path: str, generation_model: str = "claude-3-haiku-20240307",
                 vector_db = None, generation_client = None, embedding_model: str = "text-embedding-3-small", 
                 temperature: float = 0.5, max_doclen: int = 500, generate_n: int = 1, embed_query = True):
        
        if max_doclen * generate_n > 8191:
            raise ValueError("Too many tokens. Please reduce max_doclen or generate_n.")
        
        # For building chained retrieval systems -- avoid generating a ton of clients at once
        if vector_db is not None: self.embeddings = self.vector_db
        else:
            self.db_path = db_path
            self.embeddings = None
            self.load_embeddings()
        
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        
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
        
        if generation_client is not None: self.client = generation_client
        else: self.client = anthropic.Anthropic(api_key = self.anthropic_key)
        
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

    def generate_doc(self, query: str):
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

        return message.content[0].text
    
    def generate_docs(self, query: str):
        docs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_query = {executor.submit(self.generate_doc, query): query for i in range(self.generate_n)}
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    data = future.result()
                    docs.append(data)
                except Exception as exc:
                    print(f'Query {query} generated an exception: {exc}')
        return docs

    def embed_docs(self, docs: List[str]):
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(self.openai_key)
        }

        data = {
            "input": docs,
            "model": self.embedding_model
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        json_response = response.json()['data']
        
        vecs = [json_response[i]['embedding'] for i in range(len(docs))]
        return vecs

def main():
    retrieval_system = HydeRetrievalSystem()
    evaluate_main(retrieval_system)

if __name__ == "__main__":
    main()