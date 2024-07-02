import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from openai import OpenAI
import yaml
import time
import tiktoken

# Configuration
DATA_DIR = "../data/vector_store"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS = 8192
MAX_RETRIES = 5
RETRY_DELAY = 20

@dataclass
class Document:
    id: str
    abstract: str
    conclusions: str
    arxiv_id: str

class EmbeddingClient:
    def __init__(self, client: OpenAI, model: str = EMBEDDING_MODEL):
        self.client = client
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def truncate_text(self, text: str) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) > MAX_TOKENS:
            return self.tokenizer.decode(tokens[:MAX_TOKENS])
        return text

    def embed(self, text: str) -> np.ndarray:
        text = self.truncate_text(text.replace("\n", " "))
        for attempt in range(MAX_RETRIES):
            try:
                embedding = self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
                return np.array(embedding, dtype=np.float32)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to get embedding after {MAX_RETRIES} attempts: {e}")
                    return np.zeros(1536, dtype=np.float32)
                print(f"Error getting embedding (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {RETRY_DELAY} seconds.")
                time.sleep(RETRY_DELAY)

def load_data():
    with open(os.path.join(DATA_DIR, "documents.pkl"), 'rb') as f:
        documents = pickle.load(f)
    with open(os.path.join(DATA_DIR, "document_index.pkl"), 'rb') as f:
        document_index = pickle.load(f)
    embeddings_matrix = np.load(os.path.join(DATA_DIR, "embeddings_matrix.npy"))
    with open(os.path.join(DATA_DIR, "index_mapping.pkl"), 'rb') as f:
        index_mapping = pickle.load(f)
    return documents, document_index, embeddings_matrix, index_mapping

def save_data(documents, document_index, embeddings_matrix, index_mapping):
    with open(os.path.join(DATA_DIR, "documents.pkl"), 'wb') as f:
        pickle.dump(documents, f)
    with open(os.path.join(DATA_DIR, "document_index.pkl"), 'wb') as f:
        pickle.dump(document_index, f)
    np.save(os.path.join(DATA_DIR, "embeddings_matrix.npy"), embeddings_matrix)
    with open(os.path.join(DATA_DIR, "index_mapping.pkl"), 'wb') as f:
        pickle.dump(index_mapping, f)

def fix_zero_embeddings(documents, document_index, embeddings_matrix, index_mapping, embedding_client):
    fixed_count = 0
    for doc_id, mappings in tqdm(index_mapping.items(), desc="Fixing zero embeddings"):
        doc = document_index[doc_id]
        
        # Check and fix abstract embedding
        if 'abstract' in mappings:
            idx = mappings['abstract']
            if np.all(embeddings_matrix[idx] == 0):
                new_embedding = embedding_client.embed(doc.abstract)
                embeddings_matrix[idx] = new_embedding
                fixed_count += 1
                print(f"\nFixed zero embedding for abstract:")
                print(f"Document ID: {doc_id}")
                print(f"Index: {idx}")
                print(f"Abstract: {doc.abstract[:200]}...")  # Print first 200 characters
                print(f"First 5 dimensions of new embedding: {new_embedding[:5]}")
        
        # Check and fix conclusions embedding
        if 'conclusions' in mappings:
            idx = mappings['conclusions']
            if np.all(embeddings_matrix[idx] == 0):
                new_embedding = embedding_client.embed(doc.conclusions)
                embeddings_matrix[idx] = new_embedding
                fixed_count += 1
                print(f"\nFixed zero embedding for conclusions:")
                print(f"Document ID: {doc_id}")
                print(f"Index: {idx}")
                print(f"Conclusions: {doc.conclusions[:200]}...")  # Print first 200 characters
                print(f"First 5 dimensions of new embedding: {new_embedding[:5]}")
    
    print(f"\nFixed {fixed_count} zero embeddings")
    return embeddings_matrix

def main():
    # Load configuration
    config = yaml.safe_load(open('../config.yaml', 'r'))
    client = OpenAI(api_key=config['openai_api_key'])
    embedding_client = EmbeddingClient(client)

    # Load data
    print("Loading data...")
    documents, document_index, embeddings_matrix, index_mapping = load_data()

    # Fix zero embeddings
    print("Fixing zero embeddings...")
    embeddings_matrix = fix_zero_embeddings(documents, document_index, embeddings_matrix, index_mapping, embedding_client)

    # Save updated data
    print("Saving updated data...")
    save_data(documents, document_index, embeddings_matrix, index_mapping)

    print("Zero embeddings fixed successfully!")

if __name__ == "__main__":
    main()