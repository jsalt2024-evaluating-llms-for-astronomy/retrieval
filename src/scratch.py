# import os
# import pickle
# import numpy as np
# from typing import List, Dict, Any, Tuple
# from dataclasses import dataclass
# from tqdm import tqdm
# from openai import OpenAI
# import yaml
# import time
# import tiktoken
# import feedparser
# import urllib.request
# import re

# # Configuration
# DATA_DIR = "../data/vector_store"
# EMBEDDING_MODEL = "text-embedding-3-small"
# MAX_TOKENS = 8192
# MAX_RETRIES = 5
# RETRY_DELAY = 20
# MIN_CONTENT_LENGTH = 100 
# SAVE_INTERVAL = 25000

# @dataclass
# class Document:
#     id: str
#     abstract: str
#     conclusions: str
#     arxiv_id: str

# class EmbeddingClient:
#     def __init__(self, client: OpenAI, model: str = EMBEDDING_MODEL):
#         self.client = client
#         self.model = model
#         self.tokenizer = tiktoken.get_encoding("cl100k_base")

#     def truncate_text(self, text: str) -> str:
#         tokens = self.tokenizer.encode(text)
#         if len(tokens) > MAX_TOKENS:
#             return self.tokenizer.decode(tokens[:MAX_TOKENS])
#         return text

#     def embed(self, text: str) -> np.ndarray:
#         text = self.truncate_text(text.replace("\n", " "))
#         for attempt in range(MAX_RETRIES):
#             try:
#                 embedding = self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
#                 return np.array(embedding, dtype=np.float32)
#             except Exception as e:
#                 if attempt == MAX_RETRIES - 1:
#                     print(f"Failed to get embedding after {MAX_RETRIES} attempts: {e}")
#                     return np.zeros(1536, dtype=np.float32)
#                 print(f"Error getting embedding (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {RETRY_DELAY} seconds.")
#                 time.sleep(RETRY_DELAY)

# def load_data():
#     with open(os.path.join(DATA_DIR, "documents.pkl"), 'rb') as f:
#         documents = pickle.load(f)
#     with open(os.path.join(DATA_DIR, "document_index.pkl"), 'rb') as f:
#         document_index = pickle.load(f)
#     embeddings_matrix = np.load(os.path.join(DATA_DIR, "embeddings_matrix.npy"))
#     with open(os.path.join(DATA_DIR, "index_mapping.pkl"), 'rb') as f:
#         index_mapping = pickle.load(f)
#     return documents, document_index, embeddings_matrix, index_mapping

# def save_data(documents, document_index, embeddings_matrix, index_mapping):
#     with open(os.path.join(DATA_DIR, "documents.pkl"), 'wb') as f:
#         pickle.dump(documents, f)
#     with open(os.path.join(DATA_DIR, "document_index.pkl"), 'wb') as f:
#         pickle.dump(document_index, f)
#     np.save(os.path.join(DATA_DIR, "embeddings_matrix.npy"), embeddings_matrix)
#     with open(os.path.join(DATA_DIR, "index_mapping.pkl"), 'wb') as f:
#         pickle.dump(index_mapping, f)

# def insert_slash(s):
#     if re.search(r'[^0-9.]', s) or '.' in s:
#         for i in range(len(s)-1, -1, -1):
#             if not s[i].isdigit() and s[i] != '.':
#                 return s[:i+1] + '/' + s[i+1:]
#     return s

# def get_abstract_from_arxiv(arxiv_id):
#     arxiv_id = arxiv_id.replace('_arXiv.txt', '')
#     arxiv_id = insert_slash(arxiv_id)
#     query = arxiv_id
#     url = 'http://export.arxiv.org/api/query?search_query='+query+'&start=0&max_results=1'
#     data = urllib.request.urlopen(url)
#     data = data.read().decode('utf-8')
#     feed = feedparser.parse(data)
#     if len(feed['entries']) == 0:
#         return None
#     return feed['entries'][0]['summary']

# def fix_zero_embeddings(documents, document_index, embeddings_matrix, index_mapping, embedding_client):
#     fixed_count = 0
#     processed_count = 0
    
#     for doc_id, mappings in tqdm(index_mapping.items(), desc="Fixing zero embeddings"):
#         processed_count += 1
#         doc = document_index[doc_id]
        
#         # Check and fix abstract embedding
#         if 'abstract' in mappings:
#             idx = mappings['abstract']
#             if np.all(embeddings_matrix[idx] == 0) or len(doc.abstract) < MIN_CONTENT_LENGTH:
#                 abstract = doc.abstract if len(doc.abstract) >= MIN_CONTENT_LENGTH else None
#                 if not abstract:
#                     abstract = get_abstract_from_arxiv(doc.arxiv_id)
#                 if abstract and len(abstract) >= MIN_CONTENT_LENGTH:
#                     new_embedding = embedding_client.embed(abstract)
#                     embeddings_matrix[idx] = new_embedding
#                     doc.abstract = abstract  # Update the document with the new abstract
#                     document_index[doc_id] = doc  # Update the document in the index
#                     fixed_count += 1
#                     print(f"\nFixed zero/short embedding for abstract:")
#                     print(f"Document ID: {doc_id}")
#                     print(f"Index: {idx}")
#                     print(f"Abstract: {abstract[:200]}...")
#                     print(f"First 5 dimensions of new embedding: {new_embedding[:5]}")
#                 else:
#                     print(f"\nWarning: Could not find valid abstract for document {doc_id}")
        
#         # Check and fix conclusion embedding
#         if 'conclusions' in mappings:
#             idx = mappings['conclusions']
#             if np.all(embeddings_matrix[idx] == 0) and len(doc.conclusions) >= MIN_CONTENT_LENGTH:
#                 new_embedding = embedding_client.embed(doc.conclusions)
#                 embeddings_matrix[idx] = new_embedding
#                 fixed_count += 1
#                 print(f"\nFixed zero embedding for conclusion:")
#                 print(f"Document ID: {doc_id}")
#                 print(f"Index: {idx}")
#                 print(f"Conclusion: {doc.conclusions[:200]}...")
#                 print(f"First 5 dimensions of new embedding: {new_embedding[:5]}")
        
#         # Save progress periodically
#         if processed_count % SAVE_INTERVAL == 0 and processed_count > 100_000:
#             print(f"\nSaving progress after processing {processed_count} documents...")
#             save_data(documents, document_index, embeddings_matrix, index_mapping)
#             print("Progress saved.")
    
#     print(f"\nFixed {fixed_count} zero/short embeddings")
#     return embeddings_matrix, documents, document_index

# def main():
#     # Load configuration
#     config = yaml.safe_load(open('../config.yaml', 'r'))
#     client = OpenAI(api_key=config['openai_api_key'])
#     embedding_client = EmbeddingClient(client)

#     # Load data
#     print("Loading data...")
#     documents, document_index, embeddings_matrix, index_mapping = load_data()

#     # Fix zero embeddings
#     print("Fixing zero/short embeddings...")
#     embeddings_matrix, documents, document_index = fix_zero_embeddings(documents, document_index, embeddings_matrix, index_mapping, embedding_client)

#     # Save final updated data
#     print("Saving final updated data...")
#     save_data(documents, document_index, embeddings_matrix, index_mapping)

#     print("Zero/short embeddings fixed successfully!")

# if __name__ == "__main__":
#     main()

import os
from huggingface_hub import HfApi, create_repo

# Replace these with your Hugging Face details and local directory
hf_token = "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"
hf_repo_id = "JSALT2024-Astro-LLMs/astro-embeddings"
local_directory = "../data/vector_store"

# Initialize the Hugging Face API
api = HfApi()

# Create the new repository
create_repo(repo_id=hf_repo_id, token=hf_token, exist_ok=True)

# Function to upload a file or directory
def upload_to_hf(local_path, repo_path):
    if os.path.isfile(local_path):
        print(f"Uploading file: {local_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=hf_repo_id,
            token=hf_token
        )
    elif os.path.isdir(local_path):
        for item in os.listdir(local_path):
            item_local_path = os.path.join(local_path, item)
            item_repo_path = os.path.join(repo_path, item)
            upload_to_hf(item_local_path, item_repo_path)

# Start the upload process
print(f"Starting upload of {local_directory} to {hf_repo_id}")
upload_to_hf(local_directory, "")

print("Upload completed successfully!")