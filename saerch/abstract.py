import numpy as np
import pickle
import json
from typing import List
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    abstract: str
    conclusions: str
    arxiv_id: str
    title: str = None
    score: float = None
    n_citation: int = None
    keywords: List[str] = None

# Load the full embeddings matrix
embeddings = np.load('../data/vector_store/embeddings_matrix.npy')

# Load the index mapping
with open('../data/vector_store/index_mapping.pkl', 'rb') as f:
    index_mapping = pickle.load(f)

# Load the documents
with open('../data/vector_store/documents.pkl', 'rb') as f:
    documents = pickle.load(f)

# Create lists to store the abstract embeddings and corresponding text
abstract_embeddings = []
abstract_texts = []
doc_ids = []

# Iterate through the index mapping to find abstract indices and corresponding text
for doc_id, mappings in index_mapping.items():
    if 'abstract' in mappings:
        abstract_index = mappings['abstract']
        abstract_embeddings.append(embeddings[abstract_index])
        
        # Find the corresponding document and extract the abstract text
        doc = next((d for d in documents if d.id == doc_id), None)
        if doc:
            abstract_texts.append(doc.abstract)
            doc_ids.append(doc_id)
        else:
            print(f"Warning: Document with ID {doc_id} not found.")

# Convert lists to numpy arrays
abstract_embeddings = np.array(abstract_embeddings)

print(f"Processed {len(abstract_embeddings)} abstract embeddings.")
print(f"Shape of abstract embeddings: {abstract_embeddings.shape}")
print(f"Length of abstract texts: {len(abstract_texts)}")
print(f"Length of document IDs: {len(doc_ids)}")

# Save the abstract embeddings
np.save('../data/vector_store/abstract_embeddings.npy', abstract_embeddings)

# Save the abstract texts and document IDs
with open('../data/vector_store/abstract_texts.json', 'w') as f:
    json.dump({
        'doc_ids': doc_ids,
        'abstracts': abstract_texts
    }, f)

print("Saved abstract embeddings and texts.")
