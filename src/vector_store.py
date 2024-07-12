import pickle
import numpy as np
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
import yaml
import time
from dataclasses import dataclass, asdict, field

class EmbeddingClient:
    def __init__(self, client: OpenAI, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model

    def embed(self, text: str) -> np.ndarray:
        embedding = self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
        return np.array(embedding, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = self.client.embeddings.create(input=texts, model=self.model).data
        return [np.array(embedding.embedding, dtype=np.float32) for embedding in embeddings]

@dataclass
class Document:
    id: str
    abstract: str
    conclusions: str
    arxiv_id: str = field(default="")
    title: str = None
    score: float = None
    n_citation: int = None
    keywords: List[str] = None

class DocumentLoader:
    def __init__(self, document_path: str = "charlieoneill/jsalt-astroph-dataset"):
        self.document_path = document_path

    def load_documents(self) -> List[Document]:
        documents = []
        dataset = load_dataset(self.document_path, split="train")
        for paper in dataset:
            documents.append(Document(
                id=paper['arxiv_id'],
                abstract=paper['abstract'],
                conclusions=paper['conclusions'],
                arxiv_id=paper['arxiv_id'],
            ))
        return documents

class VectorStore:
    def __init__(self, embeddings_path: str, document_loader: DocumentLoader, embedding_client: EmbeddingClient, data_dir: str = "../data/vector_store"):
        self.embedding_client = embedding_client
        self.data_dir = data_dir
        self.documents_path = os.path.join(data_dir, "documents.pkl")
        self.document_index_path = os.path.join(data_dir, "document_index.pkl")
        self.embeddings_matrix_path = os.path.join(data_dir, "embeddings_matrix.npy")
        self.index_mapping_path = os.path.join(data_dir, "index_mapping.pkl")

        if self._check_saved_data():
            self._load_saved_data()
        else:
            self.documents = document_loader.load_documents()
            self.document_index = self._build_document_index()
            self.embeddings_matrix, self.index_mapping = self._process_embeddings(embeddings_path)
            self._save_data()

    def _check_saved_data(self) -> bool:
        return all(os.path.exists(path) for path in [
            self.documents_path, self.document_index_path,
            self.embeddings_matrix_path, self.index_mapping_path
        ])

    def _load_saved_data(self):
        with open(self.documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        with open(self.document_index_path, 'rb') as f:
            self.document_index = pickle.load(f)
        self.embeddings_matrix = np.load(self.embeddings_matrix_path)
        with open(self.index_mapping_path, 'rb') as f:
            self.index_mapping = pickle.load(f)

    def _save_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        with open(self.document_index_path, 'wb') as f:
            pickle.dump(self.document_index, f)
        np.save(self.embeddings_matrix_path, self.embeddings_matrix)
        with open(self.index_mapping_path, 'wb') as f:
            pickle.dump(self.index_mapping, f)

    def _build_document_index(self) -> Dict[str, Document]:
        return {doc.id: doc for doc in self.documents}

    def _process_embeddings(self, path: str) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
        with open(path, 'rb') as f:
            embeddings_dict = pickle.load(f)

        embeddings = []
        index_mapping = {}
        current_index = 0
        total_nans = 0

        config = yaml.safe_load(open('../config.yaml', 'r'))

        document_loader = DocumentLoader("charlieoneill/jsalt-astroph-dataset")
        embedding_client = EmbeddingClient(OpenAI(api_key=config['openai_api_key']))

        for doc_id, doc_embeddings in tqdm(embeddings_dict.items(), desc="Processing embeddings"):
            if 'abstract' in doc_embeddings:
                if doc_embeddings['abstract'] is not None:
                    embeddings.append(list(doc_embeddings['abstract']))
                else:
                    print(f"NaN embedding for document {doc_id}")
                    print(self.document_index[doc_id].abstract)
                    embedding = embedding_client.embed(self.document_index[doc_id].abstract)
                    embeddings.append(embedding)
                    #embeddings.append(np.zeros(1536))
                    total_nans += 1
                index_mapping[doc_id] = {'abstract': current_index}
                current_index += 1
            if 'conclusions' in doc_embeddings:
                if doc_embeddings['conclusions'] is not None:
                    embeddings.append(list(doc_embeddings['conclusions']))
                else:
                    print(f"NaN embedding for document {doc_id}")
                    print(self.document_index[doc_id].conclusions)
                    embedding = embedding_client.embed(self.document_index[doc_id].conclusions)
                    embeddings.append(embedding)
                    #embeddings.append(np.zeros(1536))
                    total_nans += 1
                index_mapping[doc_id]['conclusions'] = current_index
                current_index += 1

        print(f"Total NaNs: {total_nans}")

        # Print total embeddings
        print(f"Total embeddings: {len(embeddings)}")

        return np.array(embeddings), index_mapping

    def search(self, query: str, k: int = 5, search_type: str = 'both') -> List[Dict[str, Any]]:
        query_embedding = np.array(self.embedding_client.embed(query), dtype=np.float32)
        print(f"Query embedding: {query_embedding}")
        
        start = time.time()
        # Print size of vector matrix multiply
        print(f"Embeddings matrix shape: {self.embeddings_matrix.shape}")
        print(f"Query embedding shape: {query_embedding.shape}")
        similarities = np.dot(self.embeddings_matrix, query_embedding)
        print(f"Similarities time: {time.time() - start:.4f}")
        
        top_k_indices = np.argsort(similarities)[-k*2:][::-1]
        
        results = []
        seen_docs = set()
        for idx in top_k_indices:
            for doc_id, mappings in self.index_mapping.items():
                if idx in mappings.values():
                    if doc_id in seen_docs:
                        continue
                    seen_docs.add(doc_id)
                    doc = self.document_index[doc_id]
                    embed_type = 'abstract' if mappings['abstract'] == idx else 'conclusions'
                    if search_type != 'both' and embed_type != search_type:
                        continue
                    results.append({
                        'id': doc_id,
                        'similarity': similarities[idx],
                        'abstract': doc.abstract,
                        'conclusions': doc.conclusions,
                        'matched_on': embed_type
                    })
                    if len(results) == k:
                        return results

        return results

# Initialize everything
# config = yaml.safe_load(open('../config.yaml', 'r'))

# document_loader = DocumentLoader("charlieoneill/jsalt-astroph-dataset")
# embedding_client = EmbeddingClient(OpenAI(api_key=config['openai_api_key']))
# vector_store = VectorStore("../data/embeddings/embeddings_final.pkl", document_loader, embedding_client)

# query = "What mechanisms could potentially drive dynamo operation in M giant stars and how can we differentiate between different types of dynamos based on observational data?"
# results = vector_store.search(query, k=10)
# for result in results:
#     print(f"Document ID: {result['id']}")
#     print(f"Similarity: {result['similarity']}")
#     print(f"Matched on: {result['matched_on']}")
#     if result['matched_on'] == 'abstract':
#         print(f"Abstract: {result['abstract'][:1000]}...")
#     else:
#         print(f"Conclusions: {result['conclusions'][:1000]}...")
#     print("---")