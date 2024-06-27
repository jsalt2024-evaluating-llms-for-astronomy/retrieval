import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from collections import defaultdict
import wandb
import numpy as np
from tqdm import tqdm

class RetrievalSystem(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Retrieve relevant documents based on the given query.
        
        Args:
            query (str): The input query.
            top_k (int): Number of top documents to retrieve.
        
        Returns:
            List[str]: List of document IDs (folder_file) of the retrieved documents.
        """
        pass

class Evaluator:
    def __init__(self, retrieval_system: RetrievalSystem, system_name: str, wandb_log: bool = False):
        self.retrieval_system = retrieval_system
        self.system_name = system_name
        self.wandb_log = wandb_log

    def evaluate(self, single_doc_file: str, multi_doc_file: str, top_k: int = 10) -> Dict[str, Dict[str, float]]:
        if self.wandb_log:
            wandb.init(project="jsalt-astro", name=f"{self.system_name}")

        results = {}
        
        #single_results = self._evaluate_single_document(single_doc_file, top_k)
        results['single_doc'] = None #single_results

        multi_results = self._evaluate_multi_document(multi_doc_file, top_k)
        results['multi_doc'] = multi_results
        
        log_dict = {}
        for eval_type, metrics in results.items():
            for metric, value in metrics.items():
                log_dict[f"{eval_type}_{metric}"] = value
        
        if self.wandb_log:
            wandb.log(log_dict)
            wandb.finish()
        
        return results

    def _evaluate_single_document(self, ground_truth_file: str, top_k: int = 10) -> Dict[str, float]:
        ground_truth = self._load_ground_truth(ground_truth_file)
        results = defaultdict(list)
        
        total_queries = len(ground_truth) * 2
        with tqdm(total=total_queries, desc="Single-doc progress") as pbar:
            for doc_id, data in ground_truth.items():
                for question_type in ['intro', 'conclusion']:
                    query = data[f'question_{question_type}']
                    retrieved_docs = self.retrieval_system.retrieve(query, top_k=top_k)
                    
                    results['success_rate'].append(int(doc_id in retrieved_docs))
                    results['reciprocal_rank'].append(self._calculate_reciprocal_rank(retrieved_docs, doc_id))
                    results['avg_precision'].append(self._calculate_avg_precision(retrieved_docs, doc_id))

                    pbar.update(1)
        
        return {metric: sum(values) / len(values) for metric, values in results.items()}

    def _evaluate_multi_document(self, ground_truth_file: str, top_k: int = 25) -> Dict[str, float]:
        ground_truth = self._load_ground_truth(ground_truth_file)
        results = defaultdict(list)
        
        for item in tqdm(ground_truth, desc="Multi-doc progress"):
            item = item[list(item.keys())[0]]
            query = item['question']
            retrieved_docs = self.retrieval_system.retrieve(query, top_k=top_k)
            relevant_docs = self._map_arxiv_to_filenames(item['arxiv'])

            # Remove 'astro-ph' prefix from arxiv IDs
            retrieved_docs = [doc.replace('astro-ph', '') for doc in retrieved_docs]

            # Print both the retrieved and relevant documents
            print(f"\nQuery: {query}")
            print("Retrieved Documents:")
            for doc in retrieved_docs:
                print(f"  {doc}")
            print("Relevant Documents:")
            for doc in relevant_docs:
                print(f"  {doc}")
            
            results['map'].append(self._calculate_map(retrieved_docs, relevant_docs))
            results['ndcg'].append(self._calculate_ndcg(retrieved_docs, relevant_docs))
            results['recall@k'].append(self._calculate_recall_at_k(retrieved_docs, relevant_docs, top_k))
            results['precision@k'].append(self._calculate_precision_at_k(retrieved_docs, relevant_docs, top_k))
            results['f1@k'].append(self._calculate_f1_at_k(retrieved_docs, relevant_docs, top_k))
            print(results)
        
        return {metric: np.mean(values) for metric, values in results.items()}

    @staticmethod
    def _load_ground_truth(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r') as f:
            return json.load(f)

    def _map_arxiv_to_filenames(self, arxiv_ids: List[str]) -> List[str]:
        #self.document_ids.append(f"{paper['subfolder']}/{paper['filename']}")
        ids_list = []
        for arxiv_id in arxiv_ids:
            if '.' in arxiv_id:
                print(arxiv_id)
                year_month = arxiv_id.split('.')[0]
                identifier = arxiv_id.split('.')[1].split('_')[0]
                ids_list.append(f"{year_month}/{year_month}.{identifier}_arXiv.txt")
            else:
                year_month = arxiv_id[:4]
                identifier = arxiv_id[4:]
                ids_list.append(f"{year_month}/{year_month}.{identifier}_arXiv")
        return ids_list
    
    @staticmethod
    def _calculate_reciprocal_rank(retrieved_docs: List[str], relevant_doc: str) -> float:
        try:
            rank = retrieved_docs.index(relevant_doc) + 1
            return 1 / rank
        except ValueError:
            return 0

    @staticmethod
    def _calculate_avg_precision(retrieved_docs: List[str], relevant_doc: str) -> float:
        if relevant_doc not in retrieved_docs:
            return 0
        rank = retrieved_docs.index(relevant_doc) + 1
        return 1 / rank

    def _calculate_map(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        relevance = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
        precisions = [sum(relevance[:i+1]) / (i+1) for i in range(len(relevance))]
        return sum(p * r for p, r in zip(precisions, relevance)) / len(relevant_docs)

    def _calculate_ndcg(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        relevance = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
        dcg = sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(relevance))
        idcg = sum((2**1 - 1) / np.log2(i+2) for i in range(min(len(relevant_docs), len(retrieved_docs))))
        return dcg / idcg if idcg > 0 else 0

    def _calculate_recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        return len(set(retrieved_docs[:k]) & set(relevant_docs)) / len(relevant_docs)

    def _calculate_precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        return len(set(retrieved_docs[:k]) & set(relevant_docs)) / k

    def _calculate_f1_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        precision = self._calculate_precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self._calculate_recall_at_k(retrieved_docs, relevant_docs, k)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def main(retrieval_system: RetrievalSystem, system_name: str):
    evaluator = Evaluator(retrieval_system, system_name)
    results = evaluator.evaluate('../data/single_paper.json', '../data/multi_paper.json', top_k=10)
    
    print("Evaluation Results:")
    for eval_type, metrics in results.items():
        print(f"\n{eval_type.replace('_', ' ').title()} Evaluation:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    pass
    # Example usage:
    # from bow_retrieval import BagOfWordsRetrievalSystem
    # retrieval_system = BagOfWordsRetrievalSystem("charlieoneill/jsalt-astroph-dataset")
    # main(retrieval_system, "BagOfWords")