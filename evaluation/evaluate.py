import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from collections import defaultdict

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
    def __init__(self, retrieval_system: RetrievalSystem, ground_truth_file: str):
        self.retrieval_system = retrieval_system
        self.ground_truth = self._load_ground_truth(ground_truth_file)
    
    @staticmethod
    def _load_ground_truth(file_path: str) -> Dict[str, Dict[str, str]]:
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def evaluate(self, top_k: int = 10) -> Dict[str, float]:
        """
        Evaluate the retrieval system using various IR metrics.
        
        Args:
            top_k (int): Number of top documents to retrieve.
        
        Returns:
            Dict[str, float]: Dictionary containing various IR metrics.
        """
        results = defaultdict(list)
        
        for query, data in self.ground_truth.items():
            retrieved_docs = self.retrieval_system.retrieve(data['question_intro'], top_k=top_k)
            relevant_doc = query  # The folder_file is the key in ground truth
            
            precision = self._calculate_precision(retrieved_docs, relevant_doc)
            recall = self._calculate_recall(retrieved_docs, relevant_doc)
            f1_score = self._calculate_f1_score(precision, recall)
            mrr = self._calculate_mrr(retrieved_docs, relevant_doc)
            
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1_score)
            results['mrr'].append(mrr)
        
        return {metric: sum(values) / len(values) for metric, values in results.items()}
    
    @staticmethod
    def _calculate_precision(retrieved_docs: List[str], relevant_doc: str) -> float:
        return int(relevant_doc in retrieved_docs) / len(retrieved_docs)
    
    @staticmethod
    def _calculate_recall(retrieved_docs: List[str], relevant_doc: str) -> float:
        return int(relevant_doc in retrieved_docs)
    
    @staticmethod
    def _calculate_f1_score(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def _calculate_mrr(retrieved_docs: List[str], relevant_doc: str) -> float:
        try:
            rank = retrieved_docs.index(relevant_doc) + 1
            return 1 / rank
        except ValueError:
            return 0

def main(retrieval_system: RetrievalSystem):
    evaluator = Evaluator(retrieval_system, '../data/single_paper.json')
    results = evaluator.evaluate(top_k=10)
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    # This part will be commented out, as we'll import and use this script from the Bag of Words script
    pass
    # Example usage:
    # from bow_retrieval import BagOfWordsRetrievalSystem
    # retrieval_system = BagOfWordsRetrievalSystem("charlieoneill/jsalt-astroph-dataset")
    # main(retrieval_system)