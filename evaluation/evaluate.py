# import json
# from abc import ABC, abstractmethod
# from typing import List, Dict, Any
# from collections import defaultdict
# import wandb
# import numpy as np
# from tqdm import tqdm

# class RetrievalSystem(ABC):
#     @abstractmethod
#     def retrieve(self, query: str, arxiv_id: str, top_k: int = 100) -> List[str]:
#         pass

# class Evaluator:
#     def __init__(self, retrieval_system: RetrievalSystem, system_name: str, wandb_log: bool = True):
#         self.retrieval_system = retrieval_system
#         self.system_name = system_name
#         self.wandb_log = wandb_log

#     def evaluate(self, single_doc_file: str, multi_doc_file: str, multi_paper_sentences_file: str, k_values: List[int] = [10, 50, 100]) -> Dict[str, Dict[str, float]]:
#         if self.wandb_log:
#             wandb.init(project="jsalt-astro", name=f"{self.system_name}")

#         results = {}
        
#         single_results = self._evaluate_single_document(single_doc_file, 10)
#         results['single_doc'] = single_results
#         print("Single Document Results:", single_results)

#         multi_results = self._evaluate_multi_document(multi_doc_file, k_values)
#         results['multi_doc'] = multi_results
#         print("Multi Document Results:", multi_results)

#         multipaper_sentences_results = self._evaluate_multipaper_sentences(multi_paper_sentences_file, k_values)
#         results['multipaper_sentences'] = multipaper_sentences_results
#         print("Multipaper Sentences Results:", multipaper_sentences_results)
        
#         log_dict = {}
#         for eval_type, metrics in results.items():
#             if eval_type == 'single_doc':
#                 for metric, value in metrics.items():
#                     log_dict[f"single_doc_{metric}"] = value
#             else:  # multi_doc and multipaper_sentences
#                 for k, k_metrics in metrics.items():
#                     for metric, value in k_metrics.items():
#                         log_dict[f"{eval_type}_{metric}@{k}"] = value
        
#         if self.wandb_log:
#             wandb.log(log_dict)
#             wandb.finish()
        
#         return results

#     def _evaluate_single_document(self, ground_truth_file: str, top_k: int = 10) -> Dict[str, float]:
#         # This method remains unchanged
#         ground_truth = self._load_ground_truth(ground_truth_file)
#         results = defaultdict(list)
        
#         total_queries = sum(len(data) for data in ground_truth.values())
#         with tqdm(total=total_queries, desc="Single-doc progress") as pbar:
#             for arxiv_id, data in ground_truth.items():
#                 for question_type in ['question_abstract', 'question_conclusion']:
#                     if question_type in data:
#                         query = data[question_type]
#                         retrieved_docs = self.retrieval_system.retrieve(query, arxiv_id, top_k=top_k)
                        
#                         results['success_rate'].append(int(arxiv_id in retrieved_docs))
#                         results['reciprocal_rank'].append(self._calculate_reciprocal_rank(retrieved_docs, arxiv_id))
#                         results['avg_precision'].append(self._calculate_avg_precision(retrieved_docs, arxiv_id))

#                         pbar.update(1)
        
#         return {metric: sum(values) / len(values) for metric, values in results.items()}

#     def _evaluate_multi_document(self, ground_truth_file: str, k_values: List[int]) -> Dict[str, Dict[str, float]]:
#         ground_truth = self._load_ground_truth(ground_truth_file)
#         results = {k: defaultdict(list) for k in k_values}
        
#         for arxiv_id, item in tqdm(ground_truth.items(), desc="Multi-doc progress"):
#             arxiv_id_clean = arxiv_id.split('_')[0]
#             query = item['question']
#             retrieved_docs = self.retrieval_system.retrieve(query, arxiv_id_clean, top_k=max(k_values))
#             relevant_docs = item['arxiv']

#             for k in k_values:
#                 results[k]['map'].append(self._calculate_map(retrieved_docs[:k], relevant_docs))
#                 results[k]['ndcg'].append(self._calculate_ndcg(retrieved_docs[:k], relevant_docs))
#                 results[k]['recall'].append(self._calculate_recall_at_k(retrieved_docs[:k], relevant_docs, k))
        
#         return {k: {metric: np.mean(values) for metric, values in k_results.items()} for k, k_results in results.items()}

#     def _evaluate_multipaper_sentences(self, ground_truth_file: str, k_values: List[int]) -> Dict[str, Dict[str, float]]:
#         ground_truth = self._load_ground_truth(ground_truth_file)
#         results = {k: defaultdict(list) for k in k_values}
        
#         for item_id, item in tqdm(ground_truth.items(), desc="Multipaper sentences progress"):
#             arxiv_id_clean = item_id.split('_')[0]
#             query = item['question']
#             retrieved_docs = self.retrieval_system.retrieve(query, arxiv_id_clean, top_k=max(k_values))
#             relevant_docs = item['arxiv']

#             for k in k_values:
#                 results[k]['map'].append(self._calculate_map(retrieved_docs[:k], relevant_docs))
#                 results[k]['ndcg'].append(self._calculate_ndcg(retrieved_docs[:k], relevant_docs))
#                 results[k]['recall'].append(self._calculate_recall_at_k(retrieved_docs[:k], relevant_docs, k))
        
#         return {k: {metric: np.mean(values) for metric, values in k_results.items()} for k, k_results in results.items()}

#     @staticmethod
#     def _load_ground_truth(file_path: str) -> Dict[str, Any]:
#         with open(file_path, 'r') as f:
#             return json.load(f)

#     @staticmethod
#     def _calculate_reciprocal_rank(retrieved_docs: List[str], relevant_doc: str) -> float:
#         try:
#             rank = retrieved_docs.index(relevant_doc) + 1
#             return 1 / rank
#         except ValueError:
#             return 0

#     @staticmethod
#     def _calculate_avg_precision(retrieved_docs: List[str], relevant_doc: str) -> float:
#         if relevant_doc not in retrieved_docs:
#             return 0
#         rank = retrieved_docs.index(relevant_doc) + 1
#         return 1 / rank

#     def _calculate_map(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
#         relevance = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
#         precisions = [sum(relevance[:i+1]) / (i+1) for i in range(len(relevance))]
#         return sum(p * r for p, r in zip(precisions, relevance)) / len(relevant_docs)

#     def _calculate_ndcg(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
#         relevance = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
#         dcg = sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(relevance))
#         idcg = sum((2**1 - 1) / np.log2(i+2) for i in range(min(len(relevant_docs), len(retrieved_docs))))
#         return dcg / idcg if idcg > 0 else 0

#     def _calculate_recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
#         return len(set(retrieved_docs[:k]) & set(relevant_docs)) / len(relevant_docs)

# def main(retrieval_system: RetrievalSystem, system_name: str):
#     evaluator = Evaluator(retrieval_system, system_name)
#     results = evaluator.evaluate('../data/single_paper.json', '../data/multi_paper.json', '../data/multi_paper_sentences.json', k_values=[10, 50, 100])
    
#     print("Evaluation Results:")
#     for eval_type, metrics in results.items():
#         print(f"\n{eval_type.replace('_', ' ').title()} Evaluation:")
#         if isinstance(metrics, dict) and all(isinstance(v, dict) for v in metrics.values()):
#             for k, k_metrics in metrics.items():
#                 print(f"  For k={k}:")
#                 for metric, value in k_metrics.items():
#                     print(f"    {metric}: {value:.4f}")
#         else:
#             for metric, value in metrics.items():
#                 print(f"  {metric}: {value:.4f}")

# if __name__ == "__main__":
#     pass
#     # Example usage:
#     # from bow_retrieval import BagOfWordsRetrievalSystem
#     # retrieval_system = BagOfWordsRetrievalSystem("charlieoneill/jsalt-astroph-dataset")
#     # main(retrieval_system, "BagOfWords")


import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from collections import defaultdict
import wandb
import numpy as np
from tqdm import tqdm
from datetime import datetime

class RetrievalSystem(ABC):
    @abstractmethod
    def retrieve(self, query: str, arxiv_id: str, top_k: int = 100) -> List[str]:
        pass

    def parse_date(self, arxiv_id: str) -> datetime:
        try:
            year = int("20" + arxiv_id[:2])
            month = int(arxiv_id[2:4])
        except:
            year = 2023
            month = 1
        return datetime(year, month, 1)
    

class Evaluator:
    def __init__(self, retrieval_system: RetrievalSystem, system_name: str, wandb_log: bool = True):
        self.retrieval_system = retrieval_system
        self.system_name = system_name
        self.wandb_log = wandb_log
    
    

    def evaluate(self, single_doc_file: str, multi_paper_sentences_file: str, k_values: List[int] = [10, 50, 100]) -> Dict[str, Dict[str, float]]:
        if self.wandb_log:
            wandb.init(project="jsalt-astro", name=f"{self.system_name}")

        results = {}
        
        single_results = self._evaluate_single_document(single_doc_file, 10) #{'success_rate': 0.84, 'reciprocal_rank': 0.7353, 'avg_precision': 0.7353} #
        results['single_doc'] = single_results
        print("Single Document Results:", single_results)

        multipaper_sentences_results = self._evaluate_multipaper_sentences(multi_paper_sentences_file, k_values)
        results['multipaper_sentences'] = multipaper_sentences_results
        print("Multipaper Sentences Results:", multipaper_sentences_results)
        
        log_dict = {}
        for eval_type, metrics in results.items():
            if eval_type == 'single_doc':
                for metric, value in metrics.items():
                    log_dict[f"single_doc_{metric}"] = value
            else:  # multipaper_sentences
                for k, k_metrics in metrics.items():
                    for metric, value in k_metrics.items():
                        log_dict[f"{eval_type}_{metric}@{k}"] = value
        
        if self.wandb_log:
            wandb.log(log_dict)
            wandb.finish()
        
        return results

    def _evaluate_single_document(self, ground_truth_file: str, top_k: int = 10) -> Dict[str, float]:
        ground_truth = self._load_ground_truth(ground_truth_file)
        results = defaultdict(list)
        
        total_queries = sum(len(data) for data in ground_truth.values())
        with tqdm(total=total_queries, desc="Single-doc progress") as pbar:
            for arxiv_id, data in ground_truth.items():
                for question_type in ['question_abstract', 'question_conclusion']:
                    if question_type in data:
                        query = data[question_type]
                        retrieved_docs = self.retrieval_system.retrieve(query, arxiv_id, top_k=top_k)
                        
                        results['success_rate'].append(int(arxiv_id in retrieved_docs))
                        results['reciprocal_rank'].append(self._calculate_reciprocal_rank(retrieved_docs, arxiv_id))
                        results['avg_precision'].append(self._calculate_avg_precision(retrieved_docs, arxiv_id))

                        pbar.update(1)
        
        return {metric: sum(values) / len(values) for metric, values in results.items()}

    def _evaluate_multipaper_sentences(self, ground_truth_file: str, k_values: List[int]) -> Dict[str, Dict[str, float]]:
        ground_truth = self._load_ground_truth(ground_truth_file)
        results = {k: defaultdict(list) for k in k_values}
        
        for item_id, item in tqdm(ground_truth.items(), desc="Multipaper sentences progress"):
            arxiv_id_clean = item_id.split('_')[0]
            query = item['question']
            retrieved_docs = self.retrieval_system.retrieve(query, arxiv_id_clean, top_k=max(k_values))
            relevant_docs = item['arxiv']

            for k in k_values:
                results[k]['map'].append(self._calculate_map(retrieved_docs[:k], relevant_docs))
                results[k]['ndcg'].append(self._calculate_ndcg(retrieved_docs[:k], relevant_docs))
                results[k]['recall'].append(self._calculate_recall_at_k(retrieved_docs[:k], relevant_docs, k))
        
        return {k: {metric: np.mean(values) for metric, values in k_results.items()} for k, k_results in results.items()}

    @staticmethod
    def _load_ground_truth(file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            return json.load(f)

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

def main(retrieval_system: RetrievalSystem, system_name: str):
    evaluator = Evaluator(retrieval_system, system_name)
    results = evaluator.evaluate('../data/single_paper.json', '../data/multi_paper_sentences.json', k_values=[10, 50, 100])
    
    print("Evaluation Results:")
    for eval_type, metrics in results.items():
        print(f"\n{eval_type.replace('_', ' ').title()} Evaluation:")
        if isinstance(metrics, dict) and all(isinstance(v, dict) for v in metrics.values()):
            for k, k_metrics in metrics.items():
                print(f"  For k={k}:")
                for metric, value in k_metrics.items():
                    print(f"    {metric}: {value:.4f}")
        else:
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    pass
    # Example usage:
    # from bow_retrieval import BagOfWordsRetrievalSystem
    # retrieval_system = BagOfWordsRetrievalSystem("charlieoneill/jsalt-astroph-dataset")
    # main(retrieval_system, "BagOfWords")