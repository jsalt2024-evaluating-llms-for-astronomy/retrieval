import json
from typing import List, Dict
from datasets import load_dataset

path = "/Users/charlesoneill/retrieval/data/multi_paper.json"

with open(path, 'r') as f:
    data = json.load(f)

# # Remove any data where the arxiv field is empty list
# data = [{k: v} for k, v in data.items() if v['arxiv']]
# print(len(data))

# def _map_arxiv_to_filenames(self, arxiv_ids: List[str]) -> List[str]:
#         #self.document_ids.append(f"{paper['subfolder']}/{paper['filename']}")
#         ids_list = []
#         for arxiv_id in arxiv_ids:
#             if '.' in arxiv_id:
#                 print(arxiv_id)
#                 year_month = arxiv_id.split('.')[0]
#                 identifier = arxiv_id.split('.')[1].split('_')[0]
#                 ids_list.append(f"{year_month}/{year_month}.{identifier}_arXiv.txt")
#             else:
#                 year_month = arxiv_id[:4]
#                 identifier = arxiv_id[4:]
#                 ids_list.append(f"{year_month}/{year_month}.{identifier}_arXiv")
#         return ids_list

# # Apply this to the data
# for d in data:
#     key_zero = list(d.keys())[0]   
#     arxiv_ids = d[key_zero]['arxiv']
#     d[key_zero]['arxiv'] = _map_arxiv_to_filenames(None, arxiv_ids)

# # Save the updated data
# with open(path, 'w') as f:
#     json.dump(data, f)
    

# # Load the huggingface dataset
repo_id = "charlieoneill/jsalt-astroph-dataset"
dataset = load_dataset(repo_id, split="train")

# Get all the filenames from the dataset
filenames = dataset['filename']

# Only keep citations in the data that are in filenames
for d in data:
    key_zero = list(d.keys())[0]
    arxiv_ids = d[key_zero]['arxiv']
    d[key_zero]['arxiv'] = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id in filenames]

# Print how many citations for each d in data
for d in data:
    key_zero = list(d.keys())[0]
    print(len(d[key_zero]['arxiv']))

