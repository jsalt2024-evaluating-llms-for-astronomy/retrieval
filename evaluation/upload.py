import os
import json
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, HfFolder

# Your Hugging Face username and repository name
username = "charlieoneill"
repo_name = "jsalt-astroph-dataset"

# # Path to the local data directory
# local_data_path = "/Users/charlesoneill/Desktop/jsalt/arxiv-data"

# # Authenticate with Hugging Face
# api = HfApi()
# token = HfFolder.get_token()
# if not token:
#     raise ValueError("You need to login to Hugging Face CLI using `huggingface-cli login`")

# # Create the repository on Hugging Face if it doesn't exist
repo_id = f"{username}/{repo_name}"
# api.create_repo(repo_id=repo_id, private=False, repo_type="dataset", exist_ok=True)

# def extract_sections(file_content):
#     sections = {'abstract': '', 'introduction': '', 'conclusions': ''}
#     current_section = None

#     for line in file_content.split('\n'):
#         line_lower = line.lower()
#         if 'abstract:' in line_lower:
#             current_section = 'abstract'
#             sections[current_section] = line.split(':', 1)[1].strip()
#         elif 'introduction:' in line_lower:
#             current_section = 'introduction'
#             sections[current_section] = line.split(':', 1)[1].strip()
#         elif 'conclusions:' in line_lower or 'conclusion:' in line_lower:
#             current_section = 'conclusions'
#             sections[current_section] = line.split(':', 1)[1].strip()
#         elif current_section:
#             sections[current_section] += ' ' + line.strip()

#     return sections

# def process_all_folders(base_path):
#     combined_data = []

#     for subfolder in os.listdir(base_path):
#         subfolder_path = os.path.join(base_path, subfolder)
#         if os.path.isdir(subfolder_path):
#             for filename in tqdm(os.listdir(subfolder_path), desc=f"Processing {subfolder}"):
#                 if filename.endswith('.txt'):
#                     file_path = os.path.join(subfolder_path, filename)
#                     with open(file_path, 'r') as file:
#                         content = file.read()
#                         sections = extract_sections(content)
#                         combined_data.append({
#                             "subfolder": subfolder,
#                             "filename": filename,
#                             "abstract": sections['abstract'],
#                             "introduction": sections['introduction'],
#                             "conclusions": sections['conclusions']
#                         })
    
#     return combined_data

# def create_dataset(combined_data):
#     dataset = Dataset.from_list(combined_data)
#     return DatasetDict({"train": dataset})

# # Process all folders and create a combined dataset
# combined_data = process_all_folders(local_data_path)
# dataset_dict = create_dataset(combined_data)

# # Save the dataset to disk
# dataset_dict.save_to_disk("/Users/charlesoneill/Desktop/jsalt/arxiv-dataset")

# # Push the dataset to the Hugging Face Hub
# dataset_dict.push_to_hub(repo_id)

# print(f"Dataset successfully uploaded to {repo_id}")

# Load the dataset from huggingface
dataset = load_dataset(repo_id, split="train")

def filename_to_year_month(example):
    year_month = example['subfolder']
    print(year_month)
    try:
        example['year'] = int(year_month[:2])
        example['month'] = int(year_month[2:])
    except:
        example['year'] = 0
        example['month'] = 0
    return example
    
updated_dataset = dataset.map(filename_to_year_month)

# Push back to hub
updated_dataset.push_to_hub(repo_id)
