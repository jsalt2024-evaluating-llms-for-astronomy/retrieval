# import os
# import json
# from tqdm import tqdm
# from huggingface_hub import HfApi, HfFolder

# # Your Hugging Face username and repository name
# username = "charlieoneill"
# repo_name = "jsalt-astroph-data"

# # Path to the local data directory
# local_data_path = "/Users/charlesoneill/Desktop/jsalt/arxiv-data"

# # Authenticate with Hugging Face
# api = HfApi()
# token = HfFolder.get_token()
# if not token:
#     raise ValueError("You need to login to Hugging Face CLI using `huggingface-cli login`")

# # Create the repository on Hugging Face if it doesn't exist
# repo_id = f"{username}/{repo_name}"
# api.create_repo(repo_id=repo_id, private=False, exist_ok=True)

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
#     combined_data = {}

#     for subfolder in os.listdir(base_path):
#         subfolder_path = os.path.join(base_path, subfolder)
#         if os.path.isdir(subfolder_path):
#             for filename in tqdm(os.listdir(subfolder_path), desc=f"Processing {subfolder}"):
#                 if filename.endswith('.txt'):
#                     file_path = os.path.join(subfolder_path, filename)
#                     with open(file_path, 'r') as file:
#                         content = file.read()
#                         sections = extract_sections(content)
#                         combined_data[f"{subfolder}/{filename}"] = sections
    
#     return combined_data

# def save_json_file(data, file_path):
#     with open(file_path, 'w') as json_file:
#         json.dump(data, json_file, indent=4)

# def upload_json_file(api, json_file_path, repo_id, path_in_repo):
#     api.upload_file(
#         path_or_fileobj=json_file_path,
#         path_in_repo=path_in_repo,
#         repo_id=repo_id,
#     )

# # Process all folders and save to a single JSON file
# combined_data = process_all_folders(local_data_path)
# final_json_path = os.path.join(local_data_path, "combined_data.json")
# save_json_file(combined_data, final_json_path)

# # Upload the final JSON file
# upload_json_file(api, final_json_path, repo_id, "combined_data.json")

# print(f"Data successfully uploaded to {repo_id}")

import os
import json
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder

# Your Hugging Face username and repository name
username = "charlieoneill"
repo_name = "jsalt-astroph-dataset"

# Path to the local data directory
local_data_path = "/Users/charlesoneill/Desktop/jsalt/arxiv-data"

# Authenticate with Hugging Face
api = HfApi()
token = HfFolder.get_token()
if not token:
    raise ValueError("You need to login to Hugging Face CLI using `huggingface-cli login`")

# Create the repository on Hugging Face if it doesn't exist
repo_id = f"{username}/{repo_name}"
api.create_repo(repo_id=repo_id, private=False, repo_type="dataset", exist_ok=True)

def extract_sections(file_content):
    sections = {'abstract': '', 'introduction': '', 'conclusions': ''}
    current_section = None

    for line in file_content.split('\n'):
        line_lower = line.lower()
        if 'abstract:' in line_lower:
            current_section = 'abstract'
            sections[current_section] = line.split(':', 1)[1].strip()
        elif 'introduction:' in line_lower:
            current_section = 'introduction'
            sections[current_section] = line.split(':', 1)[1].strip()
        elif 'conclusions:' in line_lower or 'conclusion:' in line_lower:
            current_section = 'conclusions'
            sections[current_section] = line.split(':', 1)[1].strip()
        elif current_section:
            sections[current_section] += ' ' + line.strip()

    return sections

def process_all_folders(base_path):
    combined_data = []

    for subfolder in os.listdir(base_path):
        subfolder_path = os.path.join(base_path, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in tqdm(os.listdir(subfolder_path), desc=f"Processing {subfolder}"):
                if filename.endswith('.txt'):
                    file_path = os.path.join(subfolder_path, filename)
                    with open(file_path, 'r') as file:
                        content = file.read()
                        sections = extract_sections(content)
                        combined_data.append({
                            "subfolder": subfolder,
                            "filename": filename,
                            "abstract": sections['abstract'],
                            "introduction": sections['introduction'],
                            "conclusions": sections['conclusions']
                        })
    
    return combined_data

def create_dataset(combined_data):
    dataset = Dataset.from_list(combined_data)
    return DatasetDict({"train": dataset})

# Process all folders and create a combined dataset
combined_data = process_all_folders(local_data_path)
dataset_dict = create_dataset(combined_data)

# Save the dataset to disk
dataset_dict.save_to_disk("/Users/charlesoneill/Desktop/jsalt/arxiv-dataset")

# Push the dataset to the Hugging Face Hub
dataset_dict.push_to_hub(repo_id)

print(f"Dataset successfully uploaded to {repo_id}")