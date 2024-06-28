from huggingface_hub import HfApi, create_repo
import os
import yaml

# Initialize the Hugging Face API
api = HfApi()
config = yaml.safe_load(open('../config.yaml', 'r'))

# Set your Hugging Face token
# It's better to use an environment variable for security
hf_token = config['hf_token']
if not hf_token:
    raise ValueError("Please set the HF_TOKEN environment variable")

# Set the local path of your .pkl file
local_path = "/Users/charlesoneill/retrieval/data/embeddings/embeddings_final.pkl"

# Set the repository name on Hugging Face
repo_name = "JSALT2024-Astro-LLMs/jsalt-astro-embeddings"

# Create the repository if it doesn't exist
# try:
#     create_repo(repo_name, token=hf_token, private=True)
#     print(f"Repository '{repo_name}' created successfully")
# except Exception as e:
#     if "already exists" in str(e):
#         print(f"Repository '{repo_name}' already exists")
#     else:
#         raise e

# Upload the file
api.upload_file(
    path_or_fileobj=local_path,
    path_in_repo="embeddings_final.pkl",
    repo_id=repo_name,
    token=hf_token
)

print(f"File uploaded successfully to {repo_name}")