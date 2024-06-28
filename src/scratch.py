# from huggingface_hub import HfApi, create_repo
# import os
# import yaml

# # Initialize the Hugging Face API
# api = HfApi()
# config = yaml.safe_load(open('../config.yaml', 'r'))

# # Set your Hugging Face token
# # It's better to use an environment variable for security
# hf_token = config['hf_token']
# if not hf_token:
#     raise ValueError("Please set the HF_TOKEN environment variable")

# # Set the local path of your .pkl file
# local_path = "/Users/charlesoneill/retrieval/data/embeddings/embeddings_final.pkl"

# # Set the repository name on Hugging Face
# repo_name = "JSALT2024-Astro-LLMs/jsalt-astro-embeddings"

# # Create the repository if it doesn't exist
# # try:
# #     create_repo(repo_name, token=hf_token, private=True)
# #     print(f"Repository '{repo_name}' created successfully")
# # except Exception as e:
# #     if "already exists" in str(e):
# #         print(f"Repository '{repo_name}' already exists")
# #     else:
# #         raise e

# # Upload the file
# api.upload_file(
#     path_or_fileobj=local_path,
#     path_in_repo="embeddings_final.pkl",
#     repo_id=repo_name,
#     token=hf_token
# )

# print(f"File uploaded successfully to {repo_name}")

import os
from huggingface_hub import HfApi, HfFolder, create_repo
import yaml

def upload_to_huggingface(local_dir: str, repo_id: str, token: str):
    # Initialize the Hugging Face API
    api = HfApi()

    # Set up the token
    HfFolder.save_token(token)

    # List of files to upload
    files_to_upload = [
        "documents.pkl",
        "document_index.pkl",
        "embeddings_matrix.npy",
        "index_mapping.pkl"
    ]

    # Upload each file
    for filename in files_to_upload:
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=repo_id,
                #repo_type="dataset",
            )
            print(f"Uploaded {filename} successfully.")
        else:
            print(f"Warning: {filename} not found in {local_dir}")

    print("All files uploaded successfully.")

if __name__ == "__main__":
    # Load configuration
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set up variables
    local_dir = "../data"
    repo_id = "JSALT2024-Astro-LLMs/jsalt-astro-embeddings"
    hf_token = config['hf_token']

    # Run the upload function
    upload_to_huggingface(local_dir, repo_id, hf_token)