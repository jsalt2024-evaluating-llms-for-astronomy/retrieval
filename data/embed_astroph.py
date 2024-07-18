import os
import pickle
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
import time
import concurrent.futures
import yaml
import tiktoken
import random
import glob
import numpy as np

# Initialize OpenAI client
config = yaml.safe_load(open('../config.yaml', 'r'))
client = OpenAI(api_key=config['openai_api_key'])

# Configuration
DATASET_NAME = "charlieoneill/cs.LG" #"charlieoneill/jsalt-astroph-dataset"
BATCH_SIZE = 100
SAVE_INTERVAL = 1000
OUTPUT_DIR = "embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_WORKERS = 5  # Adjust this based on your API rate limits and system capabilities
MAX_TOKENS = 8192
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 10
MAX_RETRY_DELAY = 120

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens])
    return text

def get_embedding(text: Optional[str], model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
    if not text or text.strip() == "":
        return None
    text = text.replace("\n", " ")
    text = truncate_text(text)  # Truncate to MAX_TOKENS
    
    retry_delay = INITIAL_RETRY_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
            return np.array(embedding, dtype=np.float32)  # Convert to 32-bit float NumPy array
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Failed to get embedding after {MAX_RETRIES} attempts: {e}")
                return None
            
            retry_delay = min(MAX_RETRY_DELAY, retry_delay * 2)
            jitter = random.uniform(0, 0.1 * retry_delay)
            sleep_time = retry_delay + jitter
            
            print(f"Error getting embedding (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)

# def process_batch(batch: Dict) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
#     embeddings = {}
#     print(f"batch['filename']: {batch['filename']}")
#     for i in range(len(batch['filename'])):
#         filename = f"{batch['subfolder'][i]}/{batch['filename'][i]}"
#         embeddings[filename] = {
#             'abstract': get_embedding(batch['abstract'][i]),
#         }
#     return embeddings

def process_batch(batch: Dict) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    embeddings = {}
    # print(f"batch['id']: {batch['id']}")
    for i in range(len(batch['id'])):
        filename = f"{batch['id'][i]}"
        embeddings[filename] = {
            'abstract': get_embedding(batch['abstract'][i]),
        }
    return embeddings

def save_embeddings(embeddings: Dict, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(filename: str) -> Dict:
    with open(filename, 'rb') as f:
        return pickle.load(f)

def find_last_checkpoint() -> Tuple[Dict, int]:
    checkpoint_files = glob.glob(f"{OUTPUT_DIR}/embeddings_csLG_*.pkl")
    if not checkpoint_files:
        return {}, 0
    
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    embeddings = load_embeddings(latest_checkpoint)
    last_processed = int(latest_checkpoint.split('_')[-1].split('.')[0])
    return embeddings, last_processed

def cleanup_old_checkpoints(keep_file: str):
    checkpoint_files = glob.glob(f"{OUTPUT_DIR}/embeddings_csLG_*.pkl")
    for file in checkpoint_files:
        if file != keep_file and file != f"{OUTPUT_DIR}/embeddings_final_csLG.pkl":
            os.remove(file)
            print(f"Deleted old checkpoint: {file}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_dataset(DATASET_NAME, split="train")

    all_embeddings, processed_count = find_last_checkpoint()
    print(f"Resuming from {processed_count} processed documents")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {}
        for i in range(processed_count, len(dataset), BATCH_SIZE):
            batch = dataset[i:i+BATCH_SIZE]
            future = executor.submit(process_batch, batch)
            future_to_batch[future] = i

        for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=len(future_to_batch), desc="Processing batches"):
            batch_index = future_to_batch[future]
            try:
                batch_embeddings = future.result()
                all_embeddings.update(batch_embeddings)
                processed_count += len(batch_embeddings)

                if processed_count % SAVE_INTERVAL == 0:
                    checkpoint_file = f"{OUTPUT_DIR}/embeddings_csLG_{processed_count}.pkl"
                    save_embeddings(all_embeddings, checkpoint_file)
                    print(f"Saved embeddings for {processed_count} documents")
                    cleanup_old_checkpoints(checkpoint_file)
            
            except Exception as e:
                print(f"Batch starting at index {batch_index} generated an exception: {e}")

    final_file = f"{OUTPUT_DIR}/embeddings_final_csLG.pkl"
    save_embeddings(all_embeddings, final_file)
    cleanup_old_checkpoints(final_file)
    print(f"Finished processing. Total documents embedded: {processed_count}")

if __name__ == "__main__":
    main()