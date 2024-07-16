from umap.umap_ import UMAP
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from topk_sae import FastAutoencoder, loss_fn, unit_norm_decoder_grad_adjustment_, unit_norm_decoder_, init_from_data_
import json
import anthropic 
import yaml 
import concurrent.futures
import time
from threading import Thread
import re
import torch
config = yaml.safe_load(open('../config.yaml', 'r'))
anthropic_key = config['anthropic_api_key']
generation_client = anthropic.Anthropic(api_key = anthropic_key)

auto_results = json.load(open('sae_data/feature_analysis_results.json'))
auto_embeds = np.load('sae_data/feature_embeddings.npy')
auto_labels = open('sae_data/feature_labels.txt').read().splitlines()
abstract_embeddings = np.load("../data/vector_store/abstract_embeddings.npy")
abstract_texts = json.load(open('../data/vector_store/abstract_texts.json'))['abstracts']
num_abstracts = len(abstract_embeddings)

def get_score(scores):
    score_pairs = []
    scores = scores.split('\n\n')[1:]
    scores = '\n'.join(scores)
    for pair in scores.split('\n'):
        split = pair.split(':')
        try:
            score_pairs.append((split[0], int(split[1])))
        except:
            score_pairs.append((split[0], 0))
    return score_pairs

def claude_score(labels, generation_model = "claude-3-5-sonnet-20240620"):
    input_text = "\n".join(labels)
    message = generation_client.messages.create(
            model = generation_model,
            max_tokens = 4096,
            temperature = 0.3,
            system = """You are a professional astronomer classifying different topics in the astronomy literature.
                        You will be given a list of feature labels.
                        For each label, assess how how related it is to the methodology of astronomy research and assign a score from 1 to 10.
                        Do not assign high scores to highly jargon-specific labels, or labels that discuss named entities and astronomical objects.
                        Assign high scores to techniques related to high-level scientific ideas and abstract thinking, but not to specific phenomena.
                        Assign high scores to labels that deal with the general structure of problem-solving in astronomy.
                        Assign high scores to methodology that can be applied across multiple different astronomical problems.
                        Be self-consistent -- if you give a high score to one label, give a high score to similar labels.
                        Return scores in the format {label}: {score}.
                        Do not miss any labels, and do not add any labels.""",
            messages=[{ "role": "user",
                    "content": [{"type": "text", "text": input_text}] }]
        )

    message =  message.content[0].text
    scores = get_score(message)

    return scores

def get_all_scores(labels, batch_size=100):
    def process_batch(batch):
        return claude_score(batch)

    def retry_batch(batch):
        while True:
            try:
                scores = process_batch(batch)
                print('scores:', len(scores), len(batch))
                return scores
            except Exception as exc:
                print(f"Retry batch processing generated an exception {exc}. Retrying in 15 seconds...")
                time.sleep(15)

    all_scores = []
    batches = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]
    print('Processing {} batches'.format(len(batches)))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                scores = future.result()
                print('scores', len(scores), len(batch))
                all_scores.extend(scores)
            except Exception as exc:
                print(f"Batch processing generated an exception. Retrying in parallel...")
                retry_thread = Thread(target=lambda: all_scores.extend(retry_batch(batch)))
                retry_thread.start()
    
    return all_scores

def clean(label):
    label = label.split(' in astronomy')[0].split(' in astrophysics')[0].split(' in physics')[0].split(' in astronomical')[0].split(' in astrophysical')[0]
    label = label.split(' in Astronomy')[0].split(' in Astrophysics')[0].split(' in Physics')[0].split(' in Astronomical')[0].split(' in Astrophysical')[0]
    if len(re.findall(r"""["'][A-Za-z]["']""", label)) > 0:
        return ""
    if len(re.findall(r"[Kk]eyword", label)) > 0:
        return ""
    if len(re.findall(r"[Tt]opic", label)) > 0:
        return ""
    if len(re.findall(r"[Cc]oncept", label)) > 0:
        return ""
    if len(re.findall(r"[Pp]resence", label)) > 0:
        return ""
    return label

def get_clean_labels(auto_results):
    clean_labels = {}
    for result in auto_results:
        label = result['label']
        clean_label = clean(label)
        
        if clean_label != "":
            result['clean_label'] = clean_label
            if result['f1'] >= 0.6 and result['pearson_correlation'] >= 0.6:
                if clean_label not in clean_labels.keys():
                    clean_labels[clean_label] = {'index': result['index'], 'density': result['density'], 'f1': result['f1'], 'pearson_correlation': result['pearson_correlation'],}
                else:
                    # keep only the highest-scoring label if repeats
                    existing = clean_labels[clean_label]
                    if result['f1'] + result['pearson_correlation'] > existing['f1'] + existing['pearson_correlation']:
                        clean_labels[clean_label] = {'index': result['index'], 'density': result['density'], 'f1': result['f1'], 'pearson_correlation': result['pearson_correlation'],}
    
    return clean_labels

def get_multiplier(n_dirs, all_scores, clean_labels):
    multiplier = np.ones(n_dirs, dtype = np.float32)
    for score in all_scores:
        try:
            idx = clean_labels[score[0].strip(' ')]
            if score[1] < 5:
                multiplier[idx]
            else:
                multiplier[idx] = score[1] - 4
        except:
            print(score)
    
    return multiplier

def nn_search(abstract_id, embeddings, k):
    query_emb = embeddings[abstract_id]
    scores = np.dot(embeddings, query_emb)
    top_score_ids = np.argsort(scores)[::-1][:k]
    return [(abstract_texts[i], scores[i]) for i in top_score_ids]

def recompute_embeddings(ae, dataloader, multiplier, device, d_model = 1536, batch_size = 1024):
    new_embs = np.zeros((num_abstracts, d_model), dtype=np.float32)
    with torch.no_grad():
        for i, (batch,) in enumerate(tqdm(dataloader, desc="Processing abstracts")):
            batch = batch.to(device)
            _, info = ae(batch)
            start_idx = i * batch_size
            end_idx = start_idx + batch.size(0)
            latents_pre_act = info['latents_pre_act']

            # clamp values
            embs = ae.decode_clamp(latents_pre_act, clamp = multiplier)
            new_embs[start_idx:end_idx] = embs.cpu().numpy()
    
    new_embs /= np.linalg.norm(new_embs, axis = 1, keepdims = True)

    return new_embs

def load_scores(path):
    return json.load(open(path))

def main():
    # Setup
    torch.set_grad_enabled(False)

    d_model = 1536
    n_dirs = d_model * 6
    k = 64
    auxk = 128
    multik = 256
    batch_size = 1024

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ae = FastAutoencoder(n_dirs, d_model, k, auxk, multik = multik).to(device)
    model_path = 'checkpoints/64_9216_128_auxk_epoch_50.pth'
    ae.load_state_dict(torch.load(model_path))
    ae.eval()
    
    dataset = TensorDataset(torch.from_numpy(abstract_embeddings))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Functionality
    mode = "methodology"
    clean_labels = get_clean_labels(auto_results)
    all_scores = load_scores('sae_data/claude_scores_{}.json'.format(mode))
    # all_scores = get_all_scores(list(clean_labels.keys()))
    # json.dump(all_scores, open('sae_data/claude_scores.json', 'w'))

    multiplier = get_multiplier(n_dirs, all_scores, clean_labels)
    new_embs = recompute_embeddings(ae, dataloader, device, multiplier)
    np.save('sae_data/abstract_embeddings_{}.npy'.format(mode), new_embs)
