import gradio as gr
import numpy as np
import json
import pandas as pd
from openai import OpenAI
import yaml
from typing import Optional, List, Dict, Tuple, Any
from topk_sae import FastAutoencoder
import torch

# Load configuration and initialize OpenAI client
config = yaml.safe_load(open('../config.yaml', 'r'))
client = OpenAI(api_key=config['openai_api_key'])

EMBEDDING_MODEL = "text-embedding-3-small"

# Load pre-computed embeddings and texts
embeddings_path = "../data/vector_store/abstract_embeddings.npy"
texts_path = "../data/vector_store/abstract_texts.json"
feature_analysis_path = "sae_data/feature_analysis_results.json"
metadata_path = 'sae_data/astro_paper_metadata.csv'

abstract_embeddings = np.load(embeddings_path)
with open(texts_path, 'r') as f:
    abstract_texts = json.load(f)
with open(feature_analysis_path, 'r') as f:
    feature_analysis = json.load(f)

# Load metadata
df_metadata = pd.read_csv(metadata_path)

# Set up sparse autoencoder
torch.set_grad_enabled(False)
d_model = 1536
n_dirs = d_model * 6
k = 64
auxk = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ae = FastAutoencoder(n_dirs, d_model, k, auxk, multik=0).to(device)
model_path = 'checkpoints/64_9216_128_auxk_epoch_50.pth'
ae.load_state_dict(torch.load(model_path))
ae.eval()

def get_embedding(text: Optional[str], model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
    try:
        embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None
    

def intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae):
    return ae.decode_sparse(topk_indices, topk_values)  


with gr.Blocks() as demo:
    input_text = gr.Textbox(label="input")
    search_results_state = gr.State([])
    feature_values_state = gr.State([])

    def intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae):
        with torch.no_grad():
            return ae.decode_sparse(topk_indices, topk_values)

    def update_search_results(feature_values):
        # Reconstruct query embedding
        topk_indices = torch.tensor(range(64)).to(device)  # Assuming 64 features total
        topk_values = torch.zeros(64).to(device)
        topk_values = torch.tensor(feature_values).to(device)
        
        intervened_embedding = intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae)
        intervened_embedding = intervened_embedding.cpu().numpy().flatten()

        # Perform similarity search
        sims = np.dot(abstract_embeddings, intervened_embedding)
        topk_indices_search = np.argsort(sims)[::-1][:10]
        doc_ids = abstract_texts['doc_ids']
        topk_doc_ids = [doc_ids[i] for i in topk_indices_search]

        # Prepare search results
        search_results = []
        for doc_id in topk_doc_ids:
            metadata = df_metadata[df_metadata['arxiv_id'] == doc_id].iloc[0]
            title = metadata['title'].replace('[', '').replace(']', '')
            search_results.append([
                title,
                int(metadata['citation_count']),
                int(metadata['year'])
            ])

        return search_results, feature_values

    @gr.render(inputs=[input_text, search_results_state, feature_values_state])
    def show_components(text, search_results, feature_values):
        if len(text) == 0:
            return gr.Markdown("## No Input Provided")

        if not search_results:
            query_embedding = get_embedding(text)

            with torch.no_grad():
                recons, z_dict = ae(torch.tensor(query_embedding).unsqueeze(0).to(device))
                topk_indices = z_dict['topk_indices'][0].cpu().numpy()
                topk_values = z_dict['topk_values'][0].cpu().numpy()

            feature_values = topk_values.tolist()
            search_results, _ = update_search_results(feature_values)

        df = gr.Dataframe(
            headers=["Title", "Citation Count", "Year"],
            value=search_results,
            label="Top 10 Search Results"
        )

        sliders = []
        for i, value in enumerate(feature_values):
            feature = next((f for f in feature_analysis if f['index'] == i), None)
            label = f"{feature['label']} ({i})" if feature else f"Feature {i}"
            slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=value, label=label, key=f"slider-{i}")
            sliders.append(slider)

        def on_slider_change(changed_value, feature_values):
            new_feature_values = list(feature_values)  # Create a copy of the current feature values
            new_feature_values[i] = changed_value  # Update the changed feature
            new_results, new_values = update_search_results(new_feature_values)
            return new_results, new_values

        for i, slider in enumerate(sliders):
            slider.release(
                on_slider_change,
                inputs=[slider, feature_values_state],
                outputs=[search_results_state, feature_values_state]
            )

        return [df] + sliders


demo.launch()
