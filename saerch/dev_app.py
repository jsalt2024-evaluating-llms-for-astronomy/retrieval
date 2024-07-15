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
    feature_indices_state = gr.State([])

    def intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae):
        with torch.no_grad():
            return ae.decode_sparse(topk_indices, topk_values)

    def update_search_results(feature_values, feature_indices):
        print(f"Entering update_search_results")
        print(f"Received feature_values: {feature_values[:5]}...")
        print(f"Received feature_indices: {feature_indices[:5]}...")

        # Ensure feature_values and feature_indices are flat lists
        feature_values = [v for sublist in feature_values for v in (sublist if isinstance(sublist, list) else [sublist])]
        feature_indices = [i for i in feature_indices if isinstance(i, (int, np.integer))]

        print(f"Flattened feature_values: {feature_values[:5]}...")
        print(f"Filtered feature_indices: {feature_indices[:5]}...")

        # Reconstruct query embedding
        topk_indices = torch.tensor(feature_indices).to(device)
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

        print(f"Exiting update_search_results")
        print(f"Returning feature_values: {feature_values[:5]}...")
        print(f"Returning feature_indices: {feature_indices[:5]}...")
        return search_results, feature_values, feature_indices

    @gr.render(inputs=[input_text, search_results_state, feature_values_state, feature_indices_state])
    def show_components(text, search_results, feature_values, feature_indices):
        print(f"Entering show_components")
        print(f"Input text: {text}")
        print(f"Received search_results: {len(search_results) if search_results else 0}")
        print(f"Received feature_values: {feature_values[:5] if feature_values else 'None'}...")
        print(f"Received feature_indices: {feature_indices[:5] if feature_indices else 'None'}...")

        if len(text) == 0:
            return gr.Markdown("## No Input Provided")

        if not search_results or text != getattr(show_components, 'last_query', None):
            print("New query detected, updating results and sliders")
            show_components.last_query = text
            query_embedding = get_embedding(text)

            with torch.no_grad():
                recons, z_dict = ae(torch.tensor(query_embedding).unsqueeze(0).to(device))
                topk_indices = z_dict['topk_indices'][0].cpu().numpy()
                topk_values = z_dict['topk_values'][0].cpu().numpy()

            feature_values = topk_values.tolist()
            feature_indices = topk_indices.tolist()
            print(f"New feature_values: {feature_values[:5]}...")
            print(f"New feature_indices: {feature_indices[:5]}...")
            search_results, feature_values, feature_indices = update_search_results(feature_values, feature_indices)

        df = gr.Dataframe(
            headers=["Title", "Citation Count", "Year"],
            value=search_results,
            label="Top 10 Search Results"
        )

        sliders = []
        print(f"Creating sliders with feature_values: {feature_values[:5]}...")
        print(f"Creating sliders with feature_indices: {feature_indices[:5]}...")
        for i, (value, index) in enumerate(zip(feature_values, feature_indices)):
            print(f"Creating slider for feature {index} with value {value}")
            feature = next((f for f in feature_analysis if f['index'] == index), None)
            label = f"{feature['label']} ({index})" if feature else f"Feature {index}"
            slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=value, label=label, key=f"slider-{index}")
            sliders.append(slider)

        def on_slider_change(*values):
            print("Entering on_slider_change")
            print(f"Received values: {values[:5]}...")
            new_results, new_values, new_indices = update_search_results(values, feature_indices)
            print(f"New feature_values after update: {new_values[:5]}...")
            print(f"New feature_indices after update: {new_indices[:5]}...")
            print("Exiting on_slider_change")
            return new_results, new_values, new_indices

        gr.Button("Update Results").click(
            on_slider_change,
            inputs=sliders + [feature_indices_state],
            outputs=[search_results_state, feature_values_state, feature_indices_state]
        )

        print(f"Exiting show_components")
        return [df] + sliders


demo.launch()

# import gradio as gr
# import numpy as np
# import json
# import pandas as pd
# from openai import OpenAI
# import yaml
# from typing import Optional, List, Dict, Tuple, Any
# from topk_sae import FastAutoencoder
# import torch

# # Load configuration and initialize OpenAI client
# config = yaml.safe_load(open('../config.yaml', 'r'))
# client = OpenAI(api_key=config['openai_api_key'])

# EMBEDDING_MODEL = "text-embedding-3-small"

# # Load pre-computed embeddings and texts
# embeddings_path = "../data/vector_store/abstract_embeddings.npy"
# texts_path = "../data/vector_store/abstract_texts.json"
# feature_analysis_path = "sae_data/feature_analysis_results.json"
# metadata_path = 'sae_data/astro_paper_metadata.csv'

# abstract_embeddings = np.load(embeddings_path)
# with open(texts_path, 'r') as f:
#     abstract_texts = json.load(f)
# with open(feature_analysis_path, 'r') as f:
#     feature_analysis = json.load(f)

# # Load metadata
# df_metadata = pd.read_csv(metadata_path)

# # Set up sparse autoencoder
# torch.set_grad_enabled(False)
# d_model = 1536
# n_dirs = d_model * 6
# k = 64
# auxk = 128
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ae = FastAutoencoder(n_dirs, d_model, k, auxk, multik=0).to(device)
# model_path = 'checkpoints/64_9216_128_auxk_epoch_50.pth'
# ae.load_state_dict(torch.load(model_path))
# ae.eval()

# def get_embedding(text: Optional[str], model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
#     try:
#         embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
#         return np.array(embedding, dtype=np.float32)
#     except Exception as e:
#         print(f"Error getting embedding: {e}")
#         return None
    

# def intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae):
#     return ae.decode_sparse(topk_indices, topk_values)  


# with gr.Blocks() as demo:
#     input_text = gr.Textbox(label="input")
#     search_results_state = gr.State([])
#     feature_values_state = gr.State([])
#     feature_indices_state = gr.State([])

#     def intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae):
#         with torch.no_grad():
#             return ae.decode_sparse(topk_indices, topk_values)

#     def update_search_results(feature_values, feature_indices):
#         # Ensure feature_values and feature_indices are flat lists
#         feature_values = [v for sublist in feature_values for v in (sublist if isinstance(sublist, list) else [sublist])]
#         feature_indices = [i for i in feature_indices if isinstance(i, (int, np.integer))]

#         # Reconstruct query embedding
#         topk_indices = torch.tensor(feature_indices).to(device)
#         topk_values = torch.tensor(feature_values).to(device)
        
#         intervened_embedding = intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae)
#         intervened_embedding = intervened_embedding.cpu().numpy().flatten()

#         # Perform similarity search
#         sims = np.dot(abstract_embeddings, intervened_embedding)
#         topk_indices_search = np.argsort(sims)[::-1][:10]
#         doc_ids = abstract_texts['doc_ids']
#         topk_doc_ids = [doc_ids[i] for i in topk_indices_search]

#         # Prepare search results
#         search_results = []
#         for doc_id in topk_doc_ids:
#             metadata = df_metadata[df_metadata['arxiv_id'] == doc_id].iloc[0]
#             title = metadata['title'].replace('[', '').replace(']', '')
#             search_results.append([
#                 title,
#                 int(metadata['citation_count']),
#                 int(metadata['year'])
#             ])

#         return search_results, feature_values, feature_indices

#     @gr.render(inputs=[input_text, search_results_state, feature_values_state, feature_indices_state])
#     def show_components(text, search_results, feature_values, feature_indices):
#         if len(text) == 0:
#             return gr.Markdown("## No Input Provided")

#         if not search_results:
#             query_embedding = get_embedding(text)

#             with torch.no_grad():
#                 recons, z_dict = ae(torch.tensor(query_embedding).unsqueeze(0).to(device))
#                 topk_indices = z_dict['topk_indices'][0].cpu().numpy()
#                 topk_values = z_dict['topk_values'][0].cpu().numpy()

#             feature_values = topk_values.tolist()
#             feature_indices = topk_indices.tolist()
#             search_results, feature_values, feature_indices = update_search_results(feature_values, feature_indices)

#         df = gr.Dataframe(
#             headers=["Title", "Citation Count", "Year"],
#             value=search_results,
#             label="Top 10 Search Results"
#         )

#         sliders = []
#         for i, (value, index) in enumerate(zip(feature_values, feature_indices)):
#             feature = next((f for f in feature_analysis if f['index'] == index), None)
#             label = f"{feature['label']} ({index})" if feature else f"Feature {index}"
#             slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=value, label=label, key=f"slider-{i}")
#             sliders.append(slider)

#         def on_slider_change(*values):
#             print("Changing slider values and updating search results")
#             new_results, new_values, new_indices = update_search_results(values, feature_indices)
#             return new_results, new_values, new_indices

#         gr.Button("Update Results").click(
#             on_slider_change,
#             inputs=sliders + [feature_indices_state],
#             outputs=[search_results_state, feature_values_state, feature_indices_state]
#         )

#         return [df] + sliders


# demo.launch()

