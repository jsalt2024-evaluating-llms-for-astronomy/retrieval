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

def process_query(query) -> Tuple[List[List[Any]], List[Dict[str, Any]]]:
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return [["Error: Unable to generate query embedding"]], []

    # Pass through sparse autoencoder
    with torch.no_grad():
        recons, z_dict = ae(torch.tensor(query_embedding).unsqueeze(0).to(device))
        topk_indices = z_dict['topk_indices'][0].cpu().numpy()
        topk_values = z_dict['topk_values'][0].cpu().numpy()

    sims = np.dot(abstract_embeddings, query_embedding)
    topk_indices_search = np.argsort(sims)[::-1][:10]  # Get top 10 results
    doc_ids = abstract_texts['doc_ids']
    topk_doc_ids = [doc_ids[i] for i in topk_indices_search]

    # Prepare top 10 search results with metadata
    search_results = []
    for doc_id in topk_doc_ids:
        metadata = df_metadata[df_metadata['arxiv_id'] == doc_id].iloc[0]
        # Title with '[' and ']' removed
        title = metadata['title'].replace('[', '').replace(']', '')
        search_results.append([
            title,
            int(metadata['citation_count']),
            int(metadata['year'])
        ])

    # Prepare top 10 active features as a list of dictionaries
    top_features = []
    for index, value in zip(topk_indices[:10], topk_values[:10]):
        feature = next((f for f in feature_analysis if f['index'] == index), None)
        if feature:
            top_features.append({
                "label": feature['label'],
                "index": int(index),
                "value": float(value)
            })

    return search_results, top_features

def create_app():
    with gr.Blocks() as app:
        gr.Markdown("# SAErch")

        query_input = gr.Textbox(label="Enter your query")
        search_button = gr.Button("Search")
        results_output = gr.Dataframe(
            headers=["Title", "Citations", "Year"],
            label="Top 10 Search Results"
        )
        slider_container = gr.Column(visible=False)

        def search_and_display(query):
            search_results, active_features = process_query(query)
            #slider_container.clear()

            with slider_container:
                for feature in active_features:
                    slider = gr.Slider(
                        minimum=-1,
                        maximum=1,
                        step=0.01,
                        value=feature['value'],
                        label=feature['label'],
                        info=f"Index: {feature['index']}"
                    )
                    slider.render()

            slider_container.update(visible=True)
            return search_results

        search_button.click(
            search_and_display,
            inputs=query_input,
            outputs=[results_output]
        )

        # app.append(query_input)
        # app.append(search_button)
        # app.append(results_output)
        # app.append(slider_container)

    return app

# Create and launch the app
if __name__ == "__main__":
    app = create_app()
    app.launch()


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

# def process_query(query) -> Tuple[List[List[Any]], List[List[Any]]]:
#     query_embedding = get_embedding(query)
#     if query_embedding is None:
#         return [["Error: Unable to generate query embedding"]], []

#     # Pass through sparse autoencoder
#     with torch.no_grad():
#         recons, z_dict = ae(torch.tensor(query_embedding).unsqueeze(0).to(device))
#         topk_indices = z_dict['topk_indices'][0].cpu().numpy()
#         topk_values = z_dict['topk_values'][0].cpu().numpy()

#     sims = np.dot(abstract_embeddings, query_embedding)
#     topk_indices_search = np.argsort(sims)[::-1][:10]  # Get top 10 results
#     doc_ids = abstract_texts['doc_ids']
#     topk_doc_ids = [doc_ids[i] for i in topk_indices_search]

#     # Prepare top 10 search results with metadata
#     search_results = []
#     for doc_id in topk_doc_ids:
#         metadata = df_metadata[df_metadata['arxiv_id'] == doc_id].iloc[0]
#         title = metadata['title'].replace('[', '').replace(']', '')
#         search_results.append([
#             title,
#             int(metadata['citation_count']),
#             int(metadata['year'])
#         ])

#     # Prepare top 10 active features as a list of lists
#     top_features = []
#     for index, value in zip(topk_indices[:10], topk_values[:10]):
#         feature = next((f for f in feature_analysis if f['index'] == index), None)
#         if feature:
#             top_features.append([
#                 feature['label'],
#                 int(index),
#                 float(value)
#             ])

#     return search_results, top_features

# def create_app():
#     with gr.Blocks() as app:
#         gr.Markdown("# SAErch")

#         with gr.Row():
#             with gr.Column(scale=3):
#                 query_input = gr.Textbox(label="Enter your query")
#                 search_button = gr.Button("Search")
#                 results_output = gr.Dataframe(
#                     headers=["Title", "Citations", "Year"],
#                     label="Top 10 Search Results"
#                 )
#                 active_features_output = gr.Dataframe(
#                     headers=["Feature", "Index", "Value"],
#                     label="Top 10 Active Features"
#                 )

#             with gr.Column(scale=1) as feature_slider_container:
#                 feature_sliders = gr.Row()

#         def search_and_display(query):
#             search_results, active_features = process_query(query)
#             sliders = []

#             for feature in active_features:
#                 label, index, value = feature
#                 slider = gr.Slider(
#                     minimum=-1,
#                     maximum=1,
#                     value=value,
#                     step=0.01,
#                     label=label,
#                     info=f"Index: {index}"
#                 )
#                 sliders.append(slider)
            
#             feature_slider_container.children = sliders

#             return search_results, active_features

#         search_button.click(
#             search_and_display,
#             inputs=query_input,
#             outputs=[results_output, active_features_output]
#         )

#     return app

# # Create and launch the app
# if __name__ == "__main__":
#     app = create_app()
#     app.launch()
