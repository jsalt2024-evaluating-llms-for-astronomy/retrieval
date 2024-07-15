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


# Define the custom CSS
custom_css = """
#custom-slider-* {
    background-color: #ffe6e6;
}
"""

# Create the Blocks interface with the custom CSS
with gr.Blocks(css=custom_css) as demo:
    input_text = gr.Textbox(label="input")
    search_results_state = gr.State([])
    feature_values_state = gr.State([])
    feature_indices_state = gr.State([])
    manually_added_features_state = gr.State([])

    def intervened_hidden_to_intervened_embedding(topk_indices, topk_values, ae):
        with torch.no_grad():
            return ae.decode_sparse(topk_indices, topk_values)

    def update_search_results(feature_values, feature_indices, manually_added_features):
        print(f"\nEntering update_search_results")
        print(f"Received feature_values: {feature_values[:10]}...")
        print(f"Received feature_indices: {feature_indices[:10]}...")
        print(f"Received manually_added_features: {manually_added_features}")

        # Combine manually added features with query-generated features
        all_indices = []
        all_values = []
        
        # Add manually added features first
        for index in manually_added_features:
            if index not in all_indices:
                all_indices.append(index)
                all_values.append(feature_values[feature_indices.index(index)] if index in feature_indices else 0.0)
        
        # Add remaining query-generated features
        for index, value in zip(feature_indices, feature_values):
            if index not in all_indices:
                all_indices.append(index)
                all_values.append(value)

        print(f"Combined all_indices: {all_indices[:10]}...")
        print(f"Combined all_values: {all_values[:10]}...")

        # Reconstruct query embedding
        topk_indices = torch.tensor(all_indices).to(device)
        topk_values = torch.tensor(all_values).to(device)
        
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
        return search_results, all_values, all_indices

    @gr.render(inputs=[input_text, search_results_state, feature_values_state, feature_indices_state, manually_added_features_state])
    def show_components(text, search_results, feature_values, feature_indices, manually_added_features):
        print(f"Entering show_components")
        print(f"Input text: {text}")
        print(f"Received search_results: {len(search_results) if search_results else 0}")
        print(f"Received feature_values: {feature_values[:5] if feature_values else 'None'}...")
        print(f"Received feature_indices: {feature_indices[:5] if feature_indices else 'None'}...")
        print(f"Received manually_added_features: {manually_added_features}")

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
            search_results, feature_values, feature_indices = update_search_results(feature_values, feature_indices, manually_added_features)

        with gr.Row():
            with gr.Column(scale=2):
                df = gr.Dataframe(
                    headers=["Title", "Citation Count", "Year"],
                    value=search_results,
                    label="Top 10 Search Results"
                )

                feature_search = gr.Textbox(label="Search Feature Labels")
                feature_matches = gr.CheckboxGroup(label="Matching Features", choices=[])
                add_button = gr.Button("Add Selected Features")

                def search_feature_labels(search_text):
                    if not search_text:
                        return gr.CheckboxGroup(choices=[])
                    matches = [f"{f['label']} ({f['index']})" for f in feature_analysis if search_text.lower() in f['label'].lower()]
                    return gr.CheckboxGroup(choices=matches[:10])

                feature_search.change(search_feature_labels, inputs=[feature_search], outputs=[feature_matches])

                def on_add_features(selected_features, current_values, current_indices, manually_added_features):
                    if selected_features:
                        print(f"Adding selected features: {selected_features}")
                        new_indices = [int(f.split('(')[-1].strip(')')) for f in selected_features]
                        
                        # Add new indices to manually_added_features if they're not already there
                        manually_added_features = list(dict.fromkeys(manually_added_features + new_indices))
                        
                        print(f"Updated manually_added_features: {manually_added_features}")
                        
                        return gr.CheckboxGroup(value=[]), current_values, current_indices, manually_added_features
                    return gr.CheckboxGroup(value=[]), current_values, current_indices, manually_added_features


                add_button.click(
                    on_add_features,
                    inputs=[feature_matches, feature_values_state, feature_indices_state, manually_added_features_state],
                    outputs=[feature_matches, feature_values_state, feature_indices_state, manually_added_features_state]
                )

            with gr.Column(scale=1):
                update_button = gr.Button("Update Results")
                sliders = []
                print(f"\nCreating sliders:")
                print(f"feature_values: {feature_values[:10]}...")
                print(f"feature_indices: {feature_indices[:10]}...")
                for i, (value, index) in enumerate(zip(feature_values, feature_indices)):
                    print(f"Creating slider for feature {index} with value {value}")
                    feature = next((f for f in feature_analysis if f['index'] == index), None)
                    label = f"{feature['label']} ({index})" if feature else f"Feature {index}"
                    
                    # Add prefix and change color for manually added features
                    if index in manually_added_features:
                        label = f"[Custom] {label}"
                        slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=value, label=label, key=f"slider-{index}", elem_id=f"custom-slider-{index}")
                    else:
                        slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=value, label=label, key=f"slider-{index}")
                    
                    sliders.append(slider)


        def on_slider_change(*values):
            print("\nEntering on_slider_change")
            print(f"Received values: {values[:10]}...")
            
            # The last value is manually_added_features
            manually_added_features = values[-1]
            slider_values = list(values[:-1])
            
            # Reconstruct feature_indices based on the order of sliders
            reconstructed_indices = [int(slider.label.split('(')[-1].split(')')[0]) for slider in sliders]
            
            new_results, new_values, new_indices = update_search_results(slider_values, reconstructed_indices, manually_added_features)
            print(f"New feature_values after update: {new_values[:10]}...")
            print(f"New feature_indices after update: {new_indices[:10]}...")
            print("Exiting on_slider_change")
            return new_results, new_values, new_indices, manually_added_features

        update_button.click(
            on_slider_change,
            inputs=sliders + [manually_added_features_state],
            outputs=[search_results_state, feature_values_state, feature_indices_state, manually_added_features_state]
        )

        print(f"Exiting show_components")
        return [df, feature_search, feature_matches, add_button, update_button] + sliders


demo.launch()