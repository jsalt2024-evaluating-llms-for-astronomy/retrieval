import gradio as gr
import numpy as np
import json
from openai import OpenAI
import yaml
from typing import Optional

# Load configuration and initialize OpenAI client
config = yaml.safe_load(open('../config.yaml', 'r'))
client = OpenAI(api_key=config['openai_api_key'])

EMBEDDING_MODEL = "text-embedding-3-small"

# Load pre-computed embeddings and texts
embeddings_path = "../data/vector_store/abstract_embeddings.npy"
texts_path = "../data/vector_store/abstract_texts.json"
feature_analysis_path = "sae_data/feature_analysis_results.json"

abstract_embeddings = np.load(embeddings_path)
with open(texts_path, 'r') as f:
    abstract_texts = json.load(f)
with open(feature_analysis_path, 'r') as f:
    feature_analysis = json.load(f)

def get_embedding(text: Optional[str], model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
    try:
        embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def process_query(query):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return ["Error: Unable to generate query embedding"]

    sims = np.dot(abstract_embeddings, query_embedding)
    topk_indices = np.argsort(sims)[::-1][:10]  # Get top 10 results

    doc_ids = abstract_texts['doc_ids']
    topk_doc_ids = [doc_ids[i] for i in topk_indices]

    return topk_doc_ids

def create_app():
    with gr.Blocks() as app:
        gr.Markdown("# Semantic Search with Sparse Autoencoder")

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="Enter your query")
                search_button = gr.Button("Search")
                results_output = gr.JSON(label="Top 10 Document IDs")

            with gr.Column(scale=1):
                with gr.Accordion("Adjust Dimensions", open=False):
                    dimension_search = gr.Textbox(label="Search dimensions")
                    sliders = []
                    for feature in feature_analysis:
                        slider = gr.Slider(
                            minimum=-1, 
                            maximum=1, 
                            step=0.01, 
                            label=feature['label'],
                            info=f"Index: {feature['index']}"
                        )
                        sliders.append(slider)

        search_button.click(process_query, inputs=query_input, outputs=results_output)

        # Placeholder for slider functionality
        for slider in sliders:
            slider.change(
                lambda x: x,  # Placeholder function
                inputs=slider,
                outputs=slider
            )

    return app

# Create and launch the app
if __name__ == "__main__":
    app = create_app()
    app.launch()