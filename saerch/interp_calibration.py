import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import yaml
from openai import OpenAI
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from tqdm import tqdm
from openai import AzureOpenAI

# Constants
CONFIG_PATH = Path("../config.yaml")
DATA_DIR = Path("../data")
SAE_DATA_DIR = Path("sae_data")
RESULTS_FILE = Path("experiment_results.json")

class NeuronAnalyzer:
    AUTOINTERP_PROMPT = """ 
You are a meticulous AI and astronomy researcher conducting an important investigation into a certain neuron in a language model trained on astrophysics papers. Your task is to figure out what sort of behaviour this neuron is responsible for -- namely, on what general concepts, features, topics does this neuron fire? Here's how you'll complete the task:

INPUT_DESCRIPTION: 

You will be given two inputs: 1) Max Activating Examples and 2) Zero Activating Examples.

- MAX_ACTIVATING_EXAMPLES_DESCRIPTION
You will be given several examples of astronomy paper abstracts (along with their title) that activate the neuron, along with a number being how much it was activated (these number's absolute scale is meaningless, but the relative scale may be important). This means there is some feature, topic or concept in this text that 'excites' this neuron.

You will also be given several examples of paper abstracts that doesn't activate the neuron. This means the feature, topic or concept is not present in these texts.

OUTPUT_DESCRIPTION:
Given the inputs provided, complete the following tasks. Be concise, and information dense. Don't waste a single word of reasoning.

Step 1: Based on the MAX_ACTIVATING_EXAMPLES provided, write down potential topics, concepts, and features that they share in common. These will need to be specific - remember, all of the text comes from astronomy, so these need to be highly specific astronomy concepts. You may need to look at different levels of granularity (i.e. subsets of a more general topic). List as many as you can think of. However, the only requirement is that all abstracts contain this feature.
Step 2: Based on the zero activating examples, rule out any of the topics/concepts/features listed above that are in the zero-activating examples. Systematically go through your list above.
Step 3: Based on the above two steps, perform a thorough analysis of which feature, concept or topic, at what level of granularity, is likely to activate this neuron. Use Occam's razor, the simplest explanation possible, as long as it fits the provided evidence. Opt for general concepts, features and topics. Be highly rational and analytical here.
Step 4: Based on step 4, summarise this concept in 1-8 words, in the form "FINAL: <explanation>". Do NOT return anything after this. 

Here are the max-activating examples:

{max_activating_examples}

Here are the zero-activating examples:

{zero_activating_examples}

Work through the steps thoroughly and analytically to interpret our neuron.
"""

    PREDICTION_BASE_PROMPT = """
You are an AI expert that is predicting which abstracts will activate a certain neuron in a language model trained on astrophysics papers. 
Your task is to predict which of the following abstracts will activate the neuron the most. Here's how you'll complete the task:

INPUT_DESCRIPTION:
You will be given the description of the type of paper abstracts on which the neuron activates. This description will be short.

You will then be given an abstract. Based on the concept of the abstract, you will predict whether the neuron will activate or not.

OUTPUT_DESCRIPTION:
Given the inputs provided, complete the following tasks.

Step 1: Based on the description of the type of paper abstracts on which the neuron activates, reason step by step about whether the neuron will activate on this abstract or not. Be highly rational and analytical here. The abstract may not be clear cut - it may contain topics/concepts close to the neuron description, but not exact. In this case, reason thoroughly and use your best judgement.
Step 2: Based on the above step, predict whether the neuron will activate on this abstract or not. If you predict it will activate, give a confidence score from 0 to 1 (i.e. 1 if you're certain it will activate because it contains topics/concepts that match the description exactly, 0 if you're highly uncertain). If you predict it will not activate, give a confidence score from -1 to 0.
Step 3: Provide the final prediction in the form "PREDICTION: <number>". Do NOT return anything after this.

Here is the description/interpretation of the type of paper abstracts on which the neuron activates:
{description}

Here is the abstract to predict:
{abstract}

Work through the steps thoroughly and analytically to predict whether the neuron will activate on this abstract.
"""

    def __init__(self, config_path: Path, feature_index: int, num_samples: int, interpreter_model: str, predictor_model: str):
        self.config = self.load_config(config_path)
        self.client = AzureOpenAI(
            azure_endpoint=self.config["base_url"],
            api_key=self.config["azure_api_key"],
            api_version=self.config["api_version"],
        )
        self.feature_index = feature_index
        self.num_samples = num_samples
        self.topk_indices, self.topk_values = self.load_sae_data()
        self.abstract_texts = self.load_abstract_texts()
        self.embeddings = self.load_embeddings()
        self.interpreter_model = "gpt-4o" if interpreter_model == "gpt-4o" else "gpt-35-turbo"
        self.predictor_model = "gpt-4o" if predictor_model == "gpt-4o" else "gpt-35-turbo"

    @staticmethod
    def load_config(config_path: Path) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_sae_data(self) -> Tuple[np.ndarray, np.ndarray]:
        topk_indices = np.load(SAE_DATA_DIR / "topk_indices.npy")
        topk_values = np.load(SAE_DATA_DIR / "topk_values.npy")
        return topk_indices, topk_values

    def load_abstract_texts(self) -> Dict:
        with open(DATA_DIR / "vector_store/abstract_texts.json", 'r') as f:
            return json.load(f)

    def load_embeddings(self) -> np.ndarray:
        with open(DATA_DIR / "vector_store/embeddings_matrix.npy", 'rb') as f:
            return np.load(f)

    def get_feature_activations(self, m: int, min_length: int = 100) -> Tuple[List[Tuple], List[Tuple]]:
        doc_ids = self.abstract_texts['doc_ids']
        abstracts = self.abstract_texts['abstracts']
        
        feature_mask = self.topk_indices == self.feature_index
        activated_indices = np.where(feature_mask.any(axis=1))[0]
        activation_values = np.where(feature_mask, self.topk_values, 0).max(axis=1)
        
        sorted_activated_indices = activated_indices[np.argsort(-activation_values[activated_indices])]
        
        top_m_abstracts = []
        top_m_indices = []
        for i in sorted_activated_indices:
            if len(abstracts[i]) > min_length:
                top_m_abstracts.append((doc_ids[i], abstracts[i], activation_values[i]))
                top_m_indices.append(i)
            if len(top_m_abstracts) == m:
                break
        
        zero_activation_indices = np.where(~feature_mask.any(axis=1))[0]
        zero_activation_samples = []
        
        active_embedding = np.array([self.embeddings[i] for i in top_m_indices]).mean(axis = 0)  
        cosine_similarities = np.dot(active_embedding, self.embeddings[zero_activation_indices].T)
        cosine_pairs = [(index, cosine_similarities[i]) for i, index in enumerate(zero_activation_indices)]
        cosine_pairs.sort(key=lambda x: -x[1])
        
        for i, cosine_sim in cosine_pairs:
            if len(abstracts[i]) > min_length:
                zero_activation_samples.append((doc_ids[i], abstracts[i], 0))
            if len(zero_activation_samples) == m:
                break
        
        return top_m_abstracts, zero_activation_samples

    def generate_interpretation(self, top_abstracts: List[Tuple], zero_abstracts: List[Tuple]) -> str:
        max_activating_examples = "\n\n------------------------\n".join([f"Activation:{activation:.3f}\n{abstract}" for _, abstract, activation in top_abstracts])
        zero_activating_examples = "\n\n------------------------\n".join([abstract for _, abstract, _ in zero_abstracts])
        
        prompt = self.AUTOINTERP_PROMPT.format(
            max_activating_examples=max_activating_examples,
            zero_activating_examples=zero_activating_examples
        )
        
        response = self.client.chat.completions.create(
            model=self.interpreter_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        text_response = response.choices[0].message.content
        logging.info(f"Interpretation for feature {self.feature_index}:\n{text_response.split('FINAL:')[1].strip()}")
        
        return text_response.split("FINAL:")[1].strip()

    def predict_activations(self, interpretation: str, abstracts: List[str]) -> List[float]:
        predictions = []
        
        for abstract in tqdm(abstracts):
            prompt = self.PREDICTION_BASE_PROMPT.format(description=interpretation, abstract=abstract)
            response = self.client.chat.completions.create(
                model=self.predictor_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            response_text = response.choices[0].message.content
            try:
                prediction = response_text.split("PREDICTION:")[1].strip()
                predictions.append(float(prediction.replace("*", "")))
            except Exception as e:
                logging.error(f"Error predicting activation: {e}")
                predictions.append(0.0)
        
        return predictions

    @staticmethod
    def evaluate_predictions(ground_truth: List[int], predictions: List[float]) -> Tuple[float, float]:
        correlation, _ = pearsonr(ground_truth, predictions)
        binary_predictions = [1 if p > 0 else 0 for p in predictions]
        f1 = f1_score(ground_truth, binary_predictions)
        return correlation, f1

def run_experiment(feature_index: int, num_samples: int, interpreter_model: str, predictor_model: str, cached_interpretation: str = None) -> Dict:
    analyzer = NeuronAnalyzer(CONFIG_PATH, feature_index, num_samples, interpreter_model, predictor_model)
    
    top_abstracts, zero_abstracts = analyzer.get_feature_activations(num_samples)
    
    if interpreter_model == "gpt-4o" and cached_interpretation:
        interpretation = cached_interpretation
    else:
        interpretation = analyzer.generate_interpretation(top_abstracts, zero_abstracts)
    
    num_test_samples = 4
    test_abstracts = [abstract for _, abstract, _ in top_abstracts[-num_test_samples:] + zero_abstracts[-num_test_samples:]]
    ground_truth = [1] * num_test_samples + [0] * num_test_samples
    
    predictions = analyzer.predict_activations(interpretation, test_abstracts)
    correlation, f1 = analyzer.evaluate_predictions(ground_truth, predictions)
    
    return {
        "feature_index": feature_index,
        "interpreter_model": interpreter_model,
        "predictor_model": predictor_model,
        "interpretation": interpretation,
        "correlation": correlation,
        "f1_score": f1
    }

def load_existing_results() -> List[Dict]:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_results(results: List[Dict]):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    start = time.time()
    logging.basicConfig(level=logging.INFO)
    
    existing_results = load_existing_results()
    processed_features = set((result['feature_index'], result['interpreter_model'], result['predictor_model']) 
                             for result in existing_results)
    
    new_results = []
    model_combinations = [
        ("gpt-4o", "gpt-4o"),
        ("gpt-4o", "gpt-3.5-turbo"),
        ("gpt-3.5-turbo", "gpt-3.5-turbo")
    ]
    
    for feature_index in range(10, 16):
        cached_interpretation = None
        for interpreter_model, predictor_model in model_combinations:
            if (feature_index, interpreter_model, predictor_model) in processed_features:
                logging.info(f"Skipping feature {feature_index} with {interpreter_model} interpreter and {predictor_model} predictor (already processed)")
                continue
            
            logging.info(f"Running experiment for feature {feature_index} with {interpreter_model} interpreter and {predictor_model} predictor")
            
            experiment_result = run_experiment(feature_index, 10, interpreter_model, predictor_model, cached_interpretation)
            new_results.append(experiment_result)
            
            if interpreter_model == "gpt-4o" and cached_interpretation is None:
                cached_interpretation = experiment_result["interpretation"]
            
            logging.info(f"Feature {feature_index}, {interpreter_model} interpreter, {predictor_model} predictor:")
            logging.info(f"  Interpretation: {experiment_result['interpretation']}")
            logging.info(f"  Pearson correlation: {experiment_result['correlation']}")
            logging.info(f"  F1 score: {experiment_result['f1_score']}")
    
    # Combine existing and new results
    all_results = existing_results + new_results
    
    # Save all results to JSON file
    save_results(all_results)
    
    logging.info(f"Results saved to {RESULTS_FILE}")
    logging.info(f"Total time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()