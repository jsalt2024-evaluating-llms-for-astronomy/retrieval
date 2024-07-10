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

# Constants
CONFIG_PATH = Path("../config.yaml")
DATA_DIR = Path("../data")
SAE_DATA_DIR = Path("sae_data")

class NeuronAnalyzer:
    AUTOINTERP_PROMPT = """ 
You are a meticulous AI and astronomy researcher conducting an important investigation into a certain neuron in a language model trained on astrophysics papers. Your task is to figure out what sort of behaviour this neuron is responsible for -- namely, on what general concepts, features, topics does this neuron fire? Here's how you'll complete the task:

INPUT_DESCRIPTION: 

You will be given two inputs: 1) Max Activating Examples and 2) Zero Activating Examples.

- MAX_ACTIVATING_EXAMPLES_DESCRIPTION
You will be given several examples of text that activate the neuron, along with a number being how much it was activated (these number's absolute scale is meaningless, but the relative scale may be important). This means there is some feature, topic or concept in this text that 'excites' this neuron.

You will also be given several examples of text that doesn't activate the neuron. This means the feature, topic or concept is not present in these texts.

OUTPUT_DESCRIPTION:
Given the inputs provided, complete the following tasks. Be concise, and information dense. Don't waste a single word of reasoning.

Step 1: Based on the MAX_ACTIVATING_EXAMPLES provided, write down potential topics, concepts, and features that they share in common. These will need to be specific - remember, all of the text comes from astronomy, so these need to be highly specific astronomy concepts. You may need to look at different levels of granularity (i.e. subsets of a more general topic). List as many as you can think of. Give higher weight to concepts more present/prominent in examples with higher activations.
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

    def __init__(self, config_path: Path, feature_index: int, num_samples: int):
        self.config = self.load_config(config_path)
        self.client = OpenAI(api_key=self.config['openai_api_key'])
        self.feature_index = feature_index
        self.num_samples = num_samples
        self.topk_indices, self.topk_values = self.load_sae_data()
        self.abstract_texts = self.load_abstract_texts()

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

    def get_feature_activations(self, m: int, min_length: int = 100) -> Tuple[List[Tuple], List[Tuple]]:
        doc_ids = self.abstract_texts['doc_ids']
        abstracts = self.abstract_texts['abstracts']
        
        feature_mask = self.topk_indices == self.feature_index
        activated_indices = np.where(feature_mask.any(axis=1))[0]
        activation_values = np.where(feature_mask, self.topk_values, 0).max(axis=1)
        
        sorted_activated_indices = activated_indices[np.argsort(-activation_values[activated_indices])]
        
        top_m_abstracts = []
        for i in sorted_activated_indices:
            if len(abstracts[i]) > min_length:
                top_m_abstracts.append((doc_ids[i], abstracts[i], activation_values[i]))
            if len(top_m_abstracts) == m:
                break
        
        zero_activation_indices = np.where(~feature_mask.any(axis=1))[0]
        zero_activation_samples = []
        np.random.shuffle(zero_activation_indices)
        for i in zero_activation_indices:
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
            model="gpt-4o", #"gpt-3.5-turbo", #"gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        
        return response.choices[0].message.content.split("FINAL:")[1].strip()

    def predict_activations(self, interpretation: str, abstracts: List[str]) -> List[float]:
        predictions = []
        
        for abstract in tqdm(abstracts):
            prompt = self.PREDICTION_BASE_PROMPT.format(description=interpretation, abstract=abstract)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", #"gpt-4o"
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

def main(feature_index: int, num_samples: int):
    start = time.time()
    logging.basicConfig(level=logging.INFO)
    analyzer = NeuronAnalyzer(CONFIG_PATH, feature_index, num_samples)
    
    top_abstracts, zero_abstracts = analyzer.get_feature_activations(num_samples)
    interpretation = analyzer.generate_interpretation(top_abstracts, zero_abstracts)
    logging.info(f"Interpretation: {interpretation}")
    
    num_test_samples = 4
    test_abstracts = [abstract for _, abstract, _ in top_abstracts[-num_test_samples:] + zero_abstracts[-num_test_samples:]]
    ground_truth = [1] * num_test_samples + [0] * num_test_samples
    
    predictions = analyzer.predict_activations(interpretation, test_abstracts)
    correlation, f1 = analyzer.evaluate_predictions(ground_truth, predictions)
    
    logging.info(f"Pearson correlation: {correlation}")
    logging.info(f"F1 score: {f1}")

    logging.info(f"Time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze neuron activations in astrophysics papers.")
    parser.add_argument("feature_index", type=int, help="Index of the feature to analyze")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of samples to use for analysis")
    args = parser.parse_args()
    
    main(args.feature_index, args.num_samples)