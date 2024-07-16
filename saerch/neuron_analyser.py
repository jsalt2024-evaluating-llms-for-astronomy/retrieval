import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import concurrent.futures

import numpy as np
import yaml
from openai import AzureOpenAI
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from tqdm import tqdm
import tenacity

# Constants
CONFIG_PATH = Path("../config.yaml")
DATA_DIR = Path("../data")
SAE_DATA_DIR = Path("sae_data")
OUTPUT_FILE = Path("feature_analysis_results_16.json")
SAVE_INTERVAL = 10

@dataclass
class Feature:
    index: int
    label: str
    reasoning: str
    f1: float
    pearson_correlation: float
    density: float

class BatchNeuronAnalyzer:
    AUTOINTERP_PROMPT = """ 
You are a meticulous AI and astronomy researcher conducting an important investigation into a certain neuron in a language model trained on astrophysics papers. Your task is to figure out what sort of behaviour this neuron is responsible for -- namely, on what general concepts, features, topics does this neuron fire? Here's how you'll complete the task:

INPUT_DESCRIPTION: 

You will be given two inputs: 1) Max Activating Examples and 2) Zero Activating Examples.

- MAX_ACTIVATING_EXAMPLES_DESCRIPTION
You will be given several examples of astronomy paper abstracts (along with their title) that activate the neuron, along with a number being how much it was activated (these number's absolute scale is meaningless, but the relative scale may be important). This means there is some feature, topic or concept in this text that 'excites' this neuron.

You will also be given several examples of paper abstracts that doesn't activate the neuron. This means the feature, topic or concept is not present in these texts.

OUTPUT_DESCRIPTION:
Given the inputs provided, complete the following tasks. Be concise, and information dense. Don't waste a single word of reasoning.

Step 1: Based on the MAX_ACTIVATING_EXAMPLES provided, write down potential topics, concepts, and features that all examples share in common. These can be abstract but will need to be specific - remember, all of the text comes from astronomy, so these need to be highly specific astronomy concepts. You may need to look at different levels of granularity (i.e. subsets of a more general topic). List as many as you can think of. However, the only requirement is that all abstracts contain this feature.
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

    def __init__(self, config_path: Path):
        self.config = self.load_config(config_path)
        self.client = AzureOpenAI(
            azure_endpoint=self.config["base_url"],
            api_key=self.config["azure_api_key"],
            api_version=self.config["api_version"],
        )
        self.topk_indices, self.topk_values = self.load_sae_data()
        self.abstract_texts = self.load_abstract_texts()
        self.embeddings = self.load_embeddings()

    @staticmethod
    def load_config(config_path: Path) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_sae_data(self) -> Tuple[np.ndarray, np.ndarray]:
        topk_indices = np.load(SAE_DATA_DIR / "topk_indices_16.npy")
        topk_values = np.load(SAE_DATA_DIR / "topk_values_16.npy")
        return topk_indices, topk_values

    def load_abstract_texts(self) -> Dict:
        with open(DATA_DIR / "vector_store/abstract_texts.json", 'r') as f:
            return json.load(f)

    def load_embeddings(self) -> np.ndarray:
        with open(DATA_DIR / "vector_store/embeddings_matrix.npy", 'rb') as f:
            return np.load(f)

    def get_feature_activations(self, feature_index: int, m: int, min_length: int = 100) -> Tuple[List[Tuple], List[Tuple]]:
        doc_ids = self.abstract_texts['doc_ids']
        abstracts = self.abstract_texts['abstracts']
        
        feature_mask = self.topk_indices == feature_index
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
        
        active_embedding = np.array([self.embeddings[i] for i in top_m_indices]).mean(axis=0)
        cosine_similarities = np.dot(active_embedding, self.embeddings[zero_activation_indices].T)
        cosine_pairs = [(index, cosine_similarities[i]) for i, index in enumerate(zero_activation_indices)]
        cosine_pairs.sort(key=lambda x: -x[1])
        
        for i, cosine_sim in cosine_pairs:
            if len(abstracts[i]) > min_length:
                zero_activation_samples.append((doc_ids[i], abstracts[i], 0))
            if len(zero_activation_samples) == m:
                break
        
        return top_m_abstracts, zero_activation_samples

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type(Exception)
    )
    def generate_interpretation(self, top_abstracts: List[Tuple], zero_abstracts: List[Tuple]) -> str:
        max_activating_examples = "\n\n------------------------\n".join([f"Activation:{activation:.3f}\n{abstract}" for _, abstract, activation in top_abstracts])
        zero_activating_examples = "\n\n------------------------\n".join([abstract for _, abstract, _ in zero_abstracts])
        
        prompt = self.AUTOINTERP_PROMPT.format(
            max_activating_examples=max_activating_examples,
            zero_activating_examples=zero_activating_examples
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        
        return response.choices[0].message.content

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type(Exception)
    )
    def predict_activation(self, interpretation: str, abstract: str) -> float:
        prompt = self.PREDICTION_BASE_PROMPT.format(description=interpretation, abstract=abstract)
        response = self.client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        response_text = response.choices[0].message.content
        try:
            prediction = response_text.split("PREDICTION:")[1].strip()
            return float(prediction.replace("*", ""))
        except Exception:
            return 0.0

    def predict_activations(self, interpretation: str, abstracts: List[str]) -> List[float]:
        return [self.predict_activation(interpretation, abstract) for abstract in abstracts]

    @staticmethod
    def evaluate_predictions(ground_truth: List[int], predictions: List[float]) -> Tuple[float, float]:
        correlation, _ = pearsonr(ground_truth, predictions)
        binary_predictions = [1 if p > 0 else 0 for p in predictions]
        f1 = f1_score(ground_truth, binary_predictions)
        return correlation, f1

    def analyze_feature(self, feature_index: int, num_samples: int) -> Feature:
        top_abstracts, zero_abstracts = self.get_feature_activations(feature_index, num_samples)
        interpretation_full = self.generate_interpretation(top_abstracts, zero_abstracts)
        interpretation = interpretation_full.split("FINAL:")[1].strip()

        num_test_samples = 3
        test_abstracts = [abstract for _, abstract, _ in top_abstracts[-num_test_samples:] + zero_abstracts[-num_test_samples:]]
        ground_truth = [1] * num_test_samples + [0] * num_test_samples

        predictions = self.predict_activations(interpretation, test_abstracts)
        correlation, f1 = self.evaluate_predictions(ground_truth, predictions)

        density = (self.topk_indices == feature_index).any(axis=1).mean()

        return Feature(
            index=feature_index,
            label=interpretation,
            reasoning=interpretation_full,
            f1=f1,
            pearson_correlation=correlation,
            density=density
        )

def save_results(results: List[Dict], filename: Path):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(filename: Path) -> List[Dict]:
    if filename.exists():
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def main():
    analyzer = BatchNeuronAnalyzer(CONFIG_PATH)

    num_features = 3072
    num_samples = 10

    # Load existing results and determine the starting point
    results = load_results(OUTPUT_FILE)
    start_index = max([feature['index'] for feature in results], default=-1) + 1
    print(f"Starting analysis from feature {start_index}...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {executor.submit(analyzer.analyze_feature, i, num_samples): i 
                           for i in range(start_index, num_features)}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                           total=num_features - start_index, 
                           desc="Analysing features"):
            feature_index = future_to_index[future]
            try:
                feature = future.result()
                results.append(asdict(feature))
                
                # Save checkpoint
                if len(results) % SAVE_INTERVAL == 0:
                    save_results(results, OUTPUT_FILE)
                    print(f"Checkpoint saved. Processed {len(results)} features.")
                
            except Exception as exc:
                print(f"Feature {feature_index} generated an exception: {exc}")

    save_results(results, OUTPUT_FILE)
    print(f"Analysis complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()