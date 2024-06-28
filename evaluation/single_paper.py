import random
import anthropic
import openai
import yaml
import json
import time
from datasets import load_dataset
from tqdm.auto import tqdm

# Load the Hugging Face dataset
dataset_path = "charlieoneill/jsalt-astroph-dataset"
dataset = load_dataset(dataset_path, split="train")

# Create the prompts
abstract_prompt = """
You are an expert astronomer and astrophysicist. Given the following abstract of a paper, determine a specific question that relates to a specific piece of information in the abstract i.e. a question that could only be answered by reading this specific part of the paper.

Here's the paper abstract: {abstract}

Based on this, determine one specific astronomy question. Try and make the question specific, but general/abstract enough that you're not just rehashing it word-for-word from the paper i.e. don't overly quantify, and keep it one part (don't split it into a first part AND a second part). Do not refer specifically to the paper - frame it as a general astronomy question. Do not return anything other than the question.
"""

conclusion_prompt = """
You are an expert astronomer and astrophysicist. Given the following conclusion of a paper, determine a specific question that relates to a specific piece of information in the conclusion i.e. a question that could only be answered by reading this specific part of the paper.

Here's the paper conclusion: {conclusion}

Based on this, determine one specific astronomy question. Try and make the question specific, but general/abstract enough that you're not just rehashing it word-for-word from the paper i.e. don't overly quantify, and keep it one part (don't split it into a first part AND a second part). Do not refer specifically to the paper - frame it as a general astronomy question. Do not return anything other than the question.
"""

# Load API keys and config
config = yaml.safe_load(open('../config.yaml', 'r'))
anthropic_api_key = config['anthropic_api_key']
openai_api_key = config['openai_api_key']
NUM_QUESTIONS = 250
question_answer_dict = {}
MAX_RETRIES = 5
RETRY_DELAY = 20  # seconds

def retry_on_failure(max_retries):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error: {e}. Retrying in {RETRY_DELAY} seconds...")
                    retries += 1
                    time.sleep(RETRY_DELAY)
            raise Exception("Max retries exceeded")
        return wrapper
    return decorator

@retry_on_failure(MAX_RETRIES)
def get_claude_response(prompt, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

@retry_on_failure(MAX_RETRIES)
def get_gpt4_response(prompt, api_key):
    openai.api_key = api_key
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125", #"gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_questions(model):
    # Filter function to remove papers without sufficient abstract or conclusions
    def filter_papers(example):
        return len(example.get('abstract', '')) > 100 and len(example.get('conclusions', '')) > 100

    # Filter and shuffle the dataset
    filtered_dataset = dataset.filter(filter_papers)
    shuffled_dataset = filtered_dataset.shuffle(seed=42)

    # Generate questions
    for i in tqdm(range(NUM_QUESTIONS)):
        paper = shuffled_dataset[i % len(shuffled_dataset)]
        abstract = paper['abstract']
        conclusion = paper['conclusions']
        folder_file = paper['arxiv_id'] #f"{paper['subfolder']}/{paper['filename']}"

        # Alternate between abstract and conclusion
        if i % 2 == 0:
            formatted_prompt = abstract_prompt.format(abstract=abstract)
            question_type = 'abstract'
        else:
            formatted_prompt = conclusion_prompt.format(conclusion=conclusion)
            question_type = 'conclusion'

        if model == 'claude':
            question = get_claude_response(formatted_prompt, anthropic_api_key)
        elif model == 'gpt-4':
            question = get_gpt4_response(formatted_prompt, openai_api_key)
        else:
            raise ValueError("Invalid model selected. Choose either 'claude' or 'gpt-4'.")

        if folder_file not in question_answer_dict:
            question_answer_dict[folder_file] = {}
        question_answer_dict[folder_file][f'question_{question_type}'] = question

        # Print everything
        print(folder_file)
        print('-' * 100)
        print(f"Question from {question_type}:")
        print(question)
        print('-' * 100)
        print('\n')

# Save the dictionary as a JSON
def save_results():
    with open('../data/abstract_conclusion_questions.json', 'w') as f:
        json.dump(question_answer_dict, f, indent=4)
    print("Questions generated and saved to '../data/abstract_conclusion_questions.json'")

# Set the model to use ('claude' or 'gpt-4')
model_to_use = 'claude'

generate_questions(model_to_use)
save_results()