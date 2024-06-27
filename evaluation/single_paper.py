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

# Create the prompt
prompt = """
You are an expert astronomer and astrophysicist. Given the following part of the paper, determine a specific question that relates to a specific piece of information in the question i.e. a question that could only be answered by reading this specific part of the paper.

Here's the paper section: {introduction}

Based on this, determine one specific astronomy question. Try and make the question specific, but general/abstract enough that you're not just rehashing it word-for-word from the paper i.e. don't overly quantify, and keep it one part (don't split it into a first part AND a second part). Do not refer specifically to the paper - frame it as a general astronomy question. Do not return anything other than the question.
"""

# Load API keys and config
config = yaml.safe_load(open('../config.yaml', 'r'))
anthropic_api_key = config['anthropic_api_key']
openai_api_key = config['openai_api_key']
NUM_QUESTIONS = 100
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
    return message['completion']

@retry_on_failure(MAX_RETRIES)
def get_gpt4_response(prompt, api_key):
    openai.api_key = api_key
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125", #"gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_questions(model):
    # Filter function to remove papers without sufficient introduction or conclusion
    def filter_papers(example):
        return len(example.get('introduction', '')) > 0 and len(example.get('conclusions', '')) > 0

    for year in [10, 15, 20, 23]:
        print(f"Year: {year}")  
        dataset_year = dataset.filter(lambda x: x['year'] == year)
        dataset_year = dataset_year.filter(filter_papers)
        shuffled_dataset = dataset_year.shuffle(seed=41)

        num_questions_year = int(NUM_QUESTIONS / 5)

        # Generate questions
        shuffled_dataset_iterator = iter(shuffled_dataset)
        for i in tqdm(range(num_questions_year)):

            paper = next(shuffled_dataset_iterator)
            introduction = paper['introduction']
            conclusion = paper['conclusions']
            folder_file = f"{paper['subfolder']}/{paper['filename']}"

            # Format prompts
            formatted_prompt_intro = prompt.format(introduction=introduction)
            formatted_prompt_conclusion = prompt.format(introduction=conclusion)

            if model == 'claude':
                question_intro = get_claude_response(formatted_prompt_intro, anthropic_api_key)
                question_conclusion = get_claude_response(formatted_prompt_conclusion, anthropic_api_key)
            elif model == 'gpt-4':
                question_intro = get_gpt4_response(formatted_prompt_intro, openai_api_key)
                question_conclusion = get_gpt4_response(formatted_prompt_conclusion, openai_api_key)
            else:
                raise ValueError("Invalid model selected. Choose either 'claude' or 'gpt-4'.")

            question_answer_dict[folder_file] = {'question_intro': question_intro, 'question_conclusion': question_conclusion}

            # Print everything
            print(folder_file)
            print('-' * 100)
            print(question_intro)
            print('\n\n\n')
            print(question_conclusion)
            print('-' * 100)
            print('\n\n\n')

# Save the dictionary as a JSON
def save_results():
    with open('../data/single_paper.json', 'w') as f:
        json.dump(question_answer_dict, f, indent=4)
    print("Questions generated and saved to '../data/single_paper.json'")

# Set the model to use ('claude' or 'gpt-4')
model_to_use = 'gpt-4'  # Change to 'claude' to use Claude model

generate_questions(model_to_use)
save_results()
