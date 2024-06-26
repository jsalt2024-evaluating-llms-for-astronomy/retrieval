import random
import anthropic
import yaml
import json
from datasets import load_dataset
from tqdm.auto import tqdm

# Load the Hugging Face dataset with streaming and shuffle it
dataset_path = "charlieoneill/jsalt-astroph-dataset"
dataset = load_dataset(dataset_path, split="train", streaming=True)
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=1_000)

# Create the prompt
prompt = """
You are an expert astronomer and astrophysicist. Given the following part of the paper, determine a specific question that relates to a specific piece of information in the question i.e. a question that could only be answered by reading this specific part of the paper.

Here's the paper section: {introduction}

Based on this, determine one specific astronomy question. Try and make the question specific, but general/abstract enough that you're not just rehashing it word-for-word from the paper i.e. don't overly quantify, and keep it one part (don't split it into a first part AND a second part). Do not refer specifically to the paper - frame it as a general astronomy question. Do not return anything other than the question.
"""

# Load API key and config
config = yaml.safe_load(open('../config.yaml', 'r'))
API_KEY = config['anthropic_api_key']
NUM_QUESTIONS = 100
question_answer_dict = {}

def claude(prompt, **kwargs):
    client = anthropic.Anthropic(api_key=API_KEY)
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ],
        **kwargs,
    )
    return message.content[0].text

# Generate questions
shuffled_dataset_iterator = iter(shuffled_dataset)
for i in tqdm(range(NUM_QUESTIONS)):

    introduction = ''
    conclusion = ''
    folder_file = ''

    while len(introduction) < 500 and len(conclusion) < 500: #folder_file not in question_answer_dict.keys():
        paper = next(shuffled_dataset_iterator)
        introduction = paper['introduction']
        conclusion = paper['conclusions']
        folder_file = f"{paper['subfolder']}/{paper['filename']}"

    # Format prompts
    formatted_prompt_intro = prompt.format(introduction=introduction)
    question_intro = claude(formatted_prompt_intro)
    formatted_prompt_conclusion = prompt.format(introduction=conclusion) 
    question_conclusion = claude(formatted_prompt_conclusion)

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
with open('../data/single_paper.json', 'w') as f:
    json.dump(question_answer_dict, f, indent=4)

print("Questions generated and saved to '../data/single_paper.json'")
