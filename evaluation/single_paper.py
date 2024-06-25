import random
import os
import anthropic
import yaml

path = "/Users/charlesoneill/Desktop/~/Desktop/arxiv-data"

# Print a random paper
def random_paper():
    papers = os.listdir(path)
    random_folder = random.choice(papers)
    # Randomly choose a text file in the folder
    random_paper = random.choice(os.listdir(os.path.join(path, random_folder)))
    # Join with path
    random_paper = os.path.join(path, random_folder, random_paper)
    return random_paper


# Create the prompt
prompt = """
You are an expert astronomer and astrophysicist. Given the following introduction, determine a specific question that relates to a very specific piece of information in the question i.e. a question that could only be answered by reading this specific introduction.

Here's the introduction: {introduction}

Based on this, determine one specific astronomy question. Do not refer specifically to the paper - frame it as a general astronomy question. Do not return anything other than the question.
"""

def claude(prompt, **kwargs):
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key="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",
    )
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text


# Create the dataset
config = yaml.safe_load(open('../config.yaml', 'r'))
NUM_QUESTIONS = 100
API_KEY = config['anthropic_api_key']
question_answer_dict = {}

for i in range(NUM_QUESTIONS):

    introduction = ''

    while len(introduction) < 500:
        rp = random_paper()
        contents = open(rp, 'r').read()
        introduction = contents.split("Introduction: ")[1].split("Conclusions: ")[0].strip()

    # Get random paper folder and file
    folder_file = rp.split('/')[-2:]
    # Join them into one string
    folder_file = '/'.join(folder_file)
    # print(folder_file)
    # break

        
    formatted_prompt = prompt.format(introduction=introduction)
    question = claude(formatted_prompt)

    question_answer_dict[folder_file] = {'question': question, 'introduction': introduction}

    # Print everything
    print(folder_file)
    print(question)


    break




