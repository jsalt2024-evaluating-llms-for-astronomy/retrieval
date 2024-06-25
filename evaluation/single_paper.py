import random
import os
import anthropic

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

# random_paper = random_paper()
random_paper = "/Users/charlesoneill/Desktop/~/Desktop/arxiv-data/0211/astro-ph0211072_arXiv.txt"
print(random_paper)

# Print the contents of the paper
# with open(random_paper, 'r') as file:
#     print(file.read())

# Open the contents of the paper
contents = open(random_paper, 'r').read()
introduction = contents.split("Introduction: ")[1].split("Conclusions: ")[0].strip()
conclusion = contents.split("Conclusions: ")[1].strip()
# print(f"Introduction: {introduction}\n\n")
# print('-' * 100)
# print(f"Conclusions: {conclusion}")

# Assert that both are not None
assert introduction is not None
assert conclusion is not None


# Create the prompt
prompt = """
You are an expert astronomer and astrophysicist. Given the following introduction, determine a specific question that relates to a very specific piece of information in the question i.e. a question that could only be answered by reading this specific introduction.

Here's the introduction: {introduction}

Based on this, determine one specific astronomy question. Do not refer specifically to the paper - frame it as a general astronomy question. Do not return anything other than the question.
"""

# Fill in the prompt
prompt = prompt.format(introduction=introduction)
# print(prompt)

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
print(message.content)