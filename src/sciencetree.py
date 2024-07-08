import anthropic 
import yaml
from typing import List, Dict, Tuple
from vector_store import EmbeddingClient, Document, DocumentLoader
import semantic_search
import hyde
import hyde_reranking

config = yaml.safe_load(open('../config.yaml', 'r'))
anthropic_key = config['anthropic_api_key']
generation_client = anthropic.Anthropic(api_key = anthropic_key)

modes = ['Science Goal', 'Science Objective', 'Physical Parameter', 'Astronomical Observable']

class scienceTreeNode():
    def __init__(self, text, retriever, n = 2, temperature = 0.5, background = None, mode = 0, generation_model = "claude-3-5-sonnet-20240620"):
        self.text = text
        self.retriever = retriever
        self.generation_model = generation_model
        self.mode = mode
        self.n = n # branching factor
        self.temperature = temperature
        
        self.background = background
        if background is None:
            self.background = """You are an expert astronomer trying to understand the science case for a future NASA observatory."""
                            # The system will be a space-based X-ray telescope with high-resolution imaging and spectroscopy."""
            self.background += """The system will be a space-based IR/O/UV telescope with high-contrast (10-10) imaging and spectroscopy. """
        
        self.children = []
        
        if mode < 3:
            self.children = self.generate(temperature, self.n)
    
    def generate(self, temperature = 0.5, n = 2):
        docs = self.retriever.retrieve(self.text, "2401.0001", top_k = 10)
        doc_texts = self.retriever.get_document_texts(docs)
        input_text = modes[self.mode] + ": " + self.text + "\n"
        for doc in doc_texts:
            input_text += doc['id'] + ": " + doc['abstract'] + "\n"

        systems = ["""Given the following over-arching science goal and related astrophysics papers, brainstorm exactly {} focused science objectives that contribute towards the science goal.
                        Be concise. Return each science objective on a separate line, enclosed in curly braces.
                        """.format(n),
                """Given the following science objective and related astrophysics research papers, identify exactly {} astrophysical parameters that would help answer the science objective.
                        Be detailed and specific (ex. type of astrophysical systems). Return each astrophysical parameter on a separate line, enclosed in curly braces.
                        """.format(n),
                """Given the following astrophysical parameter and related research papers, identify exactly {} concrete observables that would help measure the physical parameter.
                        Ensure the observables are directly related to potential telescope observations.
                        Be concise and specific (ex. wavelength band, types of observation targets, etc.). Return each observable on a separate line, enclosed in curly braces.
                        """.format(n)]

        message = generation_client.messages.create(
                model = self.generation_model,
                max_tokens = 1000,
                temperature = temperature,
                system = self.background + "\n" + systems[self.mode],

                messages=[{ "role": "user",
                        "content": [{"type": "text", "text": input_text}] }]
            )

        message =  message.content[0].text
        

        children = []
        for pair in message.split('\n'):
            if '{' and '}' in pair:
                child = pair.split('{')[1].replace('}', '')
                children.append(scienceTreeNode(text = child, background = self.background, n = self.n, retriever = self.retriever, generation_model = self.generation_model, mode = self.mode + 1))
        
        return children
    
def generate_latex_tree(root, depth):
    if root is None:
        return ""
    latex_children = " ".join(generate_latex_tree(child, depth + 1) for child in root.children)
    if latex_children:
        return f"[{{\\node{{\\parbox{{{int(12/(depth))}cm}}{{{root.text}}}}}}} {latex_children}]"
    else:
        return f"[{{\\node{{\\parbox{{{int(12/depth)}cm}}{{{root.text}}}}}}}]"

def print_latex_tree(root):
    latex_tree = generate_latex_tree(root, depth = 1)
    latex_code = f"""
        \\documentclass{{article}}
        \\usepackage[paperheight=8.5in,paperwidth=13.0in]{{geometry}}
        \\usepackage{{tikz}}
        \\usetikzlibrary{{fit, positioning}}
        \\usepackage{{forest}}
        \\begin{{document}}
        \\centering
        \\begin{{forest}}
        for tree={{
            draw,
            rectangle,
            rounded corners,
            align=center,
            inner sep=2pt,
            anchor=north,
            fit tree
        }}
        {latex_tree}
        \\end{{forest}}
        \\end{{document}}
        """
    print(latex_code)

retrieval_mode = "semantic"

if retrieval_mode == "semantic":
    retriever = semantic_search.EmbeddingRetrievalSystem()
elif retrieval_mode == "hyde":
    retriever = hyde.HydeRetrievalSystem(config_path="../config.yaml")
elif retrieval_mode == "hydecohere":
    retriever = hyde_reranking.HydeCohereRetrievalSystem()
else:
    print("No retrieval system selected.")

tree = scienceTreeNode(text = "Map out nearby planetary systems and understand the diversity of the worlds they contain", retriever = retriever)
print_latex_tree(tree)