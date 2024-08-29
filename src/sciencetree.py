import anthropic 
import yaml
from typing import List, Dict, Tuple
from vector_store import EmbeddingClient, Document, DocumentLoader
import semantic_search
import hyde
import gradio as gr
import os
import hyde_reranking
import tempfile
import json
from openai import OpenAI

config = yaml.safe_load(open('../config.yaml', 'r'))
anthropic_key = config['anthropic_api_key']
generation_client = anthropic.Anthropic(api_key = anthropic_key)

modes = ['Science Goal', 'Science Objective', 'Physical Parameter', 'Astronomical Observable']

class scienceTreeNode():
    def __init__(self, text, year, retriever, n = 2, temperature = 0.5, background = None, experiment = None, mode = 0, generation_model = "claude-3-5-sonnet-20240620"):
        self.text = text
        self.docs = []
        self.year = year
        self.retriever = retriever
        self.generation_model = generation_model
        self.mode = mode
        self.n = n # branching factor
        self.temperature = temperature
        
        self.background = background
        if background is None:
            self.background = """You are an expert astronomer trying to understand the science case for a future observatory."""
                            # The system will be a space-based X-ray telescope with high-resolution imaging and spectroscopy."""
                            # The system will be a space-based IR/O/UV telescope with high-contrast (10-10) imaging and spectroscopy. """
            
            if experiment is not None:
                self.background += experiment
        
        self.children = []
        
        if mode < 3:
            self.children = self.generate(temperature, self.n)
    
    def generate(self, temperature = 0.5, n = 2):
        yearstr = str(self.year // 100) + "01.0001"
        docs = self.retriever.retrieve(self.text, yearstr, top_k = 10)
        doc_texts = self.retriever.get_document_texts(docs)
        self.docs = [doc['id'] for doc in doc_texts]
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
                        Be concise and specific (ex. wavelength band, resolution, precision, types of observation targets, etc.). Be as quantitative as possible, but make sure numbers are derived from the research papers. Return each observable on a separate line, enclosed in curly braces.
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
                print('Generated child at depth', self.mode + 1)
                children.append(scienceTreeNode(text = child, year = self.year, background = self.background, n = self.n, retriever = self.retriever, generation_model = self.generation_model, mode = self.mode + 1))
        
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

# retrieval_mode = "semantic"

# if retrieval_mode == "semantic":
#     retriever = semantic_search.EmbeddingRetrievalSystem()
# elif retrieval_mode == "hyde":
#     retriever = hyde.HydeRetrievalSystem(config_path="../config.yaml")
# elif retrieval_mode == "hydecohere":
#     retriever = hyde_reranking.HydeCohereRetrievalSystem()
# else:
#     print("No retrieval system selected.")

#tree = scienceTreeNode(text = "Map out nearby planetary systems and understand the diversity of the worlds they contain", retriever = retriever)
#print_latex_tree(tree)
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

\\vspace{{2cm}}

\\begin{{minipage}}{{0.9\\textwidth}}
\\textbf{{Retrieved Documents:}}
\\begin{{enumerate}}
{generate_retrieved_docs_latex(root)}
\\end{{enumerate}}
\\end{{minipage}}

\\end{{document}}
        """
    print(latex_code)

def generate_retrieved_docs_latex(node):
    latex_docs = ""
    if hasattr(node, 'docs') and node.docs:
        for doc in node.docs:
            latex_docs += f"\\item {doc}\\n"
    for child in node.children:
        latex_docs += generate_retrieved_docs_latex(child)
    return latex_docs

retriever = semantic_search.EmbeddingRetrievalSystem()

def generate_science_tree(science_objective, openai_api_key, anthropic_api_key, branching_factor, year_cutoff):
    print(f"Received inputs: {science_objective}, {openai_api_key}, {anthropic_api_key}, {branching_factor}, {year_cutoff}")
    # Set API keys
    try:
        retriever.client = EmbeddingClient(OpenAI(api_key=openai_api_key))
    except Exception as e:
        print(e)
        print("Unable to connect to OpenAI API.")
    
    try:
        generation_client = anthropic.Anthropic(api_key=anthropic_api_key)
    except:
        print("Unable to connect to Anthropic API.")

    # Generate the science tree
    tree = scienceTreeNode(text=science_objective, year=year_cutoff, retriever=retriever, n=branching_factor)
    tree.generate(temperature=0.5, n=branching_factor)

    # Function to recursively build the tree structure
    def build_tree_structure(node):
        result = {
            "text": node.text,
            "papers": [format_paper_link(paper) for paper in (node.docs if hasattr(node, 'docs') else [])]
        }
        if node.children:
            result["children"] = [build_tree_structure(child) for child in node.children]
        return result

    # Convert the tree to a dictionary structure
    tree_structure = build_tree_structure(tree)

    return tree_structure

def format_paper_link(paper_id):
    if '.txt' in paper_id:
        arxiv_id = paper_id.split('_')[0].split('astro-ph')[-1]
        return f"https://arxiv.org/abs/astro-ph/{arxiv_id}"
    else:
        return f"https://arxiv.org/abs/{paper_id}"

def save_tree_to_file(tree):
    if tree is None:
        print("No tree to save.")
        return None
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        json.dump(tree, temp_file)
    return temp_file.name

def load_tree_from_file(file):
    if file is None:
        return None
    with open(file.name, 'r') as f:
        return json.load(f)
def create_tree_html(tree):
    if tree is None:
        return ""
    
    html = "<ul>"
    html += "<li>"
    
    # Create the dropdown with <details> and <summary>
    html += f"<details><summary>{tree['text']}</summary>"
    
    # Add papers if they exist
    if tree['papers']:
        html += "<p>"
        paper_links = [f"<a href='{paper}' target='_blank'>{paper.split('/')[-1]}</a>" for paper in tree['papers']]
        html += ", ".join(paper_links)
        html += "</p>"
    
    # Recursively add children
    if 'children' in tree and tree['children']:
        html += "<ul>"
        for child in tree['children']:
            html += create_tree_html(child)
        html += "</ul>"
    
    # Close the details tag
    html += "</details>"
    
    html += "</li></ul>"
    return html
css = """
uul, details, summary, p {
    font-size: 1em; /* Set a uniform font size for all elements */
    margin: 0;
    padding: 0;
}

ul {
    list-style-type: none;
    padding-left: 20px;
}

details summary {
    cursor: pointer;
    font-weight: bold;
    margin-bottom: 5px;
    padding-left: 10px;
}

details[open] > summary::after {
    content: '▲';  /* Use an up arrow when expanded */
    float: right;
    margin-left: 10px;
}

details > summary::after {
    content: '▼';  /* Use a down arrow when collapsed */
    float: right;
    margin-left: 10px;
}

p {
    margin-top: 5px;
    margin-bottom: 5px;
}


.file-custom {
    height: 80px;
}
"""




with gr.Blocks(css=css) as iface:
    gr.Markdown("# Astronomy Experiment Generator")
    gr.Markdown("Generate experiments based on a given science objective.")

    with gr.Row():
        with gr.Column(scale=1):
            science_objective = gr.Textbox(label="Science Goal")
            branching_factor = gr.Slider(label="Branching Factor", minimum=1, maximum=10, step=1, value=2)
            year_cutoff = gr.Slider(label="Year Cutoff", minimum=2000, maximum=2024, step=1, value=2024)
            openai_key = gr.Textbox(label="OpenAI API Key", type="password")
            anthropic_key = gr.Textbox(label="Anthropic API Key", type="password")

            submit_btn = gr.Button("Generate")

        with gr.Column(scale=2):
            tree_output = gr.JSON(label="Tree Data", visible=False)
            tree_display = gr.HTML(label="Tree Display")

            with gr.Row():
                save_btn = gr.Button("Save Tree", size="sm")
                load_btn = gr.Button("Load Tree", size="sm")

            file_output = gr.File(label="Tree File", elem_classes=["file-custom"])
            file_input = gr.File(label="Load Tree File", elem_classes=["file-custom"])

    def update_tree_display(tree):
        if tree is not None:
            return create_tree_html(tree)
        return gr.update()

    submit_btn.click(
        generate_science_tree,
        inputs=[science_objective, openai_key, anthropic_key, branching_factor, year_cutoff],
        outputs=[tree_output]
    ).then(
        update_tree_display,
        inputs=[tree_output],
        outputs=[tree_display]
    )

    save_btn.click(
        save_tree_to_file,
        inputs=[tree_output],
        outputs=[file_output]
    )

    file_input.change(
        load_tree_from_file,
        inputs=[file_input],
        outputs=[tree_output]
    ).then(
        update_tree_display,
        inputs=[tree_output],
        outputs=[tree_display]
    )

# Launch the Gradio app
iface.launch()