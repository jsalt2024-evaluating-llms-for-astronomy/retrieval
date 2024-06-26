import json
import re
import numpy as np
import anthropic
import requests
from bs4 import BeautifulSoup
import nest_asyncio
import asyncio
from playwright.async_api import async_playwright
from markdownify import markdownify as md
import yaml
from urllib.parse import urlencode, quote_plus

ads_token = "6pmanBZytaNltPsonmdbJATGnDZO7mAxluAxgYfz"
with open("/users/christineye/retrieval/config.yaml", 'r') as stream:
    api_key = yaml.safe_load(stream)['anthropic_api_key']
nest_asyncio.apply()
client = anthropic.Anthropic(api_key = api_key)

# COLLECTING DATA FROM ARA&A
def load_araa(from_file = True):
    if from_file:
        with open('./araa.json', 'r') as f:
            json_results = json.load(f)
    else:
        from urllib.parse import urlencode, quote_plus
        query = {"q": "bibstem:ara&a", "fl": "title, year, bibcode, identifier", "rows":1000}
        encoded_query = urlencode(query)
        results = requests.get("https://api.adsabs.harvard.edu/v1/search/query?{}".format(encoded_query), \
                            headers={'Authorization': 'Bearer ' + ads_token})
        json_results = results.json()
    
    return json_results['response']['docs']

def pull_arxiv_and_doi(idlist):
    arXiv_pattern = r'arXiv:\d{4}\.\d{4}'
    arxiv, doi = "", ""
    for item in idlist:
        if re.match(arXiv_pattern, item):
            arxiv = item.split('arXiv:')[1]
        elif '10.1146/annurev' in item:
            doi = item
    return arxiv, doi

def format_reviews(json_docs, cutoff = 2000):
    all_reviews = []
    for result in json_docs:
        if int(result['year']) > cutoff:
            arxiv, doi = pull_arxiv_and_doi(result['identifier'])
            if doi != "" and arxiv != "":
                url = "https://www.annualreviews.org/content/journals/" + doi
                all_reviews.append({'title': result['title'][0], "id": arxiv, 'url': url, })
    return all_reviews

# FETCHING PAPER TEXTS
async def fetch_page_content(url):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)  # Set headless=True if you don't need a browser UI
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(url)
        await page.wait_for_load_state('networkidle')

        content = await page.content()

        await browser.close()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        paragraphs = [md(p.text) for p in soup.find_all('p')]
        
        return paragraphs

async def fetch_multiple_pages(urls):
    tasks = [fetch_page_content(url) for url in urls]
    return await asyncio.gather(*tasks)

def get_page_contents(reviews):
    urls = [review["url"] for review in reviews]
    return asyncio.run(fetch_multiple_pages(urls))

def scrape_all_papers(reviews, batch_size = 10):
    reviews_with_text = []

    for i in range(len(reviews) // batch_size + 1):
        batch = reviews[i * batch_size : i * batch_size + (batch_size - 1)]
        content = get_page_contents(batch)
        
        for j, paper in enumerate(content):
            if "institutional or personal subscription" not in paper[-1]:
                review = reviews[i * batch_size + j].copy()
                review['text'] = paper
                reviews_with_text.append(review)
    
    return reviews_with_text

def load_papers(from_file = True):
    if from_file:
        with open('./araa_papers.json', 'r') as f:
            papers = json.load(f)
    else:
        review_json = load_araa(from_file = True)
        reviews = format_reviews(review_json)
        papers = scrape_all_papers(reviews)
    
    return papers

# PARAGRAPH SELECTION AND QUERY GENERATION
def scrape_citations(text):
    patterns = ['([A-Z][a-z´]+)\s+(\d{4})', # Name Year
                '([A-Z][a-z´]+) et al\. (\d{4})', # Name et al. Year
                '([A-Z][a-z´]+) et al\. \((\d{4})\)', # Name et al. (Year)
                '([A-Z][a-z´]+) & ([A-Z][a-z]+) (\d{4})', # Name & Name Year
                '([A-Z][a-z´]+) & ([A-Z][a-z]+) \((\d{4})\)',
                '([A-Z][a-z´]+),\s+([A-Z][a-z]+) & ([A-Z][a-z]+) (\d{4})']
    
    citations = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            citations.append(match)
    
    return citations

def citation_density(content, k, mode = "topk", maxn = 12):
    num_citations = np.array([len(set(scrape_citations(p))) for p in content])
    
    if mode == "topk":
        indices = np.flip(np.argsort(num_citations))[:k]
    elif mode == "threshold":
        mask = np.logical_and(num_citations >= k, num_citations < maxn)
        indices = np.arange(len(content))[mask]
    
    return np.sort(indices)

def get_best_paragraphs(content, k, mode = "threshold"):
    indices = citation_density(content, k, mode)
    string = ""
    
    for index in indices:
        string += str(index)
        string += ": "
        string += content[index]
        string += "\n\n"
    
    return string

def claude_paragraphs(paper):
    # literally using a less overloaded character '{' to split the paragraphs}
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=500,
        temperature=0,
        system="""You are an expert astronomer. Given this list of paragraphs from a scientific paper, generate a focused research question for each paragraph.
                Formulate the question such that it is focused and concise, but covers all topics in the paragraph. 
                Then assess which paragraphs are most on-topic and closely related to their research question.
                If the question has multiple sub-questions, a good and focused paragraph shoudl address all of them.
                Return the 3 best question-paragraph pairs in this format: {index, question}.
                Do not include any text before or after each {index, question}, including any introduction or rationale.""",
                # Also return the 3 paragraphs and corresponding questions that are least on-topic and related to the research question.
        messages=[{"role": "user",
                "content": [{"type": "text", "text": paper,}] }]
    )
    
    return message

def process_paper(paper, verbose = False):
    content = paper['text']
    paragraphs = get_best_paragraphs(content, 5)
    message = claude_paragraphs(paragraphs)
    message = message.content[0].text
    
    results = []
    if verbose: print(message)
    for pair in message.split('\n\n'):
        if '{' and '}' in pair:
            index, question = pair[1:-1].split(',', 1)
            paragraph = content[int(index)]
            results.append({'title': paper['title'], 'id': paper['id'], 'question': question, 'paragraph': paragraph, 'citations': set(scrape_citations(paragraph))})
    
    return results

# MAIN
def main():
    papers = load_papers(from_file = True)
    print('Number of papers:', len(papers))

    test_papers = papers[:3]
    query_pairs = []
    for paper in test_papers:
        query_pairs.append(process_paper(paper['text']))

if __name__ == '__main__':
    main()