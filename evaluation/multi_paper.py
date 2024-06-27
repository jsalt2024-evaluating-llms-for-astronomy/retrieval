import json
import re
import numpy as np
import anthropic
import requests
from bs4 import BeautifulSoup
import nest_asyncio
import asyncio
from playwright.async_api import async_playwright, ElementHandle
from markdownify import markdownify as md
import yaml
from urllib.parse import urlencode, quote_plus
from tqdm import tqdm
import arxiv
from urllib.parse import urlencode, quote_plus
arxiv_client = arxiv.Client()

ads_token = "6pmanBZytaNltPsonmdbJATGnDZO7mAxluAxgYfz"
with open("/users/christineye/retrieval/config.yaml", 'r') as stream:
    api_key = yaml.safe_load(stream)['anthropic_api_key']
nest_asyncio.apply()
client = anthropic.Anthropic(api_key = api_key)

# COLLECTING DATA FROM ARA&A
def load_araa(from_file = True):
    if from_file:
        with open('./araa_index.json', 'r') as f:
            json_results = json.load(f)
    else:
        query = {"q": "bibstem:ara&a", "fl": "title, year, bibcode, identifier", "rows":1000}
        encoded_query = urlencode(query)
        results = requests.get("https://api.adsabs.harvard.edu/v1/search/query?{}".format(encoded_query), \
                            headers={'Authorization': 'Bearer ' + ads_token})
        json_results = results.json()
    
    return json_results['response']['docs']

def pull_arxiv_and_doi(idlist):
    arXiv_pattern = r'arXiv:\d{4}\.\d{4}'
    arxiv, doi = None, None
    for item in idlist:
        if re.match(arXiv_pattern, item):
            arxiv = item.split('arXiv:')[1]
        elif '10.1146/annurev' in item:
            doi = item
        elif re.match('([a-zA-Z:.]*)astro-ph/(\d*)', item):
            arxiv = item.split('astro-ph/')[1]
    return arxiv, doi

def format_reviews(json_docs, cutoff = 2010):
    all_reviews = []
    for result in json_docs:
        if int(result['year']) > cutoff:
            arxiv, doi = pull_arxiv_and_doi(result['identifier'])
            if doi is not None:
                url = "https://www.annualreviews.org/content/journals/" + doi
                all_reviews.append({'title': result['title'][0], "id": arxiv, 'url': url, })
    return all_reviews

# FETCHING PAPER TEXTS
async def fetch_page_content(url):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)  
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(url)
        await page.wait_for_load_state('networkidle')

        # Extract the reference content
        ref_elements = await page.query_selector_all('span.references li.refbody')

        async def process_ref(element: ElementHandle):
            surname_elements, year_element, collab_element = await asyncio.gather(
                element.query_selector_all('span.reference-surname'),
                #element.query_selector_all('span.reference-given-names'),
                element.query_selector('span.reference-year'),
                #element.query_selector('span.reference-source'),
                element.query_selector('span.reference-collab')
            )

            surnames = await asyncio.gather(*[surname.inner_text() for surname in surname_elements])
            #given_names = await asyncio.gather(*[given_names.inner_text() for given_names in given_names_elements])
            year = await year_element.inner_text() if year_element else None
            #source = await source_element.inner_text() if source_element else None
            collab = await collab_element.inner_text() if collab_element else None
            
            return {
                'surnames': surnames,
                #'given_names': given_names,
                'year': year,
                #'source': source,
                'collab': collab,
            }

        ref_data = await asyncio.gather(*[process_ref(element) for element in ref_elements])
        
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        paragraphs = [md(p.text) for p in soup.find_all('p')]

        await browser.close()
        return {
            'references': ref_data,
            'paragraphs': paragraphs
        }

async def fetch_multiple_pages(urls):
    tasks = [fetch_page_content(url) for url in urls]
    return await asyncio.gather(*tasks)

def get_page_contents(reviews):
    urls = [review["url"] for review in reviews]
    return asyncio.run(fetch_multiple_pages(urls))


def scrape_all_papers(reviews, batch_size = 10):
    reviews_with_text = []

    for i in tqdm(range(len(reviews) // batch_size + 1)):
        batch = reviews[i * batch_size : i * batch_size + (batch_size - 1)]
        content = get_page_contents(batch)
        
        for j, paper in enumerate(content):
            paper['paragraphs'] = [re.sub('\(#right-ref-[A-Za-z0-9]+\)', '', p) for p in paper['paragraphs']]

            if "institutional or personal subscription" not in paper['paragraphs'][-1]:
                review = reviews[i * batch_size + j].copy()
                review['text'] = paper['paragraphs']
                review['fullbib'] = paper['references']
                reviews_with_text.append(review)
    
    return reviews_with_text

def load_papers(from_file = True, batch_size = 10):
    if from_file:
        with open('./araa_papers.json', 'r') as f:
            papers = json.load(f)
    else:
        review_json = load_araa(from_file = True)
        reviews = format_reviews(review_json)
        print('Number of reviews:', len(reviews))
        papers = scrape_all_papers(reviews, batch_size = batch_size)
    
    return papers


def link_to_bib(review, citations): # paragraph level
    cited_refs = []
    for citation in set(citations):
        for ref in review['fullbib']:
            if ref['year'] == citation[-1]:
                if set(citation[:-1]).issubset(ref['surnames']) or citation[0] == ref['collab']:
                    if ref not in cited_refs:
                        cited_refs.append(ref)
    
    return cited_refs

def search_arxiv(ref, verbose = False):
    if int(ref['year']) < 2000:
        return None
    
    # query = ""
    # for i, surname in enumerate(ref['surnames']):
    #     if '-' not in surname: 
    #         query += "au:"
    #         query += surname.replace("'","")
    #         if i != len(ref['surnames']) - 1: query += " AND "
    
    # print(query)
    # search = arxiv.Search(query = query, max_results = 10)
    # results = arxiv_client.results(search)

    qstring = "first_author:{} ".format(ref['surnames'][0])
    for surname in ref['surnames'][1:]:
        if surname != "":
            qstring += "author:{} ".format(surname)
    qstring += "year:{}".format(ref['year'])
    if verbose: print(qstring)
    
    query = {"q": qstring, "fl": "title, year, bibcode, author, identifier", "rows":1000}
    encoded_query = urlencode(query)
    results = requests.get("https://api.adsabs.harvard.edu/v1/search/query?{}".format(encoded_query), \
                        headers={'Authorization': 'Bearer ' + ads_token})
    
    json_results = results.json()['response']['docs']
    
    
    for r in json_results:
        valid = True
        # if r.published.year > int(ref['year']) + 2 or r.published.year < int(ref['year']) - 2:
        #     continue
        for i, surname in enumerate(ref['surnames']):
            if i > (len(r['author']) - 1) or surname not in r['author'][i]:
                break
        if valid: 
            if verbose: print(r['identifier'])
            arxiv, doi = pull_arxiv_and_doi(r['identifier'])
            return arxiv
    print('Failed on ' + ref['surnames'][0] + ' ' + ref['year'])
    return None


# PARAGRAPH SELECTION AND QUERY GENERATION
def scrape_citations(text):
    patterns = ['([A-Z][A-Za-z´-]+) \(?(\d{4})', # Name Year
                '([A-Z][A-Za-z´-]+) et al\. \(?(\d{4})',
                '([A-Z][A-Za-z´-]+) & ([A-Z][a-z]+) \(?(\d{4})',
                '([A-Z][A-Za-z´-]+),\s+([A-Z][a-z]+) & ([A-Z][a-z]+) \(?(\d{4})',
                '([A-Z][A-Za-z]+) ([A-Z][a-zA-Z]+) et al. \(?(\d{4})',
                '([A-Z][a-zA-Z]+[A-Z][a-zA-Z]+) et al. \(?(\d{4})',
                '([A-Z][a-zA-Z]+\s[A-Z][a-zA-Z]+) & ([A-Z][a-zA-Z]+\s[A-Z][a-zA-Z]+) \(?(\d{4})']
    
    citations = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            citations.append(match)
    
    return citations

def citation_density(content, k, mode = "topk", maxn = 20):
    num_citations = np.array([len(set(scrape_citations(p))) for p in content])
    
    if mode == "topk":
        indices = np.flip(np.argsort(num_citations))
        mask = [num_citations[index] < maxn for index in indices]
        indices = indices[mask][:k]

    elif mode == "threshold":
        tmask = np.logical_and(num_citations >= k)
        indices = np.arange(len(content))[tmask]

    return np.sort(indices)

def get_best_paragraphs(content, k, mode = "topk", maxn = 25):
    indices = citation_density(content, k, mode, maxn)
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
        system="""You are an expert astronomer. Given this list of paragraphs from a scientific paper, generate a focused research question for each paragraph that is answered by the paragraph text.
                Formulate the question such that it is expert-level, focused, and relevant to the paragraph. Be concise.
                Then assess which paragraphs are most on-topic and closely related to their research question.
                If the question has multiple sub-questions, a good and focused paragraph should address all of them.
                Return the 3 best question-paragraph pairs in this format: {index, question}.
                Do not include any text before or after each {index, question}, including any introduction or rationale.""",
                # Also return the 3 paragraphs and corresponding questions that are least on-topic and related to the research question.
        messages=[{"role": "user",
                "content": [{"type": "text", "text": paper,}] }]
    )
    
    return message

def process_paper(paper, verbose = False, mode = "topk", k = 10):
    content = paper['text']
    paragraphs = get_best_paragraphs(content, k, mode)
    message = claude_paragraphs(paragraphs)
    message = message.content[0].text
    
    pqueries = []
    if verbose: print(message)
    for pair in message.split('\n\n'):
        if '{' and '}' in pair:
            index, question = pair[1:-1].split(',', 1)
            paragraph = content[int(index)]
            citations = set(scrape_citations(paragraph))
            pqueries.append({'title': paper['title'], 'id': paper['id'], 
                            'question': question, 'paragraph': paragraph, 
                            'citations': citations, 'bibs': link_to_bib(paper, citations)})
    
    return pqueries

def arxiv_link(pqueries): # on the paper level (K queries)
    for i, query in enumerate(pqueries):
        query['arxiv'] = []
        for bib in query['bibs']:
            arxiv_id = search_arxiv(bib)
            if arxiv_id is not None:
                query['arxiv'].append(arxiv_id)
    
    query['arxiv'] = set(query['arxiv'])
            
    return pqueries

# MAIN
def main():
    papers = load_papers(from_file = True)
    print('Number of papers:', len(papers))

    query_pairs = []
    for paper in papers:
        query_pairs.append(process_paper(paper['text']))

    query_pairs_formatted = {}
    for paper in query_pairs:
        for i, query in enumerate(paper):
            id_str = query['id'].split('_')[0] + '_' + str(i + 1)
            query_pairs_formatted[id_str] = {'title':query['title'], 'question':query['question'], 
                                            'text': query['paragraph'], 'citations':query['citations'],
                                            'arxiv': list(query['arxiv'])}

    fname = ""#'../data/multi_paper_examples.json'
    
    with open('../data/multi_paper.json', 'w') as f:
        json.dump(query_pairs_formatted, f, indent = 2)

if __name__ == '__main__':
    main()