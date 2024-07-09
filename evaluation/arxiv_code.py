from urllib.parse import urlencode, quote_plus
import requests
import unicodedata
import regex as re

ads_token = "YOUR_ADS_TOKEN"

def pull_arxiv_and_doi(idlist):
    """Extract the arXiv ID and DOI from the list of identifiers returned by ADS."""
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

def query_ads(ref, verbose = False, incyear = True):
    qstring = "first_author:{} ".format(ref['surnames'][0])
    for surname in ref['surnames'][1:]:
        if surname != "":
            qstring += "author:{} ".format(surname)
    if incyear: qstring += "year:{}".format(ref['year'])
    if verbose: print(qstring)
    
    query = {"q": qstring, "fl": "title, year, bibcode, author, identifier", "rows":1000}
    encoded_query = urlencode(query)
    results = requests.get("https://api.adsabs.harvard.edu/v1/search/query?{}".format(encoded_query), \
                        headers={'Authorization': 'Bearer ' + ads_token})
    
    json_results = results.json()['response']['docs']
    

    return json_results

def search_arxiv(ref, verbose = False):
    """Search for the arXiv ID of a paper based on the author list and publication year."""
    if int(ref['year']) < 1991:
        return None
    
    # arxiv code, doesn't work very well
    # query = ""
    # for i, surname in enumerate(ref['surnames']):
    #     if '-' not in surname: 
    #         query += "au:"
    #         query += surname.replace("'","")
    #         if i != len(ref['surnames']) - 1: query += " AND "
    
    # print(query)
    # search = arxiv.Search(query = query, max_results = 10)
    # results = arxiv_client.results(search)
    json_results = query_ads(ref, verbose, incyear = True)
    
    if len(json_results) == 0:
        json_results = query_ads(ref, verbose, incyear = False)

    for r in json_results: # Strict match check (author order & year)
        valid = True
        if int(r['year']) > int(ref['year']) + 1 or int(r['year']) < int(ref['year']) - 1:
            valid = False
        
        for i, surname in enumerate(ref['surnames']):
            if i > (len(r['author']) - 1) or unicodedata.normalize('NFKD', surname).encode('ascii', 'ignore') not in unicodedata.normalize('NFKD', r['author'][i]).encode('ascii', 'ignore'):
                valid = False
                break
        
        if valid: 
            if verbose: print(r['identifier'])
            arxiv, doi = pull_arxiv_and_doi(r['identifier'])
            return arxiv#, doi
    
    print('Failed on ' + ref['surnames'][0] + ' ' + ref['year'])
    return None