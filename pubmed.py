# you need to install Biopython:
# pip install biopython

# Full discussion:
# https://marcobonzanini.wordpress.com/2015/01/12/searching-pubmed-with-python/

import logging

from Bio import Entrez

log = logging.getLogger(__name__)

email = 'ewenger@intven.com'


def search(query, retmax, sort_by='relevance'):
    Entrez.email = email
    handle = Entrez.esearch(db='pubmed',
                            sort=sort_by,
                            retmax=str(retmax),
                            retmode='xml',
                            term=query)
    results = Entrez.read(handle)
    return results


def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = email
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results
