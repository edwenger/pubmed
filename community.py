from collections import defaultdict
import datetime
import json
import logging
import os
import pickle

from pubmed import search, fetch_details
import country

log = logging.getLogger(__name__)


class Institution(object):
    def __init__(self, name):
        components = name.split(', ')
        self.name = components[0].strip()
        self.country = country.match(name)


class ArticleAuthor(object):
    def __init__(self, author):
        self.name = author['ForeName'], author['LastName']
        self.initials = author['Initials']
        self.affiliations = [Institution(a['Affiliation']) for a in author['AffiliationInfo']]
        # TODO: only first-author affiliations on PubMed before 2014 :(

    def __str__(self):
        return '%s %s' % (self.initials, self.name[1])


class Article(object):
    def __init__(self, article):

        self.title = article['ArticleTitle']

        if 'AuthorList' in article:
            self.authors = [ArticleAuthor(a) for a in article['AuthorList'] if 'ForeName' in a]  # e.g. CollectiveName
        else:
            self.authors = []

        if not article['ArticleDate']:
            log.debug('No date associated with "%s"', self.title)
            self.date = None
        else:
            date = article['ArticleDate'][0]
            self.date = datetime.date(int(date.get('Year')),
                                      int(date.get('Month')),
                                      int(date.get('Day', 1)))

    def __str__(self):
        author_list = unicode(self.authors[0])
        if len(self.authors) > 1:
            author_list += ' et al'
        return '%s. "%s" (%d)' % (author_list, self.title, self.date.year)


class Community(object):
    def __init__(self):
        self.members = defaultdict(list)  # (first, last): [(date, country), ...]

    def query(self, query, retmax, sort_by='relevance'):
        results = search(query, retmax, sort_by=sort_by)
        n_results = len(results['IdList'])
        log.info('Query returned %d results...', n_results)

        i = 0
        chunksize = 5000  # Entrez.efetch seems to limit to 10k?
        for slice_ in range(0, n_results, chunksize):
            articles = fetch_details(results['IdList'][slice_:slice_+chunksize])
            log.info('Fetched %d article details...', len(articles['PubmedArticle']))
            for article in articles['PubmedArticle']:  # [u'PubmedArticle', u'PubmedBookArticle']
                i += 1
                if not (i % 1000):
                    log.info('Processed %d articles...', i)
                self.add_article(Article(article['MedlineCitation']['Article']))
            log.info('Processed %d articles...', i)

    def add_article(self, article):
        log.debug('%s', article)
        if not article.date:
            return
        log.debug('%s', article.date)

        for author in article.authors:
            log.debug('%s', author)
            for affiliation in author.affiliations:
                affiliation_country = (article.date, affiliation.country)
                self.members[author.name].append(affiliation_country)


if __name__ == '__main__':

    logging.basicConfig(format='%(message)s')
    log.setLevel(logging.INFO)
    country.log.setLevel(logging.WARN)

    community = Community()
    community.query('malaria', retmax=100000, sort_by='date')  # date, relevance
    # summary = {' '.join(k): [(t.strftime('%Y-%m'), x) for t, x in v] for k, v in community.members.items()}
    # print(json.dumps(summary, indent=4))
    with open(os.path.join('data', 'members.p'), 'wb') as fp:
        pickle.dump(community.members, fp)

    # Pretty print the first paper in full
    # print(json.dumps(papers['PubmedArticle'][0], indent=4, separators=(',', ':')))
