import logging
from collections import Counter, OrderedDict, defaultdict
import os
import unicodedata

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from incf.countryutils import transformations as tx

log = logging.getLogger(__name__)


colormap = {'Africa': '#b891a1',
            'Europe': '#7b7b96',
            'Asia': '#98b68f',
            'North America': '#fff8e2',
            'South America': '#928457',
            'Oceania': '#434363'}


test_names = [
    ('Edward A', 'Wenger'),
    ('Philip A', 'Eckhoff'),
    ('Jaline', 'Gerardin'),
    ('John M', 'Miller'),
    ('Kafula', 'Silumbe'),
    ('Teun', 'Bousema'),
    ('Gerry', 'Killeen'),
    (u'Andr\u00E9 Lin', u'Ou\u00E9draogo'),
    ('Tom', 'Smith'),
    ('John', 'Marshall'),
    ('Michael', 'White'),
    ('Simon', 'Hay')
]


def remove_accents(s):
    if isinstance(s, str):  # also handles derived, e.g. Bio.Entrez.StringElement
        s = unicode(s, 'utf-8')
    return ''.join(x for x in unicodedata.normalize('NFKD', s)
                   if unicodedata.category(x) != 'Mn')


def continent_from_country(cc):
    if not cc:
        return None
    if cc == 'SS':  # South Sudan not in look-up yet
        return 'Africa'
    try:
        return tx.cn_to_ctn(tx.cc_to_cn(cc))
    except KeyError:
        log.warn('Trouble converting country-code %s', cc)
        return None


def get_initials(forename, first_only=True):
    initials = [x[0] for x in forename.split()]
    return initials[0] if first_only else ' '.join(initials)


def merge_records(members, first_only=True, no_accents=True):
    merged = defaultdict(list)
    for (forename, lastname), v in members.items():
        initials = get_initials(forename, first_only)
        if no_accents:
            lastname = remove_accents(lastname)
        merged_name = (initials, lastname)
        merged[merged_name] += v
    return merged


def parse_member_info(members):
    all_member_info = []
    for k, v in members.items():
        log.debug(k)
        member_info = {}

        member_info['ForeName'] = k[0]
        member_info['LastName'] = k[1]

        v.sort(key=lambda t: t[0])  # sort (date, country) by date
        member_info['FirstDate'] = v[0][0]
        member_info['LastDate'] = v[-1][0]

        member_info['Publications'] = len(v)

        countries = [x[1] for x in v if x[1]]
        country_counts = Counter(countries)
        member_info['MainCountry'] = country_counts.most_common(1)[0][0] if country_counts else None
        member_info['FirstCountry'] = countries[0] if countries else None

        continents = [continent_from_country(c) for c in countries]
        member_info['FirstContinent'] = continents[0] if continents else None
        counts = Counter(continents)
        member_info.update(counts)

        all_member_info.append(member_info.copy())

    return pd.DataFrame.from_records(all_member_info).set_index(['ForeName', 'LastName']).sort_values(by='FirstDate')


def plot_entries(member_data):

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    n_entries = len(member_data)

    for i, v in enumerate(member_data.values()):
        log.debug(v)
        dates = [x[0] for x in v]
        colors = [colormap[continent_from_country(x[1])] if x[1] else 'w' for x in v]
        ax.scatter(dates, [i]*len(v), s=min(100, max(30, 5000/n_entries)), c=colors, alpha=1, zorder=100)
        ax.plot([min(dates), max(dates)], [i, i], lw=0.5, alpha=0.6, color='k', zorder=0)

    ax.set(yticks=range(n_entries),
           ylim=(-0.5, n_entries))

    ax.set_yticklabels([' '.join(k) for k in member_data.keys()], fontsize=7)

    fig.set_tight_layout(True)


def random_subset(members, n=50):
    return dict(members.items()[:n])


def subset_from_names(members, names):
    return OrderedDict([(name, members[name]) for name in names])


if __name__ == '__main__':

    logging.basicConfig(format='%(message)s')
    log.setLevel(logging.INFO)

    with open(os.path.join('data', 'members_all.p'), 'rb') as fp:
        members = pickle.load(fp)
    log.info('%s unique members', len(members.keys()))

    first_only = True  # some erroneous merges (J Cohen) but better than ~10% split entries
    members = merge_records(members, first_only=first_only)
    log.info('%s unique members (after merging on %sinitials)',
             len(members.keys()), 'first ' if first_only else '')

    max_to_parse = None
    subset = random_subset(members, max_to_parse)

    # test_names_initials = [(remove_accents(get_initials(t[0], first_only)),
    #                         remove_accents(t[1])) for t in test_names]
    # subset.update(subset_from_names(members, test_names_initials))

    df = parse_member_info(subset)

    # continent = 'Africa'  # Asia, North America, Europe, etc.
    # df = df[df.FirstContinent == continent]

    country = 'KH'  # BF, KE, AU, TZ, NG, etc.
    df = df[(df.FirstCountry == country)]

    min_pubs = 2
    df = df[df.Publications >= min_pubs]

    # df = df.xs('Ouedraogo', level='LastName', drop_level=False)

    # continents = ['Europe', 'North America', 'Asia', 'Oceania']
    # continents = ['Africa']
    # df = df[np.logical_and.reduce([df[c] > 0 for c in continents])]  # require all
    # df = df[np.logical_or.reduce([df[c] > 0 for c in continents])]  # require any

    range_to_show = slice(None, 500)
    df = df.iloc[range_to_show]

    log.info(df[['MainCountry', 'FirstContinent', 'FirstDate', 'Publications']].head(15))

    subset = subset_from_names(members, df.index.values)

    plot_entries(subset)
    plt.show()
