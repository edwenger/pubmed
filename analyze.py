import logging
from collections import Counter, OrderedDict, defaultdict
import datetime
from itertools import tee, izip
import os
import unicodedata

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from incf.countryutils import transformations as tx
from lifelines import KaplanMeierFitter

log = logging.getLogger(__name__)


colormap = {'Africa': '#b891a1',
            'Europe': '#6774a6',
            'Asia': '#98b68f',
            'North America': '#ffe07c',
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


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def timedelta_in_years(td):
    return td.total_seconds() / 3600 / 24 / 365.242199


def plot_entries(member_data, relative_dates=False):

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    n_entries = len(member_data)

    tdiffs = []

    for i, v in enumerate(member_data.values()):
        log.debug(v)
        tdiffs += [timedelta_in_years(b[0] - a[0]) for a, b in pairwise(v)]
        if relative_dates:
            dates = [timedelta_in_years(x[0] - v[0][0]) for x in v]
            career = timedelta_in_years(datetime.date.today() - v[0][0])
            ax.plot(career, i, 'k|')
        else:
            dates = [x[0] for x in v]
        colors = [colormap[continent_from_country(x[1])] if x[1] else 'lightgray' for x in v]
        ax.scatter(dates, [i]*len(v), s=min(100, max(50, 5000/n_entries)), c=colors, alpha=0.5, zorder=100)
        ax.plot([min(dates), max(dates)], [i, i], lw=0.5, alpha=0.75, color='k', zorder=0)

    ax.set(yticks=range(n_entries),
           ylim=(-0.5, n_entries))

    ax.set_yticklabels([' '.join(k) for k in member_data.keys()], fontsize=7)

    if relative_dates:
        ax.set(xlim=(-0.1, ax.get_xlim()[1]))

    fig.set_tight_layout(True)

    fig = plt.figure('time_differences')
    plt.hist(tdiffs, color='gray', alpha=0.8, bins=100)
    plt.gca().set_xlabel('Years between publications')
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

    first_only = True  # some erroneous merges (J Cohen, A Ouedraogo) but better than ~10% split entries
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

    country = 'TZ'  # BF, KE, AU, TZ, NG, TH, ZM, IN, CN, etc.
    # df = df[(df.FirstCountry == country)]
    df = df[(df.MainCountry == country)]

    min_pubs = 2
    df = df[df.Publications >= min_pubs]

    df = df[df.FirstDate > datetime.date(2008, 1, 1)]
    df = df[df.FirstDate < datetime.date(2014, 1, 1)]

    df['CareerDuration'] = df.LastDate - df.FirstDate
    df['DepartureObserved'] = (datetime.date.today() - df.LastDate) > datetime.timedelta(days=365 * 1.5)
    df.sort_values(by='CareerDuration', ascending=False, inplace=True)

    fig = plt.figure('Career Duration')
    ax = plt.gca()
    kmf = KaplanMeierFitter()
    kmf.fit(df.CareerDuration.map(timedelta_in_years),
            event_observed=df.DepartureObserved,
            label='Career Duration')
    kmf.plot(ax=ax)
    ax.set(ylim=(0, 1), xlabel='')
    fig.set_tight_layout(True)

    # df = df.xs('Ouedraogo', level='LastName', drop_level=False)

    # continents = ['Europe', 'North America', 'Asia', 'Oceania']
    # continents = ['Africa']
    # df = df[np.logical_and.reduce([df[c] > 0 for c in continents])]  # require all
    # df = df[np.logical_or.reduce([df[c] > 0 for c in continents])]  # require any

    range_to_show = slice(None, 500)
    df = df.iloc[range_to_show]

    log.info(df[['MainCountry', 'FirstContinent', 'FirstDate', 'Publications']].head(15))

    subset = subset_from_names(members, df.index.values)

    relative_dates = False  # plot careers with respect to first publication
    plot_entries(subset, relative_dates=relative_dates)
    plt.show()
