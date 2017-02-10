import logging

import pycountry

from states import states

log = logging.getLogger(__name__)


common_fixes = {
    'UK': 'GB'
}


def try_find(country, strict=False):

    try:
        return pycountry.countries.get(name=country).alpha_2
    except KeyError:
        pass

    if strict:
        return

    try:
        return pycountry.countries.lookup(country).alpha_2
    except LookupError:
        pass

    try:
        return common_fixes[country]
    except KeyError:
        pass

    # recover US states and zip codes (with no country listed)

    if country:
        last = country.split()[-1]
        if last.isdigit() and len(last) == 5:
            return 'US'
        if last in states.keys() or last in states.values():
            return 'US'


def match(name):

    # try the most common pattern first
    country = name.split(',')[-1].replace('.', '').split(';')[0].strip()

    # remove email addresses
    country = ' '.join([x for x in country.split() if '@' not in x])

    x = try_find(country)
    if x:
        log.debug('%s --> %s', country, x)
        return x
    else:
        # TODO: Or would it be better to be cautious instead of boldly searching for country names?
        #       For example this nasty case: https://www.ncbi.nlm.nih.gov/pubmed/26488565
        for w in country.split():
            x = try_find(w, strict=True)
            if x:
                log.debug('%s --> %s', country, x)
                return x

    log.debug('\n%s\n%s --> ??', name, country)
