"""Static seed tables (copied and extended from synthetic_data_gen/src/common.py; that project is read-only).
Learned per-country lists (wordlists.py) are merged on top of these at load time."""

US_STATES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
    'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
    'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
    'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
    'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia',
    'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming', 'PR': 'Puerto Rico',
}

IN_STATES = {
    'Maharashtra': 'MH', 'Delhi': 'DL', 'New Delhi': 'DL', 'Uttar Pradesh': 'UP', 'Karnataka': 'KA', 'Tamil Nadu': 'TN',
    'Tamilnadu': 'TN', 'West Bengal': 'WB', 'Telangana': 'TG', 'Gujarat': 'GJ', 'Kerala': 'KL', 'Keralam': 'KL',
    'Haryana': 'HR', 'Rajasthan': 'RJ', 'Bihar': 'BR', 'Madhya Pradesh': 'MP', 'Andhra Pradesh': 'AP', 'Orissa': 'OD',
    'Odisha': 'OD', 'Punjab': 'PB', 'Assam': 'AS', 'Jharkhand': 'JH', 'Chhattisgarh': 'CG', 'Uttarakhand': 'UK',
    'Himachal Pradesh': 'HP', 'Jammu and Kashmir': 'JK', 'Goa': 'GA', 'Chandigarh': 'CH', 'Puducherry': 'PY',
    'Pondicherry': 'PY', 'Manipur': 'MN', 'Meghalaya': 'ML', 'Tripura': 'TR', 'Nagaland': 'NL', 'Mizoram': 'MZ',
    'Sikkim': 'SK', 'Arunachal Pradesh': 'AR', 'Paschimbanga': 'WB', 'Pashchimbanga': 'WB',
}
IN_CODE_EXTRA = {'TS': 'TG', 'OR': 'OD', 'CT': 'CG', 'UT': 'UK'}
# the generator swaps Telangana<->Andhra Pradesh, Orissa<->Odisha, Kerala->Keralam (same canonical code except TG/AP)
IN_ADMIN_EQUIV = {('TG', 'AP'), ('AP', 'TG')}
IN_CITY_ALIASES = {'mumbai': 'bombay', 'chennai': 'madras', 'kolkata': 'calcutta', 'pune': 'poona', 'gurgaon': 'gurugram',
                   'bangalore': 'bengaluru', 'vadodara': 'baroda', 'thiruvananthapuram': 'trivandrum',
                   'ahmedabad': 'ahmadabad', 'visakhapatnam': 'vishakhapatnam', 'mysore': 'mysuru', 'kochi': 'cochin',
                   'howrah': 'haora'}

FR_REGION_DEPTS = {
    'Hauts-de-France': ['Nord', 'Pas-de-Calais', 'Somme', 'Oise', 'Aisne'],
    'Nouvelle-Aquitaine': ['Gironde', 'Pyrénées-Atlantiques', 'Landes', 'Charente-Maritime', 'Haute-Vienne', 'Vienne', 'Dordogne'],
    'Pays de la Loire': ['Loire-Atlantique', 'Maine-et-Loire', 'Sarthe', 'Vendée', 'Mayenne'],
    'Île-de-France': ['Paris', 'Hauts-de-Seine', 'Seine-Saint-Denis', 'Val-de-Marne', 'Yvelines', 'Essonne'],
    'Auvergne-Rhône-Alpes': ['Rhône', 'Isère', 'Puy-de-Dôme', 'Loire', 'Haute-Savoie', 'Savoie'],
    "Provence-Alpes-Côte d'Azur": ['Bouches-du-Rhône', 'Alpes-Maritimes', 'Var', 'Vaucluse'],
    'Occitanie': ['Haute-Garonne', 'Hérault', 'Gard', 'Pyrénées-Orientales'],
    'Grand Est': ['Bas-Rhin', 'Haut-Rhin', 'Moselle', 'Marne', 'Meurthe-et-Moselle'],
    'Bretagne': ['Ille-et-Vilaine', 'Finistère', 'Morbihan', "Côtes-d'Armor"],
    'Normandie': ['Seine-Maritime', 'Calvados', 'Manche', 'Eure'],
    'Bourgogne-Franche-Comté': ["Côte-d'Or", 'Doubs', 'Saône-et-Loire'],
    'Centre-Val de Loire': ['Loiret', 'Indre-et-Loire', 'Cher'],
    'Corse': ['Corse-du-Sud', 'Haute-Corse'],
}

# street types: surface (folded) -> canonical token. Country-scoped sets are merged per country.
US_STREET = {
    'street': 'st', 'st': 'st', 'str': 'st', 'saint': 'st', 'road': 'rd', 'rd': 'rd', 'avenue': 'ave', 'ave': 'ave',
    'av': 'ave', 'drive': 'dr', 'dr': 'dr', 'drv': 'dr', 'lane': 'ln', 'ln': 'ln', 'boulevard': 'blvd', 'blvd': 'blvd',
    'court': 'ct', 'ct': 'ct', 'circle': 'cir', 'cir': 'cir', 'place': 'pl', 'pl': 'pl', 'parkway': 'pkwy', 'pkwy': 'pkwy',
    'highway': 'hwy', 'hwy': 'hwy', 'trail': 'trl', 'trl': 'trl', 'terrace': 'ter', 'ter': 'ter', 'square': 'sq', 'sq': 'sq',
    'point': 'pt', 'pt': 'pt', 'crossing': 'xing', 'xing': 'xing', 'heights': 'hts', 'hts': 'hts', 'mount': 'mt', 'mt': 'mt',
    'center': 'ctr', 'ctr': 'ctr', 'centre': 'ctr', 'expressway': 'expy', 'expy': 'expy', 'freeway': 'fwy', 'fwy': 'fwy',
    'cove': 'cv', 'cv': 'cv', 'north': 'n', 'n': 'n', 'south': 's', 's': 's', 'east': 'e', 'e': 'e', 'west': 'w', 'w': 'w',
    'northeast': 'ne', 'ne': 'ne', 'northwest': 'nw', 'nw': 'nw', 'southeast': 'se', 'se': 'se', 'southwest': 'sw', 'sw': 'sw',
    'apartment': 'apt', 'apt': 'apt', 'suite': 'ste', 'ste': 'ste', 'floor': 'fl', 'fl': 'fl', 'flr': 'fl',
    'building': 'bldg', 'bldg': 'bldg', 'fort': 'ft', 'ft': 'ft',
}
FR_STREET = {
    'rue': 'rue', 'r': 'rue', 'boulevard': 'bd', 'bd': 'bd', 'blvd': 'bd', 'avenue': 'av', 'av': 'av', 'ave': 'av',
    'allee': 'all', 'allees': 'all', 'all': 'all', 'alle': 'all', 'chemin': 'ch', 'ch': 'ch', 'che': 'ch', 'place': 'pl', 'pl': 'pl',
    'cours': 'crs', 'crs': 'crs', 'route': 'rte', 'rte': 'rte', 'impasse': 'imp', 'imp': 'imp', 'quai': 'qu', 'qu': 'qu',
    'square': 'sq', 'sq': 'sq', 'residence': 'res', 'res': 'res', 'faubourg': 'fg', 'fg': 'fg', 'fbg': 'fg',
    'saint': 'st', 'st': 'st', 'sainte': 'ste', 'ste': 'ste', 'cite': 'cite', 'lotissement': 'lot', 'lot': 'lot',
    'passage': 'pass', 'pass': 'pass', 'promenade': 'prom', 'prom': 'prom', 'esplanade': 'esp', 'esp': 'esp',
}
IN_STREET = dict(US_STREET)
IN_STREET.update({'marg': 'marg', 'nagar': 'nagar', 'sector': 'sec', 'sec': 'sec', 'opp': 'opp', 'opposite': 'opp',
                  'near': 'nr', 'nr': 'nr', 'behind': 'bh', 'bh': 'bh', 'cross': 'cross', 'main': 'main'})

ORDINAL_WORDS = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
                 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth',
                 'eighteenth', 'nineteenth', 'twentieth']
ORDINAL_MAP = {w: str(i + 1) for i, w in enumerate(ORDINAL_WORDS)}
ORDINAL_MAP.update({'premier': '1', 'premiere': '1', 'deuxieme': '2', 'troisieme': '3'})

# legal forms: folded, dot-free surface -> canonical
LEGAL_CANON = {
    'private': 'PVT', 'pvt': 'PVT', 'prvt': 'PVT', 'pte': 'PVT', 'limited': 'LTD', 'ltd': 'LTD', 'ltda': 'LTD',
    'llc': 'LLC', 'inc': 'INC', 'incorporated': 'INC', 'corp': 'CORP', 'corporation': 'CORP', 'co': 'CO', 'company': 'CO',
    'llp': 'LLP', 'pllc': 'PLLC', 'pc': 'PC', 'lp': 'LP', 'plc': 'PLC', 'public': 'PUBLIC', 'sarl': 'SARL', 'sas': 'SAS',
    'sasu': 'SASU', 'eurl': 'EURL', 'sci': 'SCI', 'sa': 'SA', 'ei': 'EI', 'snc': 'SNC', 'cie': 'CIE', 'opc': 'OPC',
    'gmbh': 'GMBH', 'pa': 'PA',
}
CONNECTIVES = {'and', 'of', 'the', 'for', 'in'}
FR_CONNECTIVES = {'de', 'du', 'des', 'la', 'le', 'les', 'et', 'd', 'l', 'a', 'au', 'aux', 'en'}
COUNTRY_TAGS = ('india', 'france')
HONORIFICS = ('the the', 'the', 'mr', 'mrs', 'ms', 'dr', 'm/s', 'sri', 'shri', 'smt', 'shree')
# seed name filler (6.2) -- the learned list replaces/extends this
SEED_FILLER = {
    'US': {'center', 'services', 'service', 'partners', 'enterprises', 'trust', 'group', 'holdings', 'district', 'council',
           'commission', 'association', 'board', 'federation', 'foundation', 'trading', 'society', 'authority'},
    'India': {'center', 'services', 'service', 'partners', 'enterprises', 'trading', 'group', 'holdings', 'industries',
              'infratech', 'society'},
    'France': {'groupe', 'developpement', 'participations', 'holding', 'distribution', 'international', 'fils', 'associes',
               'services', 'centre'},
}
# seed address generator words (6.3)
SEED_ADDR_FILLER = {
    'US': {'city', 'cdp', 'township', 'twp', 'village', 'of', 'town', 'borough', 'pmb', 'po', 'box', 'null', 'unit', 'hash'},
    'India': {'door', 'no', 'plot', 'hn', 'h', 'hno', 'block', 'city', 'region', 'hq', 'null'},
    'France': {'no', 'n'},
}
