INSTRUMENTS = {
    'NG3-VSANS': 'vsans',
    'NCNR Candor': 'candor',
    'SANS:NGB30': 'ngb30msans',
    'SANS:NG7': 'ng7sans',
    'SANS:NGB': '10msans',
    '10m Sans': '10msans',
}

def lookup_instrument(entry):
    if 'instrument/name' in entry:
        name = entry['instrument/name'][0].decode('utf8')
        return INSTRUMENTS[name]
    elif 'DAS_logs/experiment/instrument' in entry:
        name = entry['DAS_logs/experiment/instrument'][0].decode('utf8')
        return INSTRUMENTS[name]
    return None