from pathlib import Path
import requests
import os

import h5py

INSTRUMENT_FROM_ID = {
    b'NG3-VSANS': 'vsans',
}

def eventfiles_from_nexus(nexusfile): #, events_folder=EVENTS_FOLDER):
    nexus = h5py.File(nexusfile, mode="r")
    files = set()
    for name, entry in nexus.items():
        instrument_id = entry['instrument/name'][()][0]
        #print(instrument_id)
        instrument = INSTRUMENT_FROM_ID[instrument_id]
        for device, group in entry['instrument'].items():
            if 'event_file_name' in group:
                files.add(group['event_file_name'][0].decode())
    return instrument, files

EVENTS_FOLDER = "event_files"
EVENTS_ENDPOINT = "http://nicedata.ncnr.nist.gov/eventfiles"


def fetch_eventfile(instrument, eventfile, events_folder=EVENTS_FOLDER, overwrite=False):
    #print("events folder", events_folder)
    events_folder = Path(events_folder)
    events_folder.mkdir(parents=True, exist_ok=True)
    fullpath = events_folder / eventfile
    if overwrite or not fullpath.exists():
        url = EVENTS_ENDPOINT
        #print(f"retrieving eventfile {eventfile} from {url}")
        r = requests.get(url, params={"instrument": instrument, "filename": eventfile})
        if r.ok:
            open(fullpath, 'wb').write(r.content)
            print(f"Fetched {eventfile}")
        else:
            # TODO: maybe store an empty missing file?
            print(f"Failure: {r.status_code} '{r.reason}' during {url}?instrument={instrument}&filename={eventfile}")

def retrieve_events(nexus_path, events_folder=EVENTS_FOLDER, overwrite=False):
    instrument, files = eventfiles_from_nexus(nexus_path)
    #print(f"Event files in {nexus_path}", files)
    paths = []
    for eventfile in sorted(files):
        fetch_eventfile(instrument, eventfile, events_folder=events_folder, overwrite=overwrite)
        paths.append(Path(events_folder) / eventfile)
    return paths


NEXUS_FOLDER = Path("nexus_files").absolute()
METADATA_ENDPOINT = "https://ncnr.nist.gov/ncnrdata/metadata/api/v1"
NCNRDATA_ENDPOINT = "https://ncnr.nist.gov/pub/ncnrdata/"

def nexus_url(nexusfile):
    """Lookup the download path for a nexus file given its name"""
    # Need cycle and experiment ID to retrieve nexus file.
    url = METADATA_ENDPOINT + "/datafiles"
    print(f"Finding location of {nexusfile} using {url}")
    r = requests.get(url, params={"filename": nexusfile})
    if not r.ok:
        raise RuntimeError(f"Nexus lookup <{url}?filename={nexusfile}> failed.")
    location = r.json()[0]["localdir"]
    nexus_url = NCNRDATA_ENDPOINT + location + "/" + nexusfile
    return nexus_url

def cache_url(url, cachedir, filename=None):
    """Lookup the download path for a nexus file given its name"""
    # TODO: possible filename collisions
    if filename is None:
        filename = url.rsplit('/', 1)[-1]
    cachedir.mkdir(parents=True, exist_ok=True)
    fullpath = cachedir / filename
    if not fullpath.exists():
        print(f"Fecthing {url} into {filename}")
        r = requests.get(url)
        if not r.ok:
            raise RuntimeError(f"Fetch <{url}> failed.")
        open(fullpath, 'wb').write(r.content)
        #print(f"fetched {filename}")
    return fullpath

def main():
    import sys
    filenames = ["sans69040.nxs.ngv"] if len(sys.argv) == 1 else sys.argv[1:]
    for filename in filenames:
        nexus_path = cache_url(nexus_url(filename), NEXUS_FOLDER, filename)
        paths = retrieve_events(nexus_path)
    #print(paths)

if __name__ == '__main__':
    main()
