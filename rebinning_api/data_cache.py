from pathlib import Path
import requests
import os

NEXUS_FOLDER = Path("cache/nexus_files").absolute()
METADATA_ENDPOINT = "https://ncnr.nist.gov/ncnrdata/metadata/api/v1"
NCNRDATA_ENDPOINT = "https://ncnr.nist.gov/pub/ncnrdata/"
#NCNRDATA_ENDPOINT = "https://charlotte.ncnr.nist.gov/pub/ncnrdata/"

def nexus_lookup(nexusfile):
    """Lookup the download path for a nexus file given its name"""
    # Need cycle and experiment ID to retrieve nexus file.
    url = METADATA_ENDPOINT + "/datafiles"
    print(f"Finding location of {nexusfile} using {url}")
    r = requests.get(url, params={"filename": nexusfile})
    if not r.ok:
        raise RuntimeError(f"Nexus lookup <{url}?filename={nexusfile}> failed.")
    location = r.json()[0]["localdir"]
    #print("at", location)
    return location

def nexus_url(datapath, nexusfile):
    nexus_url = NCNRDATA_ENDPOINT + datapath + "/" + nexusfile
    return nexus_url

def cache_url(url, cachedir, filename=None, refresh=False):
    """Lookup the download path for a nexus file given its name"""
    # TODO: possible filename collisions
    if filename is None:
        filename = url.rsplit('/', 1)[-1]
    cachedir.mkdir(parents=True, exist_ok=True)
    fullpath = cachedir / filename
    if refresh or not fullpath.exists():
        print(f"Fecthing {url} into {filename}")
        r = requests.get(url)
        if not r.ok:
            raise RuntimeError(f"Fetch <{url}> failed.")
        open(fullpath, 'wb').write(r.content)
        #print(f"fetched {filename}")
    return fullpath
