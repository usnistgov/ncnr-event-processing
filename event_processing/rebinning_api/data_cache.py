from pathlib import Path
import os

import requests
import h5py
from importlib import resources

# 'my_package.certs' is the dot-notation to your folder
# 'server_chain.pem' is the actual file


# Use it in your request
#response = requests.get('https://your-server.com', verify=str(cert_path))

#print(response.status_code)

CACHE_ROOT = Path.cwd() / "cache" # default to current directory / cache
NEXUS_FOLDER = (CACHE_ROOT / "nexus_files").absolute()
METADATA_ENDPOINT = "https://charlotte.ncnr.nist.gov/ncnrdata/metadata/api/v1"
METADATA_CERTFILE_NAME = "charlotte-ncnr-nist-gov-chain.pem"
NCNRDATA_ENDPOINT = "https://ncnr.nist.gov/pub/ncnrdata/"
#NCNRDATA_ENDPOINT = "https://charlotte.ncnr.nist.gov/pub/ncnrdata/"

pkg_files = resources.files('event_processing.certs')
cert_path = pkg_files / METADATA_CERTFILE_NAME

def search_filename(nexusfile):
    """Lookup the download path for a nexus file given its name"""
    # Need cycle and experiment ID to retrieve nexus file.
    url = METADATA_ENDPOINT + "/datafiles"
    print(f"Finding location of {nexusfile} using {url}")
    r = requests.get(url, params={"filename": nexusfile}, verify=str(cert_path))
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
    print(f"getting file at fullpath {fullpath}")
    if refresh or not fullpath.exists():
        print(f"Fetching {url} into {filename}")
        r = requests.get(url)
        if not r.ok:
            raise RuntimeError(f"Fetch <{url}> failed.")
        open(fullpath, 'wb').write(r.content)
        #print(f"fetched {filename}")
    return fullpath

def configure(cache_root):
    """Set the base cache directory; nexus files are cached in <cache_root>/nexus_files."""
    global CACHE_ROOT, NEXUS_FOLDER
    CACHE_ROOT = Path(cache_root)
    NEXUS_FOLDER = (CACHE_ROOT / "nexus_files").absolute()

def load_nexus(filename, datapath=None, refresh=False):
    fullpath = NEXUS_FOLDER / filename
    if refresh or not fullpath.exists():
        if datapath is None:
            datapath = search_filename(filename)
        url = nexus_url(datapath, filename)
        fullpath = cache_url(url, NEXUS_FOLDER, filename=filename, refresh=refresh)
    else:
        print(f"Loading nexus file from cache: {fullpath}")
    return h5py.File(fullpath)
