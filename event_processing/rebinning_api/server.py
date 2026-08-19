import asyncio
import base64
from collections import deque
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from typing import Annotated
import uuid
import logging
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException, status

#from dateutil.parser import isoparser
import numpy as np

from . import models
from . import hst
from . import nexus_util
from . import event_capture
from . import binning
from . import data_cache
from . import event_cache

# Use the logger from uvicorn so we get pretty formatting
logger = logging.getLogger("uvicorn.error")

CACHE = None
CACHE_PATH = "/tmp/event-processing"
# Base folder for the on-disk nexus/event cache; nexus files live in
# <CACHE_ROOT>/nexus_files, event files in <CACHE_ROOT>/hst_files, and
# persisted cleaned events live in <CACHE_ROOT>/events_cache.
CACHE_ROOT: Path = Path.cwd() / "cache"
# When True (via "--auto-hst-file"), events are read from local .hst files
# under CACHE_ROOT / "hst_files" instead of being fetched from the live
# kafka stream.
AUTO_HST_FILE: bool = False
# When True, nexus files are re-searched and re-downloaded even if already
# present in CACHE_ROOT / "nexus_files".
REFRESH_CACHE: bool = False
# When set (via "rebin --events-file"), cleaned events are loaded from this
# user-provided events+nexus file before falling back to the normal cache
# lookup / live fetch. Set from the CLI, not from the web API.
EVENTS_FILE_OVERRIDE: Path | None = None
# When set (via "rebin --event-file" / "save-events --event-file",
# repeatable), local .hst event files are loaded from these explicit paths
# instead of being auto-discovered via the nexus entry's recorded
# eventFileName. Takes priority over AUTO_HST_FILE auto-discovery.
LOCAL_HST_FILES: list[Path] | None = None
CACHE_VERSION = "0.2"
CACHE_SIZE = int(100e9) # 100 GB
CACHE_ITEMS = 100 # max number of items, if not using items size in cache
DOWNLOAD_HISTORY_SIZE = 10000 # number of downloads to keep track of

app = FastAPI()
# app.add_middleware(GZipMiddleware, minimum_size=1000)
# app.add_middleware(MessagePackMiddleware)

# Mount the static files directory
current_path = Path(__file__).parent
static_path = current_path.parent / "rebinning_client" / "web-client" / "dist"
app.mount("/static", StaticFiles(directory=static_path, html=True), name="frontend")


origins = [
    "*",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# dist_path = Path(__file__).resolve().parents[1] / "rebinning_client" / "web-client" / "dist"

# # Mount the directory so that any request that doesn't match an API route
# # falls back to the static files (index.html, JS, CSS, assets, …)
# app.mount("/static/", StaticFiles(directory=str(dist_path), html=True), name="frontend")

def disk_cache():
    import diskcache
    cache = diskcache.Cache(
        CACHE_PATH,
        size_limit=CACHE_SIZE,
        eviction_policy='least-recently-used',
        )
    return cache
def mem_cache():
    import pylru
    cache = pylru.lrucache(CACHE_ITEMS)
    return cache
CACHE = mem_cache()

_ = '''
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

# from msgpack_asgi import MessagePackMiddleware
from msgpack import packb

from models import VSANS, RebinUniformCount, RebinUniformWidth, NumpyArray, RebinnedData, InstrumentName

default_instrument = VSANS()
default_rebinning = RebinUniformCount(num_bins=10)

def make_dummy_response():
    points = 12
    # detector_shapes = [(48,128), (128, 48), (512,480)]
    detector_shapes = [(2,3), (2, 3), (4,5)]
    detectors = dict([(str(i), NumpyArray.from_ndarray(np.random.rand(points, *s))) for i, s in enumerate(detector_shapes)])
    devices = {"A3": NumpyArray.from_ndarray(np.arange(points, dtype="<i4"))}
    return RebinnedData(detectors=detectors, devices=devices)


@app.post("/rebin")
@app.get("/rebin")
def rebin(
    instrument_def: Union[VSANS, None] = default_instrument,
    rebin_parameters: Union[RebinUniformWidth, RebinUniformCount] = default_rebinning
) -> RebinnedData:
    print(instrument, rebin_parameters)
    response_data = {"instrument": instrument.model_dump(), "rebin_parameters": rebin_parameters.model_dump(), "result": make_dummy_response().model_dump()}
    return Response(content=packb(response_data), media_type="application/msgpack")


def bundle(reply):
    data = serial.dumps(reply)
    #print("encoded reply", data)
    return Response(content=data, media_type="application/json")

def unbundle(reply):
    data = reply.body
    return serial.loads(data)
'''

# ===================================
@app.post("/metadata")
def get_metadata(request: models.Measurement):
    logger.debug(f"get_metadata {request}")
    point = request.point
    try:
        entry = nexus_util.open_nexus_entry(request, refresh=REFRESH_CACHE)
    except Exception as exc:
        detail = f"Unable to load {request.path}/{request.filename}.\n   {exc}"
        logger.error(detail)
        # Return a proper FastAPI error response when metadata cannot be loaded
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    #print("entry", entry, entry.parent)
    # Go up to the root to get the other NXentry items in the file.
    entries = nexus_util.nexus_entries(entry.parent)
    timestamp = entry['start_time'][0].decode('ascii')
    duration = float(entry['control/count_time'][point])
    if 'DAS_logs/trajectory/liveScanLength' in entry:
        numpoints = entry['DAS_logs/trajectory/liveScanLength'][0]
    elif 'DAS_logs/trajectory/length' in entry:
        numpoints = entry['DAS_logs/trajectory/length'][0]
    else:
        raise ValueError(f"can't extract numpoints")
    replacement = nexus_util.nexus_detector_replacement(entry)
    #print("detector data links", replacement)
    detectors = list(replacement.keys())
    # TODO: determine mode for sweep device and triggers
    event_mode = 'time' # not written yet... this is the default
    trigger_interval = 0.0
    # TODO: lookup sweep controls from nexus file
    logs = {}
    sweep = None

    # TODO: If no event data then return an API exception

    reply = models.MetadataReply(
        measurement=request,
        entries=entries,
        numpoints=numpoints,
        # TODO: who is parsing the timestamp ??
        # timestamp=isoparser().isoparse(timestamp),
        timestamp=timestamp,
        duration=duration,
        trigger_interval=trigger_interval,
        detectors=detectors,
        logs=logs,
        event_mode=event_mode,
        sweep=sweep,
    )
    return reply

# ===================================
@app.post("/summary_time")
def get_summary_time(request: models.SummaryTimeRequest):
    return get_summary(request.measurement, request.bins)

def get_summary(measurement, bins):
    result = bin_events(measurement, bins, summary=True)
    devices = {}
    monitor = None
    reply = models.SummaryReply(
        measurement=measurement,
        bins=bins,
        duration=result['count_time'],
        counts=result['detectors'],
        monitor=monitor,
        devices=devices,
    )
    return reply

# ===================================
@app.post("/timebin/frame/{index}")
def get_timebin_frame(index: int, request: models.SummaryTimeRequest):
    return get_frame_range(request.measurement, request.bins, index, index+1)

@app.post("/timebin/frame/{start}-{end}")
def get_timebin_frame_range(start: int, end: int, request: models.SummaryTimeRequest):
    return get_frame_range(request.measurement, request.bins, start, end)

def get_frame_range(measurement, bins, start, end):
    binned = bin_events(measurement, bins, summary=False)
    counts = binned['detectors']
    data = {k: v[start:end] for k, v in counts.items()}
    reply = models.FrameReply(
        data=data,
    )
    return reply

# ===================================
@app.post("/timebin/nexus")
def get_timebin_nexus(request: models.SummaryTimeRequest):
    data, outfile, mimetype = get_nexus(request.measurement, request.bins, request.split)
    reply = models.NexusReply(mimetype=mimetype, filename=outfile, base64_data=base64.b64encode(data))
    return reply

COMPLETED_DOWNLOADS = deque(maxlen=DOWNLOAD_HISTORY_SIZE)
PROCESSING_ERRORS = {}

@app.get('/timebin/nexus_download_status/{download_id}')
def get_download_status(download_id: str):
    error_state = PROCESSING_ERRORS.pop(download_id, None)
    if error_state is not None:
        return { "complete": False, "error": error_state }
    else:
        return { "complete": COMPLETED_DOWNLOADS.count(download_id) > 0 }

@app.post('/timebin/nexus_download')
async def download_nexus_form(request_str: Annotated[str, Form()], download_id: Annotated[str, Form()] = ''):
    """post request coming from HTML form, that can trigger a download """
    request_dict = json.loads(request_str)
    request = models.SummaryTimeRequest(**request_dict)
    coro = asyncio.to_thread(get_nexus, request.measurement, request.bins, request.split)
    try:
        data, filename, mimetype = await coro
        buffer_size = 2**16 # 64K
        async def result_streamer():
            with io.BytesIO(data) as mem_fd:
                buffer = mem_fd.read(buffer_size)
                while buffer:
                    yield buffer
                    buffer = mem_fd.read(buffer_size)
            if (download_id != ''):
                COMPLETED_DOWNLOADS.append(download_id)

        last_updated_pattern = "%a, %d %b %Y %H:%M:%S GMT"
        last_modified = datetime.datetime.strftime(datetime.datetime.now(datetime.timezone.utc), last_updated_pattern)
        content_length = str(len(data))
        etag = hashlib.md5(f'{last_modified}-{content_length}'.encode(), usedforsecurity=False).hexdigest()
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"',
            'Content-Type': mimetype,
            'Content-Length': content_length,
            'Last-Modified': last_modified,
            'ETag': etag,
            'Access-Control-Allow-Origin': '*',
        }
        return StreamingResponse(result_streamer(), headers=headers)
    except Exception as e:
        if (download_id != ''):
            PROCESSING_ERRORS[download_id] = str(e)
        raise e


def ensure_events_download(measurement: models.Measurement, hst_files: list[Path] | None = None) -> Path:
    """
    Helper for the events download endpoint: makes sure the on-disk
    events-cache copy of the nexus file has this measurement's cleaned
    events persisted, and returns its path.

    hst_files overrides the LOCAL_HST_FILES global for this call only
    (see ensure_cleaned_events) -- lets scripts/notebooks pass explicit
    .hst paths without touching global state.
    """
    entry = nexus_util.open_nexus_entry(measurement, refresh=REFRESH_CACHE)
    try:
        ensure_cleaned_events(measurement, entry, hst_files=hst_files)
    finally:
        entry.file.close()
    return event_cache.events_cache_path(measurement.filename)


@app.post('/events/download')
async def download_events_form(request_str: Annotated[str, Form()], download_id: Annotated[str, Form()] = ''):
    """ post request coming from HTML form, that can trigger a download of the events+nexus file """
    request_dict = json.loads(request_str)
    measurement = models.Measurement(**request_dict)
    coro = asyncio.to_thread(ensure_events_download, measurement)
    try:
        path = await coro
        orig_filename = measurement.filename
        orig_path = Path(orig_filename)
        file_suffixes = ''.join(orig_path.suffixes)
        file_stem = re.sub(f"{file_suffixes}$", '', orig_filename)
        new_filename = f"{file_stem}_events{file_suffixes}"
        buffer_size = 2**16 # 64K
        content_length = str(path.stat().st_size)
        async def result_streamer():
            with open(path, 'rb') as fd:
                buffer = fd.read(buffer_size)
                while buffer:
                    yield buffer
                    buffer = fd.read(buffer_size)
            if (download_id != ''):
                COMPLETED_DOWNLOADS.append(download_id)

        last_updated_pattern = "%a, %d %b %Y %H:%M:%S GMT"
        last_modified = datetime.datetime.strftime(datetime.datetime.now(datetime.timezone.utc), last_updated_pattern)
        etag = hashlib.md5(f'{last_modified}-{content_length}'.encode(), usedforsecurity=False).hexdigest()
        headers = {
            'Content-Disposition': f'attachment; filename="{new_filename}"',
            'Content-Type': 'application/x-hdf5',
            'Content-Length': content_length,
            'Last-Modified': last_modified,
            'ETag': etag,
            'Access-Control-Allow-Origin': '*',
        }
        return StreamingResponse(result_streamer(), headers=headers)
    except Exception as e:
        if (download_id != ''):
            PROCESSING_ERRORS[download_id] = str(e)
        raise e


def get_nexus(measurement: models.Measurement, bins, split: bool = False):
    """
    Helper for nexus writer endpoints, which takes the binned detectors, etc.
    and produces an updated nexus file.
    """
    # TODO: only supports single point files for now
    # TODO: support files with different binning at each point
    # Could split points across files or across entries within a file
    # TODO: maybe provide "explode" option to split each bin to a different file
    # TODO: check that there is only one entry with one point
    # TODO: replace monitor, and any devices that are binned
    # logger.debug(f"get_nexus {measurement.filename}")
    binned = bin_events(measurement, bins, summary=False)
    entry = nexus_util.open_nexus_entry(measurement, refresh=REFRESH_CACHE)
    try:
        data, outfile, mimetype = nexus_util.nexus_dup(entry, binned, bins, split=split, filename=measurement.filename)
    finally:
        entry.file.close()

    return data, outfile, mimetype


def ensure_cleaned_events(measurement: models.Measurement, entry, hst_files: list[Path] | None = None):
    """
    Return cleaned (time-of-flight corrected) events for measurement+entry,
    loading them from the on-disk events cache if already persisted there,
    or fetching and cleaning them from the raw source (kafka / local event
    files) and persisting the result otherwise.

    hst_files, when given, takes priority over the AUTO_HST_FILE / live-kafka
    fallback, same as LOCAL_HST_FILES (the global that "serve" and
    "rebin" populate from their own --hst-file CLI flag ahead of time, since
    there's no per-request way to pass it through the web API).
    """
    from timeit import default_timer as tic; T0 = tic()
    events = None
    if EVENTS_FILE_OVERRIDE is not None:
        print(f"{tic()-T0:.6f}: loading events from {EVENTS_FILE_OVERRIDE}")
        events = event_cache.load_cached_events(measurement, entry.name, path=EVENTS_FILE_OVERRIDE)
    elif not REFRESH_CACHE:
        events = event_cache.load_cached_events(measurement, entry.name)
    if events is not None:
        return events

    if hst_files is None:
        hst_files = LOCAL_HST_FILES
    if hst_files:
        print(f"{tic()-T0:.6f}: loading events from {hst_files}")
        raw_events = hst.events_manager_from_paths(entry, hst_files)
    elif AUTO_HST_FILE:
        events_folder = CACHE_ROOT / "hst_files"
        print(f"{tic()-T0:.6f}: loading events from {events_folder}")
        raw_events = hst.events_manager_from_files(entry, events_folder=events_folder)
    else:
        print(f"{tic()-T0:.6f}: fetching raw events for {entry.file.filename}")
        # event_capture.setup()  # in case it hasn't already been setup for sim
        raw_events = event_capture.fetch_events_to_memory(entry, measurement.point)
    print(f"{tic()-T0:.6f}: correcting events")
    event_capture.event_cleanup(entry, raw_events)
    events = raw_events._cleaned_fields
    meta = dict(start=raw_events.start, stop=raw_events.stop, arm=raw_events.arm, disarm=raw_events.disarm)
    event_cache.save_cleaned_events(measurement, entry.name, events, meta=meta)
    print(f"{tic()-T0:.6f}: cached events to disk")
    return events


def bin_events(measurement: models.Measurement, bins: models.TimeBins, summary=False):
    from timeit import default_timer as tic; T0 = tic()
    if bins.mode != "time":
        raise NotImplementedError("only time-mode binning implemented for now")

    key = (request_key(measurement), request_key(bins))
    #print("Key:", key)
    # Increment version number if the data changes
    binned_key = (*key, "binned", "v1")   # binning keyed by both entry and bin spec
    summed_key = (*key, "summed", "v1")
    if binned_key not in CACHE:
        #print("processing events")
        entry = nexus_util.open_nexus_entry(measurement, refresh=REFRESH_CACHE)
        try:
            # CRUFT: we are allowing some old vsans histograms to run for demo purposes.
            # Skipped when reading events from a local cache, since that path already
            # handles old-format vsans files via the full event_cleanup/binning pipeline.            
            events = ensure_cleaned_events(measurement, entry)
            print(f"{tic()-T0:.6f}: binning")
            result = binning.bin(entry, measurement.point, bins, events)
            print(f"{tic()-T0:.6f}: binned")
            #binned = _bin_by_time(entry, events, bins)
        finally:
            entry.file.close()
        CACHE[binned_key] = result
        print(f"{tic()-T0:.6f}: cached")

    if not summary:
        result = CACHE[binned_key]
        print(f"{tic()-T0:.6f}: retrieved bins")
        return result

    if summed_key not in CACHE:
        print(f"{tic()-T0:.1f}: accumulating events")
        binned = CACHE[binned_key]
        summed = {}
        for detector, data in binned['detectors'].items():
            # Sum the individual frames
            total = np.reshape(data, (data.shape[0], -1)).sum(axis=1)
            #print("in summary", detector, data, total)
            summed[detector] = total
        result = dict(detectors=summed, count_time=binned['count_time'])
        print(f"{tic()-T0:.1f}: summed")
        CACHE[summed_key] = result
        print(f"{tic()-T0:.1f}: cached summary")
    result = CACHE[summed_key]
    print(f"{tic()-T0:.1f}: retrieved summary")
    return result


# CRUFT: code for old-style histograms
def _bin_by_time_old_vsans(entry, bins):
    #point = measurement.point # ignored in vsans rebin old
    edges = bins.edges
    #mask = bins.mask # ignored in vsans rebin old
    # TODO: not binning monitor or devices
    binned = {}
    from timeit import default_timer as tic; T0 = tic()
    for z, detector in (("front", "FL"), ("middle", "ML")):
        print(f"{tic()-T0:.1f}: fetching", detector)
        eventfile = entry[f'instrument/detector_{detector}/event_file_name'][0].decode()
        eventpath = hst.fetch_eventfile("vsans", eventfile)
        if not Path(eventpath).exists():
            print("missing", eventpath)
            continue
        print(f"{tic()-T0:.1f}: loading", detector)
        events = hst.VSANSEvents(eventpath)
        #events._repeat(10)
        # TODO: correct for time of flight
        # TODO: elide events in mask
        print(f"{tic()-T0:.1f}: binning", detector)
        partial_counts, _ = events.rebin(edges)
        print(f"{tic()-T0:.1f}: binned", detector)
        for xy, data in partial_counts.items():
            # form detector_FB, etc. from first letter of names
            name = f"detector_{z[0].upper()}{xy[0].upper()}"
            binned[name] = data
    result = dict(detectors=binned, count_time=np.diff(bins.edges))
    return result


def request_key(request):
    data = request.model_dump_json()
    digest = hashlib.sha1(data.encode('utf-8')).hexdigest()
    return digest

# TODO: how do we clear the cache when upgrading the application?

def check(filename=None, verbose=False):
    from . import client

    if filename is None:
        filename = "sans68869.nxs.ngv"
    measurement = models.Measurement(filename=filename)
    metadata = get_metadata(measurement)
    if verbose: print("metadata", metadata)
    bins = client.time_linbins(metadata, interval=5)
    #print("num bins", bins.edges.size)
    request = models.SummaryTimeRequest(measurement=measurement, bins=bins)
    summary = get_summary_time(request)
    if verbose: print("summary", summary)
    index = np.searchsorted(bins.edges, 500.)
    r_one = get_timebin_frame(index, request)
    if verbose: print("frame", index, {k: v.shape for k, v in r_one.data.items()})
    r_many = get_timebin_frame_range(index, index+2, request)
    detector = "detector_FL"
    if verbose: print(r_one.data[detector].shape, r_many.data[detector].shape)
    assert (r_one.data[detector][0] == r_many.data[detector][0]).all()
    reply = get_timebin_nexus(request)
    with open('/tmp/sample.hdf', 'wb') as fd:
        fd.write(base64.b64decode(reply.base64_data))

def check2():
    path = "vsans/202102/27861/data"
    nexusfile = "sans72109.nxs.ngv"
    # event_capture.setup()
    with event_capture.kafka_consumer() as consumer:
        event_capture.fetch_events_for_file(consumer, nexusfile, datapath=path)

def check3():
    from . import client
    # event_capture.setup()
    # path = "vsans/202102/27861/data"
    # nexusfile = "sans72110.nxs.ngv"
    path = "vsans/202102/nonims6/data"
    nexusfile = "sans72222.nxs.ngv"
    measurement = models.Measurement(filename=nexusfile, path=path, point=0)
    metadata = get_metadata(measurement)
    bins = client.time_linbins(metadata, interval=501)
    request = models.SummaryTimeRequest(measurement=measurement, bins=bins)
    reply = get_timebin_nexus(request)
    with open('/tmp/end-to-end.hdf', 'wb') as fd:
        fd.write(base64.b64decode(reply.base64_data))

# TODO: cache a version number, clearing the cache if there is a version mismatch
usage = """
Usage: server [clear|check|check2|check3] [-f FILENAME]

clear: Empties any caches associated with the data. This should happen
    automatically if you bump server.CACHE_VERSION to a new value, but you
    may still want to clear the version manually when e.g., testing speed.
check: Runs some simple event processing to make sure that the pieces
    work together. This is a development tool acting as a poor substitute
    for a proper test harness. Optionally specify a filename with -f.
check2: Runs check2 (see function).
check3: Runs check3 (see function).

To run the actual server for responding to web requests use uvicorn:

    uvicorn event_processing.rebinning_api.server:app
"""

def configure_data_source(cache: str = None, refresh: bool = False, auto_hst_file: bool = False, hst_files: list[str] = None, events_file: str = None):
    """Configure the global cache/event-source state shared by every CLI action that reads nexus/event data (serve, rebin, save-events)."""
    global CACHE_ROOT, AUTO_HST_FILE, REFRESH_CACHE, EVENTS_FILE_OVERRIDE, LOCAL_HST_FILES
    CACHE_ROOT = Path(cache) if cache else Path.cwd() / "cache"
    AUTO_HST_FILE = auto_hst_file
    REFRESH_CACHE = refresh
    EVENTS_FILE_OVERRIDE = Path(events_file) if events_file else None
    LOCAL_HST_FILES = [Path(p) for p in hst_files] if hst_files else None
    data_cache.configure(CACHE_ROOT)
    event_cache.configure(CACHE_ROOT)

def cli_rebin(filename: str, path: str, preview: bool = False, interval: int = None, nbins: int = None, events_file: str = None, split: bool = False):
    """Handles the CLI execution for rebinning or launching the preview."""
    if preview:
        import webbrowser
        # Launch the Vue application. Ensure your frontend server port matches!
        url = f"http://localhost:8080/?filename={filename}&path={path}"
        print(f"Opening preview in browser: {url}")
        webbrowser.open(url)
        return

    from . import client
    print(f"Loading measurement for {filename}...")
    measurement = models.Measurement(filename=filename, path=path, point=0)

    print("Fetching metadata...")
    metadata = get_metadata(measurement)

    if interval is not None:
        print(f"Generating bins with interval {interval}...")
    else:
        print(f"Generating {nbins or 10} bins...")
    bins = client.time_linbins(metadata, interval=interval, nbins=nbins)
    request = models.SummaryTimeRequest(measurement=measurement, bins=bins, split=split)

    print("Processing events and generating Nexus output (this may take a moment)...")
    try:
        reply = get_timebin_nexus(request)
    except Exception as exc:
        raise
        print(f"cli_rebin error: {exc}")
        sys.exit(1)
    with open(reply.filename, 'wb') as fd:
        fd.write(base64.b64decode(reply.base64_data))
    print(f"Success! Rebinned file saved to ./{reply.filename}")

def cli_save_events(filename: str, path: str, hst_files: list[str] = None, point: int = 0, entry: int = 0, output: str = None):
    """Handles the CLI execution for fetching, cleaning, and persisting events without binning."""
    hst_paths = [Path(p) for p in hst_files] if hst_files else None

    measurement = models.Measurement(filename=filename, path=path, point=point, entry=entry)
    print(f"Fetching and cleaning events for {filename} (entry {entry}, point {point})...")
    cache_path = ensure_events_download(measurement, hst_files=hst_paths)
    print(f"Success! Events persisted to {cache_path}")

    if output:
        import shutil
        shutil.copyfile(cache_path, output)
        print(f"Copied events+nexus file to {output}")

def open_preview(host: str = 'localhost', port: int = 8000, filename: str = '', path: str = ''):
    import time
    import threading
    import webbrowser
    import urllib.request
    import urllib.error

    def wait_and_open():
        # Now that FastAPI serves the frontend, we use its port!
        base_url = f"http://{host}:{port}/static"
        
        # Append query parameters if a file was specified
        if filename:
            target_url = f"{base_url}/?filename={filename}&path={path}"
        else:
            target_url = base_url

        logger.info(f"Waiting for server to become ready at {base_url} ...")
        
        # Poll the server until it responds
        while True:
            try:
                # Attempt to connect to the server
                urllib.request.urlopen(base_url)
                break # If we get here, the server is up!
            except urllib.error.URLError:
                # Connection refused; sleep for a quarter-second and try again
                logger.warning(f"Server not ready yet... retrying in 250ms")
                time.sleep(0.25)
        
        logger.info(f"Server is up! Opening browser: {target_url}")
        webbrowser.open(target_url, new=0)
    
    # Start the polling thread
    threading.Thread(target=wait_and_open, daemon=True).start()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Event processing server and CLI utilities.')
    # Global log level for uvicorn (applies to any subcommand)
    parser.add_argument('--log-level', choices=['critical', 'error', 'warning', 'info', 'debug', 'trace'],
                        default='info', help='Set server log level')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # --- Shared argument groups, applicable to any command that reads
    # nexus/event data (serve, rebin, save-events) ---
    data_source_args = argparse.ArgumentParser(add_help=False)
    data_source_args.add_argument('--path', type=str, default='', help='Path to the data directory.')
    data_source_args.add_argument('--cache', type=str, default=None, help='Base folder for the local cache (default: ./cache). Nexus files are cached in <cache>/nexus_files; cleaned events are persisted to <cache>/events_cache; event files are read from <cache>/hst_files when local events are used.')
    data_source_args.add_argument('--refresh', action='store_true', help='Force re-searching/re-downloading/re-fetching data even if already present in the cache.')
    data_source_args.add_argument('--auto-hst-file', action='store_true', default=False, help='Automatically read event data from pre-populated files in <cache>/hst_files instead of the live kafka stream.')
    data_source_args.add_argument('--hst-file', dest='hst_files', action='append', default=None, metavar='PATH', help='Explicit path to a local .hst event file, bypassing nexus-based auto-discovery of event files by name. Repeat for VSANS (once for the front carriage, once for the middle carriage); pass once for SANS. Takes priority over --auto-hst-file auto-discovery.')

    preview_args = argparse.ArgumentParser(add_help=False)
    preview_args.add_argument('--preview', action='store_true', help='Open the result in the GUI browser.')

    # --- Command: serve ---
    parser_serve = subparsers.add_parser('serve', parents=[data_source_args, preview_args], help='Start the FastAPI backend server.')
    parser_serve.add_argument('--host', type=str, default='127.0.0.1', help='Host IP address to bind to.')
    parser_serve.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000).')
    parser_serve.add_argument('--filename', type=str, help='Filename to pre-load in the GUI on startup.')

    # --- Command: clear ---
    subparsers.add_parser('clear', help='Empties any caches associated with the data.')

    # --- Command: check / check2 / check3 ---
    parser_check = subparsers.add_parser('check', help='Run diagnostic checks.')
    parser_check.add_argument('-f', '--filename', help='Filename to use for check command')
    subparsers.add_parser('check2', help='Run check2 diagnostic.')
    subparsers.add_parser('check3', help='Run check3 diagnostic.')

    # --- Command: rebin ---
    parser_rebin = subparsers.add_parser('rebin', parents=[data_source_args, preview_args], help='Rebin a file directly from the CLI.')
    parser_rebin.add_argument('filename', type=str, help='Name of the nexus file (e.g., sans72222.nxs.ngv)')
    parser_rebin_bins = parser_rebin.add_mutually_exclusive_group()
    parser_rebin_bins.add_argument('--interval', type=int, default=None, help='Bin interval.')
    parser_rebin_bins.add_argument('--nbins', type=int, default=None, help='Number of bins (default: 10, used when neither --interval nor --nbins is given).')
    parser_rebin.add_argument('--events-file', type=str, default=None, help='Reload cleaned events from a previously saved events+nexus file (see "save-events") instead of fetching them live.')
    parser_rebin.add_argument('--split', action='store_true', help='Write a per‑bin ZIP archive (uses nexus_zip) instead of a single rebinned file.')

    # --- Command: save-events ---
    parser_save_events = subparsers.add_parser('save-events', parents=[data_source_args], help='Fetch, clean, and persist events for a measurement without binning.')
    parser_save_events.add_argument('filename', type=str, help='Name of the nexus file (e.g., sans72222.nxs.ngv)')
    parser_save_events.add_argument('--point', type=int, default=0, help='Point index within the measurement (default: 0)')
    parser_save_events.add_argument('--entry', type=int, default=0, help='Entry index within the nexus file (default: 0)')
    parser_save_events.add_argument('--output', type=str, default=None, help='Copy the resulting events+nexus file to this path.')

    args = parser.parse_args()

    # Apply the requested log level to uvicorn's loggers before the server starts.
    # uvicorn expects a string like "debug"; we set the Python logging level accordingly.
    level_name = args.log_level.upper()
    # Map textual level to logging constant (default INFO if unknown).
    log_level_val = getattr(logging, level_name, logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(log_level_val)
    logging.getLogger("uvicorn.access").setLevel(log_level_val)

    # Set the root logger so our own logger respects the same level. With level debug some
    # third party packages will get really noisy.
    #logging.getLogger().setLevel(log_level_val)

    if args.command == "serve":
        import uvicorn

        print(f"args: {args}")

        configure_data_source(
            cache=args.cache,
            refresh=args.refresh,
            auto_hst_file=args.auto_hst_file,
            hst_files=args.hst_files,
        )

        if args.preview:
            open_preview(host=args.host, port=args.port, filename=args.filename, path=args.path)

        # Start the server in a busy loop. This never returns.
        print(f"Starting API and Web server on http://{args.host}:{args.port} ...")
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)

    elif args.command == "clear":
        CACHE.clear()
        print("Cache cleared.")
    elif args.command == "check":
        if args.filename:
            check(args.filename)
        else:
            check()
    elif args.command == "check2":
        check2()
    elif args.command == "check3":
        check3()
    elif args.command == "rebin":
        configure_data_source(
            cache=args.cache,
            refresh=args.refresh,
            auto_hst_file=args.auto_hst_file,
            hst_files=args.hst_files,
            events_file=args.events_file,
        )
        cli_rebin(
            args.filename,
            args.path, 
            preview=args.preview,
            interval=args.interval,
            nbins=args.nbins,
            split=args.split,
        )
    elif args.command == "save-events":
        configure_data_source(
            cache=args.cache,
            refresh=args.refresh,
            auto_hst_file=args.auto_hst_file,
        )
        cli_save_events(
            args.filename,
            args.path,
            hst_files=args.hst_files,
            point=args.point,
            entry=args.entry,
            output=args.output,
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
