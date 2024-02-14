import asyncio
import base64
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Annotated
import uuid
import logging

from fastapi import FastAPI, Form
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

#from dateutil.parser import isoparser
import numpy as np
import diskcache

from . import models
from . import data_cache
from . import rebin_vsans_old
from . import nexus_util
from . import event_capture


CACHE = None
CACHE_PATH = "/tmp/event-processing"
CACHE_VERSION = "0.2"
CACHE_SIZE = int(100e9) 
app = FastAPI()
# app.add_middleware(GZipMiddleware, minimum_size=1000)
# app.add_middleware(MessagePackMiddleware)

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

def start_cache():
    cache = diskcache.Cache(
        CACHE_PATH, 
        size_limit=CACHE_SIZE,
        eviction_policy='least-recently-used',
        )

    version_file = Path(CACHE_PATH) / "version.txt"
    if version_file.exists():
        disk_version = version_file.read_text()
        if disk_version != CACHE_VERSION:
            logging.info(f"Updating cache from {disk_version} to {CACHE_VERSION}")
            cache.clear()
            version_file.write_text(CACHE_VERSION)
    else:
        cache.clear() # CRUFT: harmless, but clears out Brian's cache the first time
        version_file.write_text(CACHE_VERSION)
    return cache
CACHE = start_cache()

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
    point = request.point
    entry = nexus_util.open_nexus_entry(request)
    #print("entry", entry, entry.parent)
    # Go up to the root to get the other NXentry items in the file.
    entries = nexus_util.nexus_entries(entry.parent)
    timestamp = entry['start_time'][0].decode('ascii')
    duration = float(entry['control/count_time'][point])
    numpoints = entry['DAS_logs/trajectory/liveScanLength'][0]
    replacement = nexus_util.nexus_detector_replacement(entry)
    #print("detector data links", replacement)
    detectors = list(replacement.keys())
    # TODO: determine mode for sweep device and triggers
    event_mode = 'time' # not written yet... this is the default
    trigger_interval = 0.0
    # TODO: lookup sweep controls from nexus file
    logs = {}
    sweep = None

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
    summed, duration = bin_events(measurement, bins, summary=True)
    # TODO: need duration from other event binners
    devices = {}
    monitor = None
    reply = models.SummaryReply(
        measurement=measurement,
        bins=bins,
        duration=duration,
        counts=summed,
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
    counts, duration = bin_events(measurement, bins, summary=False)
    data = {k: v[..., start:end] for k, v in counts.items()}
    reply = models.FrameReply(
        data=data,
    )
    return reply

# ===================================
@app.post("/timebin/nexus")
def get_timebin_nexus(request: models.SummaryTimeRequest):
    data = get_nexus(request.measurement, request.bins)
    reply = models.NexusReply(
        base64_data=base64.b64encode(data),
    )
    return reply


ACTIVE_DOWNLOADS = set()

@app.get('/timebin/nexus_download_status/{download_id}')
def get_download_status(download_id: str):
    return (download_id in ACTIVE_DOWNLOADS)

@app.post('/timebin/nexus_download')
async def download_nexus_form(request_str: Annotated[str, Form()], download_id: Annotated[str, Form()] = ''):
    """ post request coming from HTML form, that can trigger a download """
    request_dict = json.loads(request_str)
    request = models.SummaryTimeRequest(**request_dict)
    if (download_id != ''):
        ACTIVE_DOWNLOADS.add(download_id)

    coro = asyncio.to_thread(get_nexus, request.measurement, request.bins)
    data = await coro
    orig_filename = request.measurement.filename
    orig_path = Path(orig_filename)
    file_suffixes = ''.join(orig_path.suffixes)
    file_stem = re.sub(f"{file_suffixes}$", '', orig_filename)
    new_filename = f"{file_stem}_rebinned{file_suffixes}"
    buffer_size = 2**16 # 64K
    async def result_streamer():
        with io.BytesIO(data) as bio:
            buffer = bio.read(buffer_size)
            while buffer:
                yield buffer
                buffer = bio.read(buffer_size)
        if (download_id != ''):
            ACTIVE_DOWNLOADS.remove(download_id)

    last_updated_pattern = "%a, %d %b %Y %H:%M:%S GMT"
    last_modified = datetime.datetime.strftime(datetime.datetime.now(datetime.timezone.utc), last_updated_pattern)
    content_length = str(len(data))
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


def get_nexus(measurement, bins):
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
    counts, duration = bin_events(measurement, bins, summary=False)
    entry = nexus_util.open_nexus_entry(measurement)
    try:
        data = nexus_util.nexus_dup(entry, counts, duration, bins)
    finally:
        entry.file.close()
    return data


def bin_events(measurement, bins, summary=False):
    if bins.mode != "time":
        raise NotImplementedError("only time-mode binning implemented for now")

    key = (request_key(measurement), request_key(bins))
    #print("Key:", key)
    raw_events_key = (key[0], "raw")  # events keyed by entry, not bin spec
    events_key = (key[0], "events")
    binned_key = (*key, "binned")   # binning keyed by both entry and bin spec
    summed_key = (*key, "summed")
    if 0 or binned_key not in CACHE:
        #print("processing events")
        entry = nexus_util.open_nexus_entry(measurement)
        try:
            # CRUFT: we are allowing some old vsans histograms to run for demo purposes.
            if measurement.filename.startswith('sans') and measurement.filename < "sans72000":
                binned = _bin_by_time_old_vsans(entry, bins)
            else:
                # TODO: drop raw events cache once we have event_cleanup working for everything
                if 0 or raw_events_key not in CACHE:
                    print(f"fetching raw events for {entry.file.filename}")
                    event_capture.setup()  # in case it hasn't already been setup for sim
                    raw_events = event_capture.fetch_events_to_memory(entry, measurement.point)
                    print("caching raw events to", raw_events_key)
                    CACHE[raw_events_key] = raw_events
                if 0 or events_key not in CACHE:
                    print("correcting")
                    raw_events = CACHE[raw_events_key]
                    #print(raw_events.__dict__)
                    #raw_events = event_capture.fetch_events_to_memory(entry, measurement.point)
                    events = event_capture.event_cleanup(entry, raw_events)
                    #print(events)
                    CACHE[events_key] = events
                events = CACHE[events_key]
                binned = _bin_by_time(events, bins.edges)
                #binned = _bin_by_time(entry, events, bins)
            # TODO: should be recording detectors and various devices in binned
            # TODO: check the last edge is the correct length when it is truncated
            # TODO: duration is incorrect with masking and/or incomplete bins
            edges = bins.edges
            duration = (edges[1:] - edges[:-1])
        finally:
            entry.file.close()
        CACHE[binned_key] = binned, duration

    if not summary:
        return CACHE[binned_key]

    if summed_key not in CACHE:
        print("accumulating events")
        binned, duration = CACHE[binned_key]
        summed = {}
        for detector, data in binned.items():
            total = np.sum(np.sum(data, axis=0), axis=0)
            summed[detector] = total
        CACHE[summed_key] = summed, duration
    return CACHE[summed_key]


# CRUFT: code for old-style histograms
def _bin_by_time_old_vsans(entry, bins):
    #point = measurement.point # ignored in vsans rebin old
    edges = bins.edges
    #mask = bins.mask # ignored in vsans rebin old
    # TODO: not binning monitor or devices
    binned = {}
    for z, detector in (("front", "FL"), ("middle", "ML")):
        print("fetching", detector)
        eventfile = entry[f'instrument/detector_{detector}/event_file_name'][0].decode()
        eventpath = rebin_vsans_old.fetch_eventfile("vsans", eventfile)
        print("loading", detector)
        events = rebin_vsans_old.VSANSEvents(eventpath)
        # TODO: correct for time of flight
        # TODO: elide events in mask
        print("binning", detector)
        partial_counts, _ = events.rebin(edges)
        for xy, data in partial_counts.items():
            # form detector_FB, etc. from first letter of names
            name = f"detector_{z[0].upper()}{xy[0].upper()}"
            binned[name] = data
    return binned

def _bin_by_time(events, edges):
    nbins = len(edges) - 1
    edges = np.asarray(edges*1e9, 'int64')
    binned = {}
    for name, detector in events.items():
        dims, ts, x, y = detector['dims'], detector['ts'], detector['x'], detector['y']
        #print(f"binning {name} {dims} events={len(ts)} bins={len(edges)-1}")
        ##print(edges[:5], edges[-5:])
        #print(edges)
        ny, nx = dims
        time_bins = np.searchsorted(edges, ts)
        data = np.zeros((ny, nx, nbins+2), 'int32')
        np.add.at(data, (y, x, time_bins), 1)
        binned[name] = data[:, :, 1:-1]
        print(f"{name} {dims} bins={len(edges)-1} events={len(ts):<8d} keeping={binned[name].sum():<8d}")
    return binned

def request_key(request):
    data = request.model_dump_json()
    digest = hashlib.sha1(data.encode('utf-8')).hexdigest()
    return digest

# TODO: how do we clear the cache when upgrading the application?

def check():
    from . import client

    filename = "sans68869.nxs.ngv"
    measurement = models.Measurement(filename=filename)
    #data_cache.load_nexus(request.filename, datapath=request.path)
    #print(get_metadata(measurement).body)
    metadata = get_metadata(measurement)
    #print("metadata", metadata)
    bins = client.time_linbins(metadata, interval=5)
    request = models.SummaryTimeRequest(measurement=measurement, bins=bins)
    summary = get_summary_time(request)
    #print("summary", summary)
    index = np.searchsorted(bins.edges, 500.)
    r_one = get_timebin_frame(index, request)
    #print("frame", index, {k: v.shape for k, v in r_one.data.items()})
    r_many = get_timebin_frame_range(index, index+2, request)
    detector = "detector_FL"
    #print(r_one.data[detector].shape, r_many.data[detector].shape)
    assert (r_one.data[detector][...,0] == r_many.data[detector][..., 0]).all()
    hdf = get_timebin_nexus(request)
    with open('/tmp/sample.hdf', 'wb') as fd:
        fd.write(base64.b64decode(hdf.base64_data))

def check2():
    path = "vsans/202102/27861/data"
    nexusfile = "sans72109.nxs.ngv"
    event_capture.setup()
    with event_capture.kafka_consumer() as consumer:
        event_capture.fetch_events_for_file(consumer, nexusfile, datapath=path)

def check3():
    from . import client
    event_capture.setup()
    path = "vsans/202102/27861/data"
    nexusfile = "sans72110.nxs.ngv"
    measurement = models.Measurement(filename=nexusfile, path=path, point=0)
    metadata = get_metadata(measurement)
    bins = client.time_linbins(metadata, interval=501)
    request = models.SummaryTimeRequest(measurement=measurement, bins=bins)
    hdf = get_timebin_nexus(request)
    with open('/tmp/end-to-end.hdf', 'wb') as fd:
        fd.write(base64.b64decode(hdf.base64_data))

# TODO: cache a version number, clearing the cache if there is a version mismatch
usage = """
Usage: server clear|check

clear: Empties any caches associated with the data. This should happen
    automatically if you bump server.CACHE_VERSION to a new value, but you
    may still want to clear the version manually when e.g., testing speed.
check: Runs some simple event processing to make sure that the pieces
    work together. This is a development tool acting as a poor substitute
    for a proper test harness.

To run the actual server for responding to web requests use uvicorn:

    uvicorn event_processing.rebinning_api.server:app
 """

def main():
    import sys
    # TODO: admit early that we need an options parser
    if "clear" in sys.argv[1:]:
        CACHE.clear()
    elif "check" in sys.argv[1:]:
        check()
    elif "check2" in sys.argv[1:]:
        check2()
    elif "check3" in sys.argv[1:]:
        check3()
    else:
        print(usage)

if __name__ == "__main__":
    main()
