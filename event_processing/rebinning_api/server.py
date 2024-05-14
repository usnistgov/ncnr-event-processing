import asyncio
import base64
from collections import deque
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

from . import models
from . import rebin_vsans_old
from . import nexus_util
from . import event_capture
from . import binning


CACHE = None
CACHE_PATH = "/tmp/event-processing"
CACHE_VERSION = "0.2"
CACHE_SIZE = int(100e9) # 100 GB
CACHE_ITEMS = 100 # max number of items, if not using items size in cache
DOWNLOAD_HISTORY_SIZE = 10000 # number of downloads to keep track of

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
    data = get_nexus(request.measurement, request.bins)
    reply = models.NexusReply(
        base64_data=base64.b64encode(data),
    )
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
    """ post request coming from HTML form, that can trigger a download """
    request_dict = json.loads(request_str)
    request = models.SummaryTimeRequest(**request_dict)
    coro = asyncio.to_thread(get_nexus, request.measurement, request.bins)
    try:
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
                COMPLETED_DOWNLOADS.append(download_id)

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
    except Exception as e:
        if (download_id != ''):
            PROCESSING_ERRORS[download_id] = str(e)
        raise e


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
    binned = bin_events(measurement, bins, summary=False)
    entry = nexus_util.open_nexus_entry(measurement)
    try:
        data = nexus_util.nexus_dup(entry, binned, bins)
    finally:
        entry.file.close()
    return data


def bin_events(measurement, bins, summary=False):
    from timeit import default_timer as tic; T0 = tic()
    if bins.mode != "time":
        raise NotImplementedError("only time-mode binning implemented for now")

    key = (request_key(measurement), request_key(bins))
    #print("Key:", key)
    # Increment version number if the data changes
    raw_events_key = (key[0], "raw", "v1")  # events keyed by entry, not bin spec
    events_key = (key[0], "events", "v1")
    binned_key = (*key, "binned", "v1")   # binning keyed by both entry and bin spec
    summed_key = (*key, "summed", "v1")
    if binned_key not in CACHE:
        #print("processing events")
        entry = nexus_util.open_nexus_entry(measurement)
        try:
            # CRUFT: we are allowing some old vsans histograms to run for demo purposes.
            if measurement.filename.startswith('sans') and measurement.filename < "sans72000":
                result = _bin_by_time_old_vsans(entry, bins)
            else:
                # TODO: drop raw events cache once we have event_cleanup working for everything
                if raw_events_key not in CACHE:
                    print(f"{tic()-T0:.1f}: fetching raw events for {entry.file.filename}")
                    event_capture.setup()  # in case it hasn't already been setup for sim
                    raw_events = event_capture.fetch_events_to_memory(entry, measurement.point)
                    print(f"{tic()-T0:.1f}: caching raw events to", raw_events_key)
                    CACHE[raw_events_key] = raw_events
                if events_key not in CACHE:
                    print(f"{tic()-T0:.1f}: correcting events")
                    raw_events = CACHE[raw_events_key]
                    #print(raw_events.__dict__)
                    #raw_events = event_capture.fetch_events_to_memory(entry, measurement.point)
                    events = event_capture.event_cleanup(entry, raw_events)
                    #print(events)
                    CACHE[events_key] = events
                events = CACHE[events_key]
                print(f"{tic()-T0:.1f}: binning")
                result = binning.bin(entry, measurement.point, bins, events)
                print(f"{tic()-T0:.1f}: binned")
                #binned = _bin_by_time(entry, events, bins)
        finally:
            entry.file.close()
        CACHE[binned_key] = result
        print(f"{tic()-T0:.1f}: cached")

    if not summary:
        result = CACHE[binned_key]
        print(f"{tic()-T0:.1f}: retrieved bins")
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
        eventpath = rebin_vsans_old.fetch_eventfile("vsans", eventfile)
        if not Path(eventpath).exists():
            print("missing", eventpath)
            continue
        print(f"{tic()-T0:.1f}: loading", detector)
        events = rebin_vsans_old.VSANSEvents(eventpath)
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

def check(verbose=False):
    from . import client

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
