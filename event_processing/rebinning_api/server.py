import hashlib

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response, StreamingResponse

#from dateutil.parser import isoparser
import numpy as np
import h5py
import diskcache

from . import models
from . import data_cache
from . import rebin_vsans_old

CACHE = None
CACHE_PATH = "/tmp/event-processing"
CACHE_SIZE = int(100e9) 
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
# app.add_middleware(MessagePackMiddleware)

def start_cache():
    cache = diskcache.Cache(
        CACHE_PATH, 
        size_limit=CACHE_SIZE,
        eviction_policy='least-recently-used',
        )
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

@app.post("/metadata")
def get_metadata(request: models.Measurement):
    #print("metadata", type(request), request)
    point = request.point
    path, filename = request.path, request.filename
    nexus = data_cache.load_nexus(filename, datapath=path)

    # TODO: what if there are multiple entries?
    entry = list(nexus.values())[0]
    timestamp = entry['start_time'][()][0].decode('ascii')
    duration = float(entry['control/count_time'][()][point])
    # TODO: find active detectors
    detectors = [
        f"{z} {xy}" for z in "front middle".split()
        for xy in "bottom left right top".split()] # + ["back"]
    #detectors = [f"detector_{k}" for k in 'FB FL FR FT MB ML MR MT'.split()] # + ["detector_B"]
    # TODO: determine mode for sweep device and triggers
    event_mode = 'time' # not written yet... this is the default
    # TODO: lookup sweep controls from nexus file
    reply = models.MetadataReply(
        measurement=request,
        numpoints=1,
        # timestamp=isoparser().isoparse(timestamp),
        timestamp=timestamp,
        duration=duration,
        trigger_interval=0.0,
        detectors=detectors,
        logs={}, # No temperature logs in initial dataset
        event_mode=event_mode,
        sweep=None,
    )
    return reply

@app.post("/summary_time")
def get_summary_time(request: models.SummaryTimeRequest):
    summed = bin_by_time(request.measurement, request.bins, summary=True)
    # TODO: check the last edge is the correct length when it is truncated
    # TODO: duration is incorrect with masking and/or incomplete bins
    edges = request.bins.edges
    duration = (edges[1:] - edges[:-1])
    devices = {}
    monitor = None
    reply = models.SummaryReply(
        measurement=request.measurement,
        bins=request.bins,
        duration=duration,
        counts=summed,
        monitor=monitor,
        devices=devices,
    )
    return reply

@app.post("/timebin/frame/{index}")
def get_timebin_frame(index: int, request: models.SummaryTimeRequest):
    print("loading")
    counts = bin_by_time(request.measurement, request.bins, summary=False)
    print("loaded")
    for k, v in counts.items():
        print(k, v.shape)
    frames = {k: v[..., index:index+1] for k, v in counts.items()}
    for k, v in frames.items():
        print(k, v.shape)
    reply = models.FrameReply(
        frames=frames,
    )
    return reply

@app.post("/timebin/frame/{start}-{end}")
def get_timebin_frame_range(start: int, end: int, request: models.SummaryTimeRequest):
    counts = bin_by_time(request.measurement, request.bins, summary=False)
    frames = {k: v[..., start:end] for k, v in counts.items()}
    reply = models.FrameReply(
        frames=frames,
    )
    return reply

def bin_by_time(measurement, bins, summary=False):
    point = measurement.point
    path, filename = measurement.path, measurement.filename
    edges = bins.edges
    mask = bins.mask

    key = (request_key(measurement), request_key(bins))
    binned_key = (*key, "binned")
    summed_key = (*key, "summed")
    if binned_key not in CACHE:
        print("processing events")
        nexus = data_cache.load_nexus(filename, datapath=path)
        entry = list(nexus.values())[0]
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
                binned[f"{z} {xy}"] = data
        CACHE[binned_key] = binned
    else:
        print("cache hit", binned_key)

    if not summary:
        return CACHE[binned_key]

    if summed_key not in CACHE:
        print("accumulating events")
        binned = CACHE[binned_key]
        summed = {}
        for detector, data in binned.items():
            total = np.sum(np.sum(data, axis=0), axis=0)
            summed[detector] = total
        CACHE[summed_key] = summed
    else:
        print("cache hit", summed_key)
    return CACHE[summed_key]

def request_key(request):
    data = request.model_dump_json()
    digest = hashlib.sha1(data.encode('utf-8')).hexdigest()
    return digest

def demo():
    from . import client
    filename = "sans68869.nxs.ngv"
    measurement = models.Measurement(filename=filename)
    #data_cache.load_nexus(request.filename, datapath=request.path)
    #print(get_metadata(measurement).body)
    metadata = get_metadata(measurement)
    print("metadata", metadata)
    bins = client.time_linbins(metadata)
    request = models.SummaryTimeRequest(measurement=metadata.measurement, bins=bins)
    summary = get_summary_time(request)
    print("summary", summary)
    r_one = get_timebin_frame(250, request)
    print("frame", r_one)
    r_many = get_timebin_frame_range(250, 252, request)
    detector = "front left"
    print(r_one.frames[detector].shape, r_many.frames[detector].shape)
    assert (r_one.frames[detector][...,0] == r_many.frames[detector][..., 0]).all()


if __name__ == "__main__":
    demo()
