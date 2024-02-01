from dataclasses import asdict

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response, StreamingResponse

#from dateutil.parser import isoparser
import numpy as np
import h5py

# TODO: make into a package with relative imports
import models
import serial
import data_cache
import rebin_vsans_old

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
# app.add_middleware(MessagePackMiddleware)

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
'''

def bundle(reply):
    data = serial.dumps(reply)
    #print("encoded reply", data)
    return Response(content=data, media_type="application/json")

def unbundle(reply):
    data = reply.body
    return serial.loads(data)

@app.post("/metadata")
def get_metadata(request): #models.Measurement):
    #print("metadata", type(request), request)
    point = request.point
    path, filename = request.path, request.filename
    nexus = data_cache.load_nexus(filename, datapath=path)

    # TODO: what if there are multiple entries?
    entry = list(nexus.values())[0]
    timestamp = entry['start_time'][()][0].decode('ascii')
    duration = entry['control/count_time'][()][point]
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
    return bundle(reply)

@app.post("/summary_time")
def get_summary_time(request): # models.SummaryTimeRequest):
    #print("summary_time", request)

    point = request.measurement.point
    path, filename = request.measurement.path, request.measurement.filename
    edges = request.bins.edges
    mask = request.bins.mask

    nexus = data_cache.load_nexus(filename, datapath=path)
    entry = list(nexus.values())[0]

    # TODO: memoize counts based on request
    counts = {}
    for z, detector in (("front", "FL"), ("middle", "ML")):
        #print("fetching")
        eventfile = entry[f'instrument/detector_{detector}/event_file_name'][0].decode()
        eventpath = rebin_vsans_old.fetch_eventfile("vsans", eventfile)
        #print("loading")
        events = rebin_vsans_old.VSANSEvents(eventpath)
        # TODO: correct for time of flight
        # TODO: elide events in mask
        #print("binning")
        partial_counts, _ = events.rebin(edges)
        for xy, data in partial_counts.items():
            counts[f"{z} {xy}"] = data

    # summarize
    #print("summing")
    for detector, data in counts.items():
        integrated = np.sum(np.sum(data, axis=0), axis=0)
        counts[detector] = integrated

    # TODO: check the last edge is the correct length when it is truncated
    duration = (edges[1:] - edges[:-1])
    devices = {}
    monitor = None
    reply = models.SummaryReply(
        measurement=request.measurement,
        bins=request.bins,
        duration=duration,
        counts=counts,
        monitor=monitor,
        devices=devices,
    )
    #print("bundling")
    return bundle(reply)

def demo():
    import client
    filename = "sans68869.nxs.ngv"
    measurement = models.Measurement(filename=filename)
    #data_cache.load_nexus(request.filename, datapath=request.path)
    #print(get_metadata(measurement).body)
    metadata = unbundle(get_metadata(measurement))
    print("metadata", metadata)
    bins = client.time_linbins(metadata)
    request = models.SummaryTimeRequest(measurement=metadata.measurement, bins=bins)
    summary = unbundle(get_summary_time(request))
    print("summary", summary)


if __name__ == "__main__":
    demo()
