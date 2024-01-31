from dataclasses import asdict

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response, StreamingResponse

import models
import serial

import numpy as np
import h5py

# TODO: make into a package with relative imports
import models, data_cache, iso8601

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

# Not yet implemented....
@app.post("/metadata")
def metadata(request: models.MetadataRequest):
    from dateutil.parser import isoparser

    point = request.point
    path, filename = request.path, request.filename
    nexus = data_cache.load_nexus(filename, datapath=path)

    # TODO: what if there are multiple entries?
    entry = list(nexus.values())[0]
    timestamp = entry['start_time'][()][0].decode('ascii')
    duration = entry['control/count_time'][()][point]
    # TODO: find active detectors
    detectors = [f"detector_{k}" for k in 'B FB FL FR FT MB ML MR MT'.split()]
    # TODO: determine mode for sweep device and triggers
    event_mode = 'relaxation' # not written yet... this is the default
    # TODO: lookup sweep controls from nexus file
    reply = models.MetadataReply(
        request=request,
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
    return Response(content=serial.dumps(asdict(reply)), media_type="application/json")


def lookup_nexus(request):

    # TODO: mtime on remote file for invalidating cache?
    # TODO: nexus filename collisions?
    # Neither of these are an issue for vsans so ignore for now.
    print("request", request, request.path, request.filename, request.point)
    if request.path is None:
        datapath = data_cache.nexus_lookup(request.filename)
    else:
        datapath = request.path
    url = data_cache.nexus_url(datapath, request.filename)
    fullpath = data_cache.cache_url(url, data_cache.NEXUS_FOLDER)
    return h5py.File(fullpath)

def demo():
    filename = "sans68869.nxs.ngv"
    request = models.MetadataRequest(filename=filename,refresh=True)
    #data_cache.load_nexus(request.filename, datapath=request.path)
    print(metadata(request))

if __name__ == "__main__":
    demo()
