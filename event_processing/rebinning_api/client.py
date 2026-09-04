import base64
import logging

import requests
import numpy as np

from . import models
from . import binning

HOST = "http://localhost:8000"

_ = '''
from msgpack import unpackb
from models import VSANS, RebinnedData, RebinUniformCount, RebinUniformWidth, NumpyArray
def get_rebinned():
    data = VSANS().model_dump_json()
    r = requests.post(f"{HOST}/rebin", data=data, headers={"Content-Type": "application/json", "Accept": "application/msgpack"})
    result = unpackb(r.content)['result']
    for dname, v in result['detectors'].items():
        result['detectors'][dname] = NumpyArray(**v).to_ndarray()
    for dname, v in result['devices'].items():
        result['devices'][dname] = NumpyArray(**v).to_ndarray()
    print(result)
'''

def post(endpoint, request):
    url = f"{HOST}/{endpoint}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    # print("post", HOST, endpoint, request)
    data = request.model_dump_json()
    try:
        r = requests.post(url, data=data, headers=headers)
    except Exception as exc:
        logging.error(f"Exception raised during POST {url} with {data}\n   {exc}")
        raise
    # result = unpackb(r.content)['result']
    result = r.json()
    if r.status_code != 200:
        msg = f"POST {r.url} {r.status_code} {r.reason} : {result.get('detail', 'unknown reason')}"
        logging.error(msg)
        raise r.raise_for_status()
    return result

def get_metadata(filename, path=None, point=0):
    request = models.Measurement(filename=filename, path=path, point=point)
    reply = post("metadata", request=request)
    # print("metadata post reply", reply)
    return models.MetadataReply.model_validate(reply)

def get_summary(metadata, bins):
    """
    bins can be by time, by time with strobe, or by device value.

    Use one of time_linbins, strobe_linbins, or sweep_linbins to specify.
    """
    if bins.mode == "time":
        assert isinstance(bins, models.TimeBins)
        request = models.SummaryTimeRequest(measurement=metadata.measurement, bins=bins)
    else:
        raise NotImplementedError('slicing other than time not yet implemented')
    reply = post(f"summary_{bins.mode}", request)
    return models.SummaryReply(**reply)

def get_frames(metadata, bins, start=0, stop=1):
    """
    bins can be by time, by time with strobe, or by device value.

    Use one of time_linbins, strobe_linbins, or sweep_linbins to specify.
    """
    if bins.mode == "time":
        assert isinstance(bins, models.TimeBins)
        request = models.SummaryTimeRequest(measurement=metadata.measurement, bins=bins)
    #reply = post(f"timebin/frame/{start}-{stop}", request)
    reply = post(f"timebin/frame/{start}", request)
    return models.FrameReply(**reply)

def get_timebin_nexus(metadata, bins):
    request = models.SummaryTimeRequest(measurement=metadata.measurement, bins=bins)
    reply = post("timebin/nexus", request) # point number ignored
    return models.NexusReply(**reply)



def demo():
    filename = "sans68869.nxs.ngv"
    path = "vsans/202009/27861/data"
    metadata = get_metadata(filename, path=path)
    #print("metadata", metadata)
    bins = binning.time_linbins(metadata, interval=10.0)
    #summary = get_summary(metadata, bins)
    #print("summary", summary)
    #frames = get_frames(metadata, bins, 0, 2)
    #print("frames", frames)
    reply = get_timebin_nexus(metadata, bins, split=False)
    data = base64.b64decode(reply.base64_data)
    print(f"writing /tmp/{reply.filename}")
    with open(f'/tmp/{reply.filename}', 'wb') as fd:
        fd.write(data)

def test_frame():
    filename = "sans68869.nxs.ngv"
    path = "vsans/202009/27861/data"
    metadata = get_metadata(filename, path=path)
    #print("metadata", metadata)
    bins = binning.time_linbins(metadata, interval=3.6)
    summary = get_summary(metadata, bins)
    frames = get_frames(metadata, bins, 100, 101)
    return frames

if __name__ == '__main__':
    demo()
