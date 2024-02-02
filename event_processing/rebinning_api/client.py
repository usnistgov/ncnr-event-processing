from dataclasses import asdict

import requests
import numpy as np

from . import models

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
    #print("post", endpoint, request)
    data = request.model_dump_json()
    r = requests.post(url, data=data, headers=headers)
    # result = unpackb(r.content)['result']
    if r.status_code != 200:
        raise r.raise_for_status()
    result = r.json()
    return result

def get_metadata(filename, path=None, point=0):
    request = models.Measurement(filename=filename, path=path, point=point)
    reply = post("metadata", request=request)
    print("metadata post reply", reply)
    return models.MetadataReply.model_validate(reply)

def get_triggers(metadata):
    reply = post("triggers", request=metadata.request)
    return reply

def get_summary(metadata, bins):
    """
    bins can be by time, by time with strobe, or by device value.

    Use one of time_linbins, strobe_linbins, or sweep_linbins to specify.
    """
    if bins.mode == "time":
        assert isinstance(bins, models.TimeBins)
        request = models.SummaryTimeRequest(measurement=metadata.measurement, bins=bins)
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
    return reply

def get_nexus(metadata, bins):
    reply = post("timebin/nexus", metadata.measurement) # point number ignored
    return reply


def time_linbins(metadata, start=None, end=None, interval=0.1, mask=None, point=0):
    # TODO: not sure if it is a good idea to make time bins contingent on mask
    # Reasons against: if you are applying the same bins across a number of
    # measurements but one of them has a bad region of data, you don't want the
    # bin edges to suddenly jump for one of the datasets. Much better to just
    # return NaN for that datapoint so your data is square. Leave the masking
    # default off for now. Anyway, the UI will have the limits and can choose
    # to prompt the user in odd situations.
    if False and mask is not None:
        mask_start = mask.edges[0, 0]
        mask_end = mask.edges[-1, 1]
        # Default start/end to mask limits if we have masking.
        if start is None:
            start = mask_start
        if end is None:
            end = mask_end
    edges = _lin_edges(metadata.duration, start, end, interval)
    bins = models.TimeBins(edges=edges, mask=mask)
    return bins

def strobe_linbins(metadata, start=None, end=None, interval=0.1, mask=None, point=0,
        hysterisis=False, trigger_override=None):
    edges = _lin_edges(metadata.trigger_interval, start, end, interval)
    bins = models.StrobeBins(edges=edges, mask=mask, hysterisis=hysterisis, trigger_override=trigger_override)
    return bins

def sweep_linbins(metadata, start=None, end=None, nbins=100, mask=None, point=0,
        hysterisis=False):
    if metadata.sweep is None:
        raise TypeError("Cannot do sweep binning when not sweeping a motor")
    device = metadata.sweep.device
    if start is None:
        start = metadata.sweep.start
    if stop is None:
        stop = metadata.sweep.stop
    edges = np.linspace(start, stop, nbins+1)
    bins = models.SweepBins(edges=edges, mask=mask, hysterisis=hysterisis, device=device)
    return bins

def env_linbins(metadata, device, start=None, end=None, interval=None, nbins=None, mask=None, point=0,
        hysterisis=False):
    if device not in metadata.logs:
        raise TypeError(f"Cannot bin against {device} when values are not tracked")
    if start is None:
        start = metadata.logs[device].min
    if end is None:
        end = metadata.logs[device].max

    if nbins is not None:
        edges = np.arange(start, end, nbins+1)
    elif interval is not None:
        nbins = (end - start) // interval
        edges = np.arange(start, end+interval, interval)
        if edges[-2] == duration: # last bin is full so don't need a partial bin more
            edges = edges[:-1]
    else:
        raise TypeError("Must specify one of interval or nbins for edges")
    bins = models.SweepBins(edges=edges, mask=mask, hysterisis=hysterisis, device=device)
    return bins

def _lin_edges(duration, start, end, interval=None, nbins=None):
    if start is None:
        start = 0
    elif start < 0:
        start = duration + start
    if end is None:
        end = duration
    elif end < 0:
        end = duration + end

    if nbins is not None:
        edges = np.arange(start, end, nbins+1)
    elif interval is not None:
        nbins = (end - start) // interval
        edges = np.arange(start, end+interval, interval)
        if edges[-2] == duration: # last bin is full so don't need a partial bin more
            edges = edges[:-1]
    else:
        raise TypeError("Must specify one of interval or nbins for edges")

    return edges

def demo():
    filename = "sans68869.nxs.ngv"
    path = "vsans/202009/27861/data"
    metadata = get_metadata(filename, path=path)
    #print("metadata", metadata)
    bins = time_linbins(metadata, interval=10.0)
    summary = get_summary(metadata, bins)
    #print("summary", summary)
    #frames = get_frames(metadata, bins, 0, 2)
    #print("frames", frames)
    hdf = get_nexus(metadata, bins)
    with open('/tmp/client_sample.hdf', 'wb') as fd:
        fd.write(hdf.data)

if __name__ == '__main__':
    demo()
