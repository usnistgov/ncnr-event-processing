import base64
import hashlib
import io

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
# app.add_middleware(GZipMiddleware, minimum_size=1000)
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
    point = request.point
    entry = open_nexus_entry(request)
    #print("entry", entry, entry.parent)
    # Go up to the root to get the other NXentry items in the file.
    entries = nexus_entries(entry.parent)
    timestamp = entry['start_time'][0].decode('ascii')
    duration = float(entry['control/count_time'][point])
    numpoints = entry['DAS_logs/trajectory/liveScanLength'][0]
    replacement = nexus_detector_replacement(entry)
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
    #print("loading")
    counts = bin_by_time(request.measurement, request.bins, summary=False)
    #print("loaded")
    #for k, v in counts.items(): print(k, v.shape)
    data = {k: v[..., index:index+1] for k, v in counts.items()}
    #for k, v in frames.items(): print(k, v.shape)
    reply = models.FrameReply(
        data=data,
    )
    return reply

@app.post("/timebin/frame/{start}-{end}")
def get_timebin_frame_range(start: int, end: int, request: models.SummaryTimeRequest):
    counts = bin_by_time(request.measurement, request.bins, summary=False)
    data = {k: v[..., start:end] for k, v in counts.items()}
    reply = models.FrameReply(
        data=data,
    )
    return reply

@app.post("/timebin/nexus")
def get_timebin_nexus(request: models.SummaryTimeRequest):
    # TODO: support files with different binning at each point
    # Could split points across files or across entries within a file
    # TODO: maybe provide "explode" option to split each bin to a different file
    # TODO: check that there is only one entry with one point
    entry = open_nexus_entry(request.measurement)
    counts = bin_by_time(request.measurement, request.bins, summary=False)
    #print("counts", counts)
    replacement_target = nexus_detector_replacement(entry)
    replacement = {
        v: counts[k] for k, v in replacement_target.items()
    }

    bio = io.BytesIO()
    with h5py.File(bio, 'w') as target:
        hdf_copy(entry.parent, target, replacement)
        # TODO: Consider storing the binning info in each detector
        # TODO: Add the appropriate NeXus metadata to the fields
        # TODO: Add masking info, etc.
        control = target[entry.name]['control']
        edges = control.create_dataset('bin_edges', data=request.bins.edges)
        edges.attrs['units'] = 'seconds'
        edges.attrs['long_name'] = 'bin edges for time binned data'
        control.create_dataset('bin_mode', data='time')

    data = bio.getvalue()
    reply = models.NexusReply(
        base64_data=base64.b64encode(data),
    )
    return reply

def open_nexus_entry(measurement: models.Measurement):
    point = measurement.point
    path, filename = measurement.path, measurement.filename
    entry_number = measurement.entry

    nexus = data_cache.load_nexus(filename, datapath=path)
    entries = nexus_entries(nexus)
    #print("entries", entries)
    entry_name = entries[entry_number]
    return nexus[entry_name]

def nexus_entries(nexus):
    """List of nexus entries"""
    #print({k: list(v.attrs.items()) for k, v in nexus.items()})
    return list(k for k, v in nexus.items() if v.attrs['NX_class'] == 'NXentry')

def nexus_detector_replacement(entry):
    """
    Determine the DAS_logs location of each detector data element.
    """
    #print("instrument", sorted(entry["instrument"].items()))
    # TODO: check that the linked detector has stored data
    # that may be enough to verify that it is active
    detectors = {
        name: group["data"].attrs["target"]
        for name, group in sorted(entry["instrument"].items())
        if group.attrs.get('NX_class', None) == 'NXdetector'
        and "data" in group
        and "target" in group["data"].attrs
    }
    #print("detectors", detectors)
    return detectors

def hdf_copy(source, target, replacement):
    # type: (h5py.Group, str) -> h5py.File
    """
    Copy an entry and all sub-entries from source to a destination.

    *source* is an open node in an hdf file.

    *target* is an hdf file opened for writing.

    *replacement* is a dictionary of replacement fields.
    """
    links = _hdf_copy_internal(source, target, replacement)
    for link_to, link_from in sorted(links):
        #print("linking", link_from, link_to)
        target[link_to] = target[link_from]

def _hdf_copy_internal(root, h5file, replacement):
    # type: (h5py.Group, h5py.Group, List[Tuple[str, str]]) -> None
    links = []
    # print(">>> group copy", root.name, "\n   ", "\n    ".join(sorted(root.keys())))
    for item_name, item in sorted(root.items()):
        item_path = f"{root.name}/{item_name}" if root.name != "/" else f"/{item_name}"
        #print("joining", root.name, item_name, "as", item_path)
        #item_path = posixpath.join(root.name, item_name)
        if 'target' in item.attrs and item.attrs['target'] != item_path:
            # print("linking", item_path, item.name)
            links.append((item_path, item.attrs['target']))
        elif hasattr(item, 'dtype'):
            data = replacement.get(item_path, item[()])
            # print("copying", item_path, item.name)
            node = h5file.create_dataset(item.name, data=data)
            attrs = dict(item.attrs)
            node.attrs.update(item.attrs)
        else:
            # print("making", item_path, item.name)
            # Hope that it is a group...
            node = h5file.create_group(item.name)
            node.attrs.update(item.attrs)
            links.extend(_hdf_copy_internal(item, h5file, replacement))
    return links

def bin_by_time(measurement, bins, summary=False):
    key = (request_key(measurement), request_key(bins))
    binned_key = (*key, "binned")
    summed_key = (*key, "summed")
    if binned_key not in CACHE:
        #print("processing events")
        entry = open_nexus_entry(measurement)
        #point = measurement.point # ignored in vsans rebin old
        edges = bins.edges
        #mask = bins.mask # ignored in vsans rebin old
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
        CACHE[binned_key] = binned
    #else: print("cache hit", binned_key)

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
    #else: print("cache hit", summed_key)
    return CACHE[summed_key]

def request_key(request):
    data = request.model_dump_json()
    digest = hashlib.sha1(data.encode('utf-8')).hexdigest()
    return digest

# TODO: how do we clear the cache when upgrading the application?

def demo():
    from . import client
    #CACHE.clear()
    filename = "sans68869.nxs.ngv"
    measurement = models.Measurement(filename=filename)
    #data_cache.load_nexus(request.filename, datapath=request.path)
    #print(get_metadata(measurement).body)
    metadata = get_metadata(measurement)
    #print("metadata", metadata)
    bins = client.time_linbins(metadata, interval=5)
    request = models.SummaryTimeRequest(measurement=metadata.measurement, bins=bins)
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

if __name__ == "__main__":
    demo()
