import base64
import hashlib
from pathlib import Path
import logging

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response, StreamingResponse

#from dateutil.parser import isoparser
import numpy as np
import diskcache
import scipy.integrate

from . import models
from . import data_cache
from . import rebin_vsans_old
from . import nexus_util
from . import event_capture


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
    data = {k: v[..., start:end] for k, v in counts.items()}
    reply = models.FrameReply(
        data=data,
    )
    return reply

# ===================================
@app.post("/timebin/nexus")
def get_timebin_nexus(request: models.SummaryTimeRequest):
    return get_nexus(request.measurement, request.bins)

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
    reply = models.NexusReply(
        base64_data=base64.b64encode(data),
    )
    return reply


def bin_events(measurement, bins, summary=False):
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
                    print(f"fetching raw events for {entry.file.filename}")
                    event_capture.setup()  # in case it hasn't already been setup for sim
                    raw_events = event_capture.fetch_events_to_memory(entry, measurement.point)
                    print("caching raw events to", raw_events_key)
                    CACHE[raw_events_key] = raw_events
                if events_key not in CACHE:
                    print("correcting")
                    raw_events = CACHE[raw_events_key]
                    #print(raw_events.__dict__)
                    #raw_events = event_capture.fetch_events_to_memory(entry, measurement.point)
                    events = event_capture.event_cleanup(entry, raw_events)
                    #print(events)
                    CACHE[events_key] = events
                events = CACHE[events_key]
                result = _bin_by_time(events, bins.edges)
                #binned = _bin_by_time(entry, events, bins)
            # TODO: should be recording detectors and various devices in binned
            edges = bins.edges
        finally:
            entry.file.close()
        CACHE[binned_key] = result

    if not summary:
        return CACHE[binned_key]

    if summed_key not in CACHE:
        print("accumulating events")
        binned = CACHE[binned_key]
        summed = {}
        for detector, data in binned['detectors'].items():
            total = np.sum(np.sum(data, axis=0), axis=0)
            #print("in summary", detector, data, total)
            summed[detector] = total
        result = dict(detectors=summed, count_time=binned['count_time'])
        CACHE[summed_key] = result
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
    result = dict(detectors=binned, count_time=np.diff(bins.edges))
    return result

def _bin_by_time(events, edges):
    # TODO: does not support masking
    # TODO: check the last edge is the correct length when it is truncated
    # TODO: duration is incorrect with masking and/or incomplete bins

    nbins = len(edges) - 1
    edges = np.asarray(edges*1e9, 'int64')
    result = {}
    result['mode'] = 'time'

    detectors = events['detectors']
    binned_detectors = {}
    for name, detector in detectors.items():
        dims, ts, x, y = detector['dims'], detector['ts'], detector['x'], detector['y']
        #print(f"binning {name} {dims} events={len(ts)} bins={len(edges)-1}")
        ##print(edges[:5], edges[-5:])
        #print(edges)
        ny, nx = dims
        index = np.searchsorted(edges, ts)
        data = np.zeros((ny, nx, nbins+2), 'int32')
        np.add.at(data, (y, x, index), 1)
        binned_detectors[name] = data[:, :, 1:-1]
        print(f"{name} {dims} bins={len(edges)-1} events={len(ts):<8d} keeping={binned_detectors[name].sum():<8d}")
    result['detectors'] = binned_detectors
    result['count_time'] = np.diff(edges)*1e-9

    monitors = events.get('monitors', None)
    if monitors:
        time_bins = np.searchsorted(edges, ts)
        data = np.zeros(nbins+2, 'int32')
        np.add.at(data, time_bins, 1)
        result['monitors'] = data

    # TODO: average per bin includes excluded values
    # Compute average of device value within bins by looking at the difference
    # in the cumulative integral at the edges and dividing by the duration of
    # the bin.
    devices = events.get('devices', {})
    binned_devices = {}
    for name, device in devices.items():
        ts, value = device['ts'], device['value']
        # Make sure the arrays are sorted (do it in event cleanup if necessary)
        assert (ts[1:] > ts[:-1]).all()
        # Insert values at edges of bins into the value array
        index = np.searchsorted(ts, edges)
        v_edge = np.interp(edges, ts, value) # Note: could reuse edge indices
        ts = np.insert(ts, index, edges)
        value = np.insert(value, index, v_edge)
        # Find cumulative values at edge positions. Use trapezoid rule for
        # integration because we are using linear interpolation to find the
        # edge values.
        cum_value = scipy.integrate.cumulative_trapezoid(value, ts)
        cum_index = index + np.arange(len(edges))
        avg = np.diff(cum_value[cum_index])/np.diff(edges)
        binned_devices[name] = avg
        # TODO: std, min, max
        # Other statistics are tricky. For variance you need to compute the
        # integral of (f(x)-avg)^2 over each interval. The trapezoidal integration
        # functions will not work for this, though simpsons quadrature (which
        # uses a quadratic model underneath) might. The end points are tricky
        # since the function is dual-valued at these points (value - left avg and
        # value - right avg). We might be able to do this with vector operations.
        # but easier to drop into numba and do it with a simple for loop. We
        # might even be able to do a parallel for over each bin, with a nested
        # for over the samples within the bin. We need to do this anyway for
        # max/min/mean.
    result['devices'] = binned_devices

    return result

def _bin_strobed(events, edges):
    # TODO: does not support masking
    # TODO: use stobed with one trigger for time binning?
    nbins = len(edges) - 1
    edges = np.asarray(edges*1e9, 'int64')
    result = {}
    result['mode'] = 'strobe'

    # Shift the T0 to an arbitrary phase point
    offset = edges[0]
    edges -= offset # edges is new, so we can update in place with -=

    triggers = events.get('triggers', None)
    if not triggers:
        raise ValueError("Missing trigger information in datastream")

    # TODO: can we update data from the cache inplace?
    triggers = triggers + offset # Don't use += because triggers might be reused

    if len(triggers) > 1:
        delta = np.diff(triggers)
        trigger_stats = dict(
            n=len(triggers),
            min=delta.min()/1e9, 
            max=delta.max()/1e9, 
            mean=delta.mean()/1e9,
            dev=delta.std(ddof=1)/1e9,
        )
        result['trigger'] = trigger_stats

    result['count_time'] = len(triggers)*np.diff(edges)*1e-9

    detectors = events.get('detectors', {})
    binned = {}
    for name, detector in detectors.items():
        dims, ts, x, y = detector['dims'], detector['ts'], detector['x'], detector['y']
        #print(f"binning {name} {dims} events={len(ts)} bins={len(edges)-1}")
        ##print(edges[:5], edges[-5:])
        #print(edges)
        ny, nx = dims
        index = np.searchsorted(edges, ts-triggers)
        data = np.zeros((ny, nx, nbins+2), 'int32')
        np.add.at(data, (y, x, index), 1)
        binned[name] = data[:, :, 1:-1]
        print(f"{name} {dims} bins={len(edges)-1} events={len(ts):<8d} keeping={binned[name].sum():<8d}")

    monitor_ts = events.get('monitors', None)
    if monitors:
        index = np.searchsorted(edges, monitor_ts-triggers)
        data = np.zeros(nbins+2, 'int32')
        np.add.at(data, index, 1)
        result['monitors'] = data

    devices = events.get(devices, {})
    if not devices:
        return result

    # TODO: do we need device average values for strobed?
    # Compute average of device value within bins by looking at the difference
    # in the cumulative integral at the edges and dividing by the duration of
    # the bin.
    # Basically repeat the bins once every trigger, find the area between bins
    # reshape to [ntriggers x nbins] then sum over triggers to get the total
    # area. Normalize by ntriggers times edges. A bit of weirdness because
    # this also forms the area between the end of one trigger and the beginning
    # of the next.
    # TODO: check what happens when trigger interval is shorter then bins width
    strobed_edges = (triggers[:, None] + edges[None, :]).flatten()
    binned_devices = {}
    for name, device in devices.items():
        ts, value = device['ts'], device['value']
        # Make sure the arrays are sorted (do it in event cleanup if necessary)
        assert (ts[1:] > ts[:-1]).all()
        # Insert values at edges of bins into the value array
        index = np.searchsorted(ts, strobed_edges)
        v_edge = np.interp(strobed_edges, ts, value) # Note: could reuse edge indices
        ts = np.insert(ts, index, strobed_edges)
        value = np.insert(value, index, v_edge)
        # Find cumulative values at edge positions. Use trapezoid rule for
        # integration because we are using linear interpolation to find the
        # edge values.
        cum_value = scipy.integrate.cumulative_trapezoid(value, ts)
        cum_index = index + np.arange(len(strobed_edges))
        # Need one extra value because we have an extra column for the values
        # between the one cycle and the start of the next.
        total = np.concat((np.diff(cum_value[cum_index]), 0.))
        summed = total.reshape((len(triggers),len(edges))).sum(axis=0)
        avg = summed[:-1] / np.diff(edges) / len(triggers)
        binned_devices[name] = avg
        # TODO: std, min, max
        # Other statistics are tricky. For variance you need to compute the
        # integral of (f(x)-avg)^2 over each interval. The trapezoidal integration
        # functions will not work for this, though simpsons quadrature (which
        # uses a quadratic model underneath) might. The end points are tricky
        # since the function is dual-valued at these points (value - left avg and
        # value - right avg). We might be able to do this with vector operations.
        # but easier to drop into numba and do it with a simple for loop. We
        # might even be able to do a parallel for over each bin, with a nested
        # for over the samples within the bin. We need to do this anyway for
        # max/min/mean.
    result['devices'] = binned_devices
    return result

def _bin_by_device(name, events, edges, hysterisis=True):
    device = events['devices'][name]
    device_ts, device_value = device['ts'], device['value']

    # Find value bin for each value in the log, then use this to find the
    # change points where the consecutive values are in different bins.
    # Interpolate between these change points to find all bin edges, tagged
    # with the bin number. Sum the intervals according to bin number. This
    # is the time per bin.
    # TODO: verify that poll values extend beyond measurement duration
    # TODO: use numba for the loop (or torch equivalent?)
    # TODO: what happens when value range exceeds bin range?
    # TODO: breaks if there are no change points in value array
    # TODO: maybe smooth the device values before finding transitions
    index = np.searchsorted(edges, device_value)
    change = np.argwhere(np.diff(index) != 0)[:, 0]
    # Start with timestamp and bin index of the first polled value.
    # Guess the initial direction from the direction of the first change point.
    # Note: could use the difference value 0 and value 1 but it might be flat.
    # Note: might get a lot of flips if polling is noisy near a transition value
    pairs = [(device_ts[0], index[0], index[change[0]] > index[0])]
    for k in change:
        current_bin, next_bin = index[k], index[k+1]
        up = current_bin < next_bin
        tl, tr = device_ts[k:k+1]
        vl, vr = device_value[k:k+1]
        slope = (tr-tl)/(vl-vr)
        delta = 1 if up else -1
        for edge_index in range(current_bin+delta, next_bin, delta):
            edge_value = edges[edge_index]
            edge_ts = ((edge_value) - vl)/(vr - vl) * (tr-tl) + tl
            pairs.append((edge_ts, edge_index, up))
    pairs.append((device_ts[-1], index[-1], False))  # we don't use up/down for final
    # Turn transition coordinates into vectors
    bin_ts, bin_index, bin_up = zip(*pairs)
    # Limit to start/end of the measurement
    start_index, end_index = np.searchsorted(bin_ts, [0, duration])
    # Find intervals between each change
    intervals = bin_ts[start_index:end_index+1]
    intervals[0], intervals[-1] = 0, duration
    intervals = np.diff(intervals)
    # Accumulate intervals, using two arrays if directional
    active = slice(start_index, end_index)
    if directional:
        count_time = np.zeros((nbins,2), dtype='float32')
        np.add.at(count_time, (bin_index[active],bin_up[active]), intervals)
    else:
        count_time = np.zeros((nbins,), dtype='float32')
        np.add.at(count_time, (bin_index[active],), intervals)

    detectors = events.get('detectors', {})
    binned = {}
    for name, detector in detectors.items():
        dims, ts, x, y = detector['dims'], detector['ts'], detector['x'], detector['y']
        #print(f"binning {name} {dims} events={len(ts)} bins={len(edges)-1}")
        ##print(edges[:5], edges[-5:])
        #print(edges)
        ny, nx = dims
        value = np.interp(ts, device_ts, device_value)
        index = np.searchsorted(edges, value)
        if directional:
            data = np.zeros((ny, nx, nbins+2, 2), 'int32')
            up = bin_up[np.searchsorted(bin_ts, ts)]
            np.add.at(data, (y, x, index, up), 1)
        else:
            data = np.zeros((ny, nx, nbins+2), 'int32')
            np.add.at(data, (y, x, index), 1)
        binned[name] = data[:, :, 1:-1]

    monitor_ts = events.get('monitors', None)
    if monitors:
        value = np.interp(ts, device_ts, device_value)
        index = np.searchsorted(edges, value)
        if directional:
            up = bin_up[np.searchsorted(bin_ts, ts)]
            data = np.zeros((nbins+2, 2), 'int32')
            np.add.at(data, (index, up), 1)
        else:
            data = np.zeros((nbins+2,), 'int32')
            np.add.at(data, (index, ), 1)
        result['monitors'] = data

    return result

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
