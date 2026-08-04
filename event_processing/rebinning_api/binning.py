import os
import numpy as np
import scipy.integrate

from . import models

def bin(entry, point, bins:models.Bins, events):
    """
    Note: for binning purposes a monitor is is just a point detector with a
    special name. It is split out of the detector structure after binning.
    This keeps the binning code much cleaner.
    """
    # TODO: masking currently ignored
    if bins.mode == "time":
        result = _bin_by_time(events, bins.edges)
    elif bins.mode == "strobe":
        result = _bin_strobed(events, bins.edges)
    elif bins.mode == "device":
        result = _bin_by_device(events, bins.edges, bins.device, hysteresis=bins.hysteresis)
    else:
        raise KeyError(f"Unrecognized bin mode {bins.mode}")


    # Convert detectors from area to linear or point as necessary
    result['detectors'] = _squeeze_detectors(result['detectors'])
    # Extract monitor into a separate reult
    monitor = result['detectors'].pop('monitor', None)
    if monitor is not None:
        result['monitor'] = monitor

    return result


def _squeeze_detectors(detectors):
    return {name: _squeeze_one(data) for name, data in detectors.items()}

def _squeeze_one(data):
    if data.shape[-2:] == (1,1): # point detector (tube)
        return data[..., 0, 0]
    elif data.shape[-1] == 1: # linear detector (psd)
        return data[..., 0]
    else: # area detector
        return data

def _bin_by_time(events, edges):
    # TODO: does not support masking
    # TODO: check the last edge is the correct length when it is truncated
    # TODO: duration is incorrect with masking and/or incomplete bins

    edges = np.asarray(edges*1e9, 'int64')
    result = {}
    result['mode'] = 'time'

    #print("by time", events)
    detectors = events['detectors']
    #print("detectors", detectors)
    binned_detectors = {}
    for name, detector in detectors.items():
        dims, ts, x, y = detector['dims'], detector['ts'], detector['x'], detector['y']
        binned_detectors[name] = hist(dims=dims, edges=edges, ts=ts, x=x, y=y)
        print(f"{name} {dims} bins={len(edges)-1} events={len(ts):<8d} keeping={binned_detectors[name].sum():<8d}")
    result['detectors'] = binned_detectors
    result['count_time'] = np.diff(edges)*1e-9

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

def _get_torch_gpu():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "cpu"
        return "mps"
    return "cpu"

def _hist_torch_addat(dims, edges, ts, x, y):
    import torch
    from torch.nn.functional import pad

    device = _get_torch_gpu()

    # include two extra bins for the timestamps that fall outside the defined bins
    # (those with indices 0 and n_bins + 1); for an array of size n+1 searchsorted
    # returns insertion indices from 0 (below the left edge) to n+1 (past the right edge)
    # To get the indexing to work right for values outside the range, need an
    # extra zero for the left edge.
    n_bins = len(edges) - 1
    nx, ny = dims
    edges = torch.from_numpy(np.asarray(edges, np.int64)).to(device=device)
    ts_edges = pad(edges, (1, 0), "constant", -2**63)
    ts = torch.from_numpy(ts.view(dtype=np.int64)).to(device=device)
    time_index = torch.searchsorted(ts_edges, ts, side='right').to(dtype=torch.int32, device=device)
    x = torch.from_numpy(x).to(dtype=torch.int32, device=device)
    y = torch.from_numpy(y).to(dtype=torch.int32, device=device)
    # Note: uses (int64 * ny + int?)*nx + int? which promotes to int64.
    # Rewriting as int64*ny*nx + int?*nx + int? may lose digits on y if it is int8 or int16.
    bin_index = ((time_index - 1)*nx + x)*ny + y # !!!! careful about type promotion !!!!
    source = torch.ones_like(bin_index)
    binned = torch.zeros((n_bins + 2)*nx*ny, dtype=torch.int32, device=device)
    #print("rebin_torch_index_add", time_index.dtype, tubeID.dtype, pixel.dtype, source.dtype, source.shape, bin_index.dtype, bin_index.shape)
    binned.index_add_(0, bin_index, source)
    binned = binned.reshape((n_bins+2, nx, ny))
    #print(f"events={len(ts)} binned={binned.sum()} trimmed={binned[1:-1].sum()}")

    #print(f"R:{binned[1:-1, 0:48].sum()} T:{binned[1:-1, 48:96].sum()}  B:{binned[1:-1, 96:144].sum()}  L:{binned[1:-1, 144:192].sum()}")
    # throw away the data in the outside bins
    binned = binned[1:-1].cpu().numpy()
    #print(f"trimmed={binned.sum()}")

    # returns: detectors data, and bin edges in seconds
    return binned

def _hist_torch_dd(dims, edges, ts, x, y):
    import torch

    device = _get_torch_gpu()

    # histogramdd is inclusive on the rightmost edge, so need to go one bin further
    edges = np.append(edges, edges[-1]+1)

    nx, ny = dims
    bins = (
        torch.from_numpy(edges.astype('float64')).to(device=device),
        torch.arange(nx+1, dtype=torch.float64, device=device),
        torch.arange(ny+1, dtype=torch.float64, device=device),
    )
    ts = torch.from_numpy(ts).to(dtype=torch.float64, device=device)
    x = torch.from_numpy(x).to(dtype=torch.float64, device=device)
    y = torch.from_numpy(y).to(dtype=torch.float64, device=device)
    data = torch.stack((ts, x, y), dim=1)
    # print("rebin_torch", ts.shape, data.shape, data.dtype, [v.dtype for v in bins])
    binned, edges = torch.histogramdd(data, bins=bins)
    binned = binned[:-1] # trim the right edge
    binned = binned.to('cpu').numpy()
    return np.asarray(binned, dtype='int32')

def _hist_numpy(dims, edges, ts, x, y):
    nx, ny = dims
    n_bins = len(edges) - 1
    # include two extra bins for the timestamps that fall outside the defined bins
    # (those with indices 0 and n_bins + 1); for an array of size n+1 searchsorted
    # returns insertion indices from 0 (below the left edge) to n+1 (past the right edge)
    binned = np.zeros((n_bins + 2, nx, ny), dtype='int32')
    # the operation below can be repeated... streaming histograms!
    time_index = np.searchsorted(edges, ts, side='right')
    np.add.at(binned, (time_index, x, y), 1)
    # throw away the data in the outside bins
    binned = binned[1:-1]
    return binned

def _hist_numpy_dd(dims, edges, ts, x, y):
    edges = np.asarray(edges, 'int64')
    # histogramdd is inclusive on the rightmost edge, so need to go one bin further
    edges = np.append(edges, edges[-1]+1)
    ts = np.asarray(ts, 'int64')
    x = np.asarray(x, 'int64')
    y = np.asarray(y, 'int64')

    nx, ny = dims
    bins = (
        edges,
        np.arange(nx+1, dtype='int64'),
        np.arange(ny+1, dtype='int64'),
    )
    data = np.stack((ts, x, y), axis=1)
    binned, edges = np.histogramdd(data, bins=bins)
    binned = binned[:-1] # trim the right edge
    return np.asarray(binned, dtype='int32')

def _hist_numba(dims, edges, ts, x, y):
    nx, ny = dims
    n_bins = len(edges) - 1
    index = np.argsort(ts, kind="stable")
    edges = np.asarray(edges, 'int64')
    ts = np.asarray(ts, 'int64')
    x = np.asarray(x, 'int32')
    y = np.asarray(y, 'int32')
    binned = np.zeros((n_bins, nx, ny), dtype='int32')
    _numba_binning(binned, edges, ts, x, y, index)
    return binned

# TODO: parallel algorithm
# After sorting (in parallel since the indices are already partially sorted?),
# partition the indices to the various processors, then for each process skip
# to the next edge. If still within the partition, then process until the next
# edge after the end of its partition, otherwise exit. There should be no read
# contention even though multiple processors may be reading data in the overlap
# region. There will be no write contention because every process is working
# in its own time slices.
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(*args, **kw):
        return lambda x: x
# print(f"{HAVE_NUMBA=}")

@njit('void(int32[:,:,:], int64[:], int64[:], int32[:], int32[:], int64[:])', cache=True)
def _numba_binning(bins, edges, ts, x, y, index):
    # Skip leading elements outside the histogram range
    next_edge = edges[0]
    j = 0
    while j < bins.size:
        ev = index[j]
        if ts[ev] >= next_edge:
            break
        j += 1

    # Build the histogram
    i = 0
    next_edge = edges[i+1]
    while j < ts.size:
        ev = index[j]
        if ts[ev] >= next_edge:
            i += 1
            if i == edges.size - 1:
                # Past the final edge, so we are done
                break
            next_edge = edges[i+1]
        else:
            bins[i, x[ev], y[ev]] += 1
            j += 1

    # Past the final edge or no more events so done
    return

# TODO: not sure why this is here
def force_numba_compile():
    edges = np.arange(2, dtype='uint64')
    tubeID = np.zeros(0, dtype='uint8')
    pixel = np.zeros(0, dtype='uint8')
    times = np.zeros(0, dtype='uint64')
    index = np.zeros(0, dtype='int64')
    _numba_binning(edges, tubeID, pixel, times, index)
#force_numba_compile()

# TODO: make a histogrammming backend selector
BACKEND = os.environ.get("BACKEND", "numba")
if BACKEND == "torch":
    hist = _hist_torch_addat
elif BACKEND == "torch_histdd":
    hist = _hist_torch_dd
elif BACKEND == "numpy":
    hist = _hist_numpy
elif BACKEND == "numpy_histdd":
    hist = _hist_numpy_dd
elif BACKEND == "numba":
    hist = _hist_numba if HAVE_NUMBA else _hist_numpy
else:
    raise ValueError(f"Unknown histogram backend BACKEND={BACKEND}")

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
        nx, ny = dims
        index = np.searchsorted(edges, ts-triggers)
        data = np.zeros((nx, ny, nbins+2), 'int32')
        np.add.at(data, (x, y, index), 1)
        binned[name] = data[:, :, 1:-1]
        print(f"{name} {dims} bins={len(edges)-1} events={len(ts):<8d} keeping={binned[name].sum():<8d}")

    devices = events.get("devices", {})
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
        # max/min.
        # TODO: max/min might not be sampled.
        # Could use a cubic rather than a linear model to allow overshoot.
    result['devices'] = binned_devices
    return result


def _find_device_edges(events, edges, name):
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
    return bin_ts, bin_index, bin_up


def _merge_edges(ts1, idx1, up1, ts2, idx2, up2):
    # Near equivalent in pandas:
    #   import pandas as pd
    #   t1, v1 = [1, 2, 5, 7, 9], ['a', 'b', 'c', 'd', 'e']
    #   t2, v2 = [3, 4, 5, 6, 10], [3.0, 4.0, 5.0, 6., 10.]
    #   df1 = pd.DataFrame(dict(time=t1, v1=v1))
    #   df2 = pd.DataFrame(dict(time=t2, v2=v2))
    #   df_left = pd.merge_asof(df1, df2, on='time', direction='backward')
    #   df_right = pd.merge_asof(df2, df1, on='time', direction='backward')
    #   df = pd.concat((df_left,df_right)).sort_values(by="time").drop_duplicates()
    # One difference is that column v2 will be NaN for times before the first t2
    # whereas the code below assigns the initial value

    pairs = []
    k1 = k2 = 0
    while k1 < len(ts1) and k2 < len(ts2):
        if ts1[k1] < ts2[k2]: # next boundary is on timeseries 1
            pairs.append((ts1[k1], idx1[k1], idx2[k2], up[k1], up[k2]))
            k1 += 1
        else: # next boundary is on timeseries 2 (or shared with ts1)
            pairs.append((ts2[k2], idx1[k1], idx2[k2], up[k1], up[k2]))
            if ts1[k1] == ts2[k2]:
                k1 += 1
            k2 += 1
    for k in range(k1, len(ts1)):
        pairs.append((ts1[k], idx1[k], idx2[-1], up1[k], up2[-1]))
    for k in range(k2, len(ts2)):
        pairs.append((ts2[k], idx1[-1], idx2[k], up1[-1], up2[k]))
    return [np.array(v) for v in zip(*pairs)]


def _bin_by_device(events, edges, name, hysteresis=True):
    bin_ts, bin_index, bin_up = _find_device_edges(events, edges, name)

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
        nx, ny = dims
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

    return result



def test_hist():
    # 3 x 2 detector with 4 timesteps
    dims = (3, 2)
    edges = [2, 5, 12, 18, 24]
    events = [
        (3, 1, 0),  # t=3, x=1, y=0
        (3, 1, 0),  # Another event at the same coordinates
        (5, 2, 1),  # On a left bin boundary
        (12, 2, 1),  # On a right bin boundary
        (20, 1, 1), # Out of order events
        (15, 0, 0), # Out of order events
        (2, 2, 0),  # At the first bin edge
        (24, 2, 0),  # At the last bin edge
        (1, 1, 0),  # Before the first time bin
        (25, 1, 1),  # After the last time bin
    ]
    ts, x, y = zip(*events)
    target = np.zeros((4, 3, 2), dtype='int32')
    target[0,1,0] = 2 # 3,1,0 x 2
    target[1,2,1] = 1 # 5,2,1
    target[2,2,1] = 1 # 12,2,1
    target[3,1,1] = 1 # 20,1,1
    target[2,0,0] = 1 # 15,0,0
    target[0,2,0] = 1 # 2,2,0
    #print(target)

    edges = np.asarray(edges, dtype='int64')
    x = np.asarray(x, dtype='int32')
    y = np.asarray(y, dtype='int32')
    ts = np.asarray(ts, dtype='int64')
    # The remainder are not in the histogram
    for backend in (
            _hist_torch_addat,
            _hist_numba,
            _hist_numpy,
            _hist_numpy_dd,
            _hist_torch_dd,
        ):
        # Using positional arguments to verify that x,y are in the correct order in all backends
        data = backend(dims, edges, ts, x, y)
        # print(f"{backend.__name__}\n{data}")
        assert data.shape == target.shape, f"Shape mismatch for {backend.__name__}: {data.shape}"
        assert (data == target).all(), f"Match fails for {backend.__name__}:\n{data}\n{target}"

if __name__ == "__main__":
    test_hist()
