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
        result = _bin_by_device(events, bins.edges, bins.device, hysterisis=bins.hysterisis)
    else:
        raise KeyError(f"Unrecognized bin mode {bins.mode}")


    # Convert detectors from area to linear or point as necessary
    result['detector'] = _squeeze_detectors(result['detector'])
    # Extract monitor into a separate reult
    monitor = result['detector'].pop('monitor', None)
    if monitor is not None:
        result['monitor'] = monitor


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


def _bin_by_device(events, edges, name, hysterisis=True):
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

    return result
