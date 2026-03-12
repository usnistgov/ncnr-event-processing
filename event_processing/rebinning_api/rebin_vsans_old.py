from pathlib import Path
from typing import Callable, Dict, Literal, TypeAlias

import requests
import numpy as np


#ev = open("./sample_events/20190211203348712_0.hst", "rb")
# this seems to correspond to regular file:
# https://ncnr.nist.gov/ncnrdata/view/nexus-hdf-viewer.html?pathlist=ncnrdata+vsans+201902+25309+data&filename=sans27334.nxs.ngv

# timestamp resolution seems to be 100 ns:
# https://github.com/sansigormacros/ncnrsansigormacros/blob/ee3680d660331c0748343d24da931169b4984645/NCNR_User_Procedures/Reduction/VSANS/V_EventModeProcessing.ipf#L1231

TIMESTAMP_RESOLUTION = 100e-9

EVENTS_FOLDER = "cache/event_files"
EVENTS_ENDPOINT = "http://nicedata.ncnr.nist.gov/eventfiles"

NUM_TUBE = 192
NUM_PIXEL = 128

def eventfiles_from_nexus(nexus): #, events_folder=EVENTS_FOLDER):
    intrument = "vsans"
    files = set()
    for name, entry in nexus.items():
        for device, group in entry['instrument'].items():
            if 'event_file_name' in group:
                files.add(group['event_file_name'][0].decode())
    return instrument, files

def fetch_eventfile(instrument, eventfile, events_folder=EVENTS_FOLDER, overwrite=False):
    #print("events folder", events_folder)
    events_folder = Path(events_folder)
    events_folder.mkdir(parents=True, exist_ok=True)
    fullpath = events_folder / eventfile
    if overwrite or not fullpath.exists():
        url = EVENTS_ENDPOINT
        #print(f"retrieving eventfile {eventfile} from {url}")
        r = requests.get(url, params={"instrument": instrument, "filename": eventfile})
        if r.ok:
            open(fullpath, 'wb').write(r.content)
            print(f"Fetched {eventfile}")
        else:
            # TODO: maybe store an empty missing file?
            print(f"Failure: {r.status_code} '{r.reason}' during {url}?instrument={instrument}&filename={eventfile}")
    return fullpath

def retrieve_events(nexus, events_folder=EVENTS_FOLDER, overwrite=False):
    instrument, files = eventfiles_from_nexus(nexus)
    #print(f"Event files in {nexus_path}", files)
    paths = []
    for eventfile in sorted(files):
        fetch_eventfile(instrument, eventfile, events_folder=events_folder, overwrite=overwrite)
        paths.append(Path(events_folder) / eventfile)
    return paths

REBIN_BACKEND = Literal["torch", "numpy", "numba"]

class VSANSEvents(object):
    header_dtype = np.dtype([ 
        ('magic_number', 'S5'),
        ('revision', 'u2'),
        ('data_offset', 'u2'),
        ('origin_timestamp', '10u1'),
        ('detector_carriage_group', 'S1'),
        ('HV_reading', 'u2'),
        ('timestamp_frequency', 'u4'),
        # List of disabled detectors follows the rest of the header.
        # ('disabled', '*u1'), # * is (data_offset - size(header_dtype))
    ])

    # Data in timestamp file is 8 bytes per event, with a 6 byte timestamp
    data_dtype = np.dtype([
        ('tubeID', 'u1'),
        ('pixel',  'u1'),
        ('timestamp', '6u1')
    ])

    rebin_backend: REBIN_BACKEND

    def __init__(self, filename, rebin_backend: REBIN_BACKEND ="numba"):
        self.file = open(filename, 'rb')
        self.header = np.fromfile(self.file, dtype=self.header_dtype, count=1, offset=0)
        self.data_offset = self.header['data_offset'][0]
        header_size = self.header_dtype.itemsize
        num_disabled = self.data_offset - header_size # 1 byte per disabled tube.
        self.disabled_tubes = np.fromfile(self.file, count=num_disabled, offset=header_size, dtype='u1')
        self.read_data()
        self.rebin_backend = rebin_backend
 
    def _repeat(self, n):
        # Clone data so we can do speed tests on large event streams
        self.tubeID = np.tile(self.tubeID, n)
        self.pixel = np.tile(self.pixel, n)
        self.ts = np.tile(self.ts, n)

    def seek_data_start(self):
        self.file.seek(self.data_offset)

    @property
    def simple_header(self):
        keys = self.header_dtype.names
        values = [self.header[k] for k in keys]
        values = [v[0] if len(v) == 1 else v for v in values]
        return dict(zip(keys, values))

    def read_data(self):
        self.seek_data_start()
        data = np.fromfile(self.file, dtype=self.data_dtype, count=-1)
        self.file.close()

        # convert timestamps from 6 byte LE to eight byte LE
        ts = data['timestamp']
        self.ts = np.pad(ts, ((0,0), (0, 2)), 'constant').view(np.uint64)[:,0]
        self.tubeID = data['tubeID']
        self.pixel = data['pixel']

    def rebin_numpy(self, time_slices=10):
        if hasattr(time_slices, 'size'):
            # then it's an array, treat as bin edges in seconds:
            time_slices = time_slices / TIMESTAMP_RESOLUTION
        time_edges = np.histogram_bin_edges(self.ts, bins=time_slices)
        #print("edges", time_edges)
        time_bins = np.searchsorted(time_edges, self.ts, side='left')
        n_bins = len(time_edges) - 1
        #print(n_bins, time_slices, time_edges, time_bins)

        # include two extra bins for the timestamps that fall outside the defined bins
        # (those with indices 0 and n_bins + 1); for an array of size n+1 searchsorted
        # returns insertion indices from 0 (below the left edge) to n+1 (past the right edge)
        binned = np.zeros((n_bins + 2, NUM_TUBE, NUM_PIXEL))
        # the operation below can be repeated... streaming histograms!
        np.add.at(binned, (time_bins, self.tubeID, self.pixel), 1)
        # throw away the data in the outside bins
        binned = binned[1:-1, :, :]

        detectors = {
            "right": binned[:, 0:48, :].reshape((n_bins, NUM_PIXEL, 48)),
            "left": binned[:, 144:192, :].reshape((n_bins, NUM_PIXEL, 48)),
            "top": binned[:, 48:96, :].swapaxes(1,2).reshape((n_bins, NUM_PIXEL, 48)),
            "bottom": binned[:, 96:144, :].swapaxes(1,2).reshape((n_bins, NUM_PIXEL, 48)),
        }

        # returns: detectors data, and bin edges in seconds
        return detectors, time_edges * TIMESTAMP_RESOLUTION

    def rebin_torch_histogramdd(self, time_slices=10):
        import torch

        if hasattr(time_slices, 'size'):
            # then it's an array, treat as bin edges in seconds:
            time_slices = time_slices / TIMESTAMP_RESOLUTION
        else:
            raise NotImplementedError("time slices must be a vector")
            time_slices = np.histogram_bin_edges(self.ts, bins=time_slices)

        bins = (
            torch.from_numpy(time_slices.astype('float64')),
            torch.arange(NUM_TUBE+1, dtype=torch.float64),
            torch.arange(NUM_PIXEL+1, dtype=torch.float64),
        )
        ts = torch.from_numpy(self.ts.view(dtype=np.int64)).to(torch.float64)
        tubeID = torch.from_numpy(self.tubeID).to(torch.float64)
        pixel = torch.from_numpy(self.pixel).to(torch.float64)
        data = torch.stack((ts, tubeID, pixel), dim=1)
        #print("rebin_torch", ts.shape, data.shape, data.dtype, [v.dtype for v in bins])
        binned, edges = torch.histogramdd(data, bins=bins)
        binned = binned.numpy()
        # n_bins = len(edges[0]) - 1

        detectors = {
            "right": binned[:, 0:48, :].reshape((n_bins, NUM_PIXEL, 48)),
            "left": binned[:, 144:192, :].reshape((n_bins, NUM_PIXEL, 48)),
            "top": binned[:, 48:96, :].swapaxes(1,2).reshape((n_bins, NUM_PIXEL, 48)),
            "bottom": binned[:, 96:144, :].swapaxes(1,2).reshape((n_bins, NUM_PIXEL, 48)),
        }

        # returns: detectors data, and bin edges in seconds
        return detectors, time_slices * TIMESTAMP_RESOLUTION

    def rebin_torch_addat(self, time_slices=10):
        import torch
        from torch.nn.functional import pad

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #device = 'cpu'
        if hasattr(time_slices, 'size'):
            # then it's an array, treat as bin edges in seconds:
            edges = time_slices / TIMESTAMP_RESOLUTION
        else:
            raise NotImplementedError("time slices must be a vector")
            edges = np.histogram_bin_edges(self.ts, bins=time_slices)

        # include two extra bins for the timestamps that fall outside the defined bins
        # (those with indices 0 and n_bins + 1); for an array of size n+1 searchsorted
        # returns insertion indices from 0 (below the left edge) to n+1 (past the right edge)
        # To get the indexing to work right for values outside the range, need an
        # extra zero for the left edge.
        n_bins = len(edges) - 1
        edges = torch.from_numpy(np.asarray(edges,np.int64)).to(device=device)
        ts_edges = pad(edges, (1, 0), "constant", -2**63)
        ts = torch.from_numpy(self.ts.view(dtype=np.int64)).to(device=device)
        time_index = torch.searchsorted(ts_edges, ts, side='right').to(dtype=torch.int32, device=device)
        tubeID = torch.from_numpy(self.tubeID).to(device=device)
        pixel = torch.from_numpy(self.pixel).to(device=device)
        # Warning: tubeID:uint8 * scalar => uint8 so can't do tubeID*NUM_PIXEL.
        # However we can do (time_index:int32 + tubeID:uint8)*scalar => int32.
        # Go with the faster, lower memory form, with the understanding that
        # it'll be very confusing for the next person who changes this
        bin_index = ((time_index - 1)*NUM_TUBE + tubeID)*NUM_PIXEL + pixel # !!!! uint8 to int32 type promotion can be surprising !!!!
        source = torch.ones_like(bin_index)
        binned = torch.zeros((n_bins + 2)*NUM_TUBE*NUM_PIXEL, dtype=torch.int32, device=device)
        #print("rebin_torch_index_add", time_index.dtype, tubeID.dtype, pixel.dtype, source.dtype, source.shape, bin_index.dtype, bin_index.shape)
        binned.index_add_(0, bin_index, source)
        binned = binned.reshape((n_bins+2, NUM_TUBE, NUM_PIXEL))
        #print(f"events={len(ts)} binned={binned.sum()} trimmed={binned[1:-1].sum()}")

        #print(f"R:{binned[1:-1, 0:48].sum()} T:{binned[1:-1, 48:96].sum()}  B:{binned[1:-1, 96:144].sum()}  L:{binned[1:-1, 144:192].sum()}")
        # throw away the data in the outside bins
        binned = binned[1:-1, :, :].cpu().numpy()
        #print(f"trimmed={binned.sum()}")

        detectors = {
            "right": binned[:, 0:48, ::-1].copy(),
            "left": binned[:, 192:144:-1, :].copy(),
            "top": binned[:, 48:96, :].swapaxes(1,2).copy(),
            "bottom": binned[:, 144:96:-1, ::-1].swapaxes(1,2).copy(),
        }

        # returns: detectors data, and bin edges in seconds
        return detectors, edges * TIMESTAMP_RESOLUTION

    def rebin_numba(self, time_slices=10):
        print("rebin_numba", time_slices)
        try:
            import numba
        except ImportError:
            raise NotImplementedError("Needs numba")
        if hasattr(time_slices, 'size'):
            # then it's an array, treat as bin edges in seconds:
            edges = time_slices / TIMESTAMP_RESOLUTION
        else:
            raise NotImplementedError("time slices must be a vector")
            edges = np.histogram_bin_edges(self.ts, bins=time_slices)

        #import torch
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #ts = torch.from_numpy(self.ts.view(dtype=np.int64)).to(device=device)
        #index = torch.argsort(ts).cpu()
        n_bins = len(edges) - 1
        index = np.argsort(self.ts)
        edges = np.asarray(edges, 'uint64')

        #binned = np.zeros((edges.size-1, 192, 128), dtype='int32')
        binned = numba_binning(edges, self.ts, self.tubeID, self.pixel, index)

        not_detectors = {
            "right": np.fliplr(binned[:, 0:48, :]),
            "left": np.flipud(binned[:, 144:192, :]),
            "top": (binned[:, 48:96, :]).swapaxes(1, 2),
            "bottom": np.flipud(np.fliplr((binned[:, 96:144, :]).swapaxes(1, 2)))
        }
        detectors = {
            "right": binned[:, 0:48, :].reshape((n_bins, NUM_PIXEL, 48)),
            "left": binned[:, 144:192, :].reshape((n_bins, NUM_PIXEL, 48)),
            "top": binned[:, 48:96, :].swapaxes(1,2).reshape((n_bins, NUM_PIXEL, 48)),
            "bottom": binned[:, 96:144, :].swapaxes(1,2).reshape((n_bins, NUM_PIXEL, 48)),
        }

        # returns: detectors data, and bin edges in seconds
        return detectors, edges * TIMESTAMP_RESOLUTION

    REBIN_FUNCTION_LOOKUP: Dict[REBIN_BACKEND, Callable] = {
        "torch": rebin_torch_addat,
        "numpy": rebin_numpy,
        "numba": rebin_numba,
    }

    def rebin(self, time_slices=10):
        """Dispatch to the selected rebin backend.

        Parameters
        ----------
        time_slices : int | np.ndarray
            Same meaning as in the individual backend methods.
        """
        backend_func = self.REBIN_FUNCTION_LOOKUP.get(self.rebin_backend)
        if backend_func is None:
            raise ValueError(f"Unsupported rebin backend '{self.rebin_backend}'. Available: {list(self.REBIN_BACKENDS)}")
        # Call the unbound function with the instance (self) as the first argument.
        return backend_func(self, time_slices)


    def counts_vs_time(self, start_time=0, timestep=1.0):
        """ get total counts on all detectors as a function of time,
        where the time bin size = timestep (in seconds) """
        max_timestamp = self.ts.max()
        start_timestamp = start_time / TIMESTAMP_RESOLUTION
        timestamp_step = timestep/TIMESTAMP_RESOLUTION
        bin_edges = np.arange(start_time, max_timestamp + timestamp_step, timestamp_step)
        hist, _ = np.histogram(self.ts, bins=bin_edges, range=(start_timestamp, max_timestamp))
        time_axis = (bin_edges[:-1] + timestamp_step/2.0) * TIMESTAMP_RESOLUTION
        return time_axis, hist

# TODO: parallel algorithm
# After sorting (in parallel since the indices are already partially sorted?),
# partition the indices to the various processors, then for each process skip
# to the next edge. If still within the partition, then process until the next
# edge after the end of its partition, otherwise exit. There should be no read
# contention even though multiple processors may be reading data in the overlap
# region. There will be no write contention because every process is working
# in its own time slices.
# TODO: numba binning not used at the moment, so don't require numba package
try:
    from numba import njit
except ImportError:
    def njit(*args, **kw):
        return lambda x: x
@njit('int32[:,:,:](uint64[:], uint64[:], uint8[:], uint8[:], int64[:])', cache=True)
def numba_binning(edges, times, tubeID, pixelID, index):
    bins = np.zeros((edges.size-1, 192, 128), dtype='int32')

    # Skip leading elements outside the histogram range
    next_edge = edges[0]
    j = 0
    while j < bins.size:
        ev = index[j]
        if times[ev] >= next_edge:
            break
        j += 1

    # Build the histogram
    i = 0
    next_edge = edges[i+1]
    while j < times.size:
        ev = index[j]
        if times[ev] >= next_edge:
            i += 1
            if i == edges.size - 1:
                # Past the final edge, so we are done
                break
            next_edge = edges[i+1]
        else:
            bins[i, tubeID[ev], pixelID[ev]] += 1
            j += 1

    # Past the final edge or no more events so done
    return bins

def force_compile():
    edges = np.arange(2, dtype='uint64')
    tubeID = np.zeros(0, dtype='uint8')
    pixel = np.zeros(0, dtype='uint8')
    times = np.zeros(0, dtype='uint64')
    index = np.zeros(0, dtype='int64')
    numba_binning(edges, tubeID, pixel, times, index)
#force_compile()

def demo():
    from matplotlib import pyplot as plt

    cache = "cache/event_files"
    runs = [
        "20201008221350744",
        "20201009143217794",
        "20201010115802114",
        "20201011115726522",
    ]
    run = runs[0]
    events = {}
    for position in [0, 1]:
        filename = f"{cache}/{run}_{position}.hst"
        events[position] = VSANSEvents(filename)

    for position in [0, 1]:
        detectors, edges = events[position].rebin(100)
        for name, data in detectors.items():
            integrated = np.sum(data, axis=0)
            integrated = np.sum(integrated, axis=0)
            #print(name, 'sum:', integrated, integrated.shape)
            plt.plot(edges[:-1], integrated, label=f"{name}-{position}")
    plt.legend()

    if 0:
        times, counts0 = events[0].counts_vs_time(timestep=1.0)
        times, counts1 = events[1].counts_vs_time(timestep=1.0)
        index = slice(None)
        #index = (times >= 250)&(times<=1800)

        plt.figure()
        plt.plot(times[index], counts0[index], label=f"position 0")
        plt.plot(times[index], counts1[index], label=f"position 1")
        plt.legend()

    plt.show()

    return events

if __name__ == "__main__":
    demo()