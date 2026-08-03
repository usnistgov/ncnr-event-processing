from pathlib import Path

import requests
import numpy as np

from .binning import hist

#ev = open("./sample_events/20190211203348712_0.hst", "rb")
# this seems to correspond to regular file:
# https://ncnr.nist.gov/ncnrdata/view/nexus-hdf-viewer.html?pathlist=ncnrdata+vsans+201902+25309+data&filename=sans27334.nxs.ngv

# timestamp resolution seems to be 100 ns:
# https://github.com/sansigormacros/ncnrsansigormacros/blob/ee3680d660331c0748343d24da931169b4984645/NCNR_User_Procedures/Reduction/VSANS/V_EventModeProcessing.ipf#L1231

TIMESTAMP_RESOLUTION = 100e-9

# EVENTS_FOLDER = "cache/event_files"
EVENTS_ENDPOINT = "http://nicedata.ncnr.nist.gov/eventfiles"
EVENTS_FOLDER = Path(__file__).parent.parent.parent / "cache" / "event_files"

NUM_TUBE = 192
NUM_PIXEL = 128

def eventfiles_from_nexus(nexus): #, events_folder=EVENTS_FOLDER):
    instrument = "vsans"
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

    def __init__(self, filename):
        self.file = open(filename, 'rb')
        self.header = np.fromfile(self.file, dtype=self.header_dtype, count=1, offset=0)
        self.data_offset = self.header['data_offset'][0]
        header_size = self.header_dtype.itemsize
        num_disabled = self.data_offset - header_size # 1 byte per disabled tube.
        self.disabled_tubes = np.fromfile(self.file, count=num_disabled, offset=header_size, dtype='u1')
        self.read_data()
 
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

        self.tubeID = data['tubeID'].copy()
        self.pixel = data['pixel'].copy()

        num_events = len(data)
        self.ts = np.zeros(num_events, dtype=np.uint64)
        ts_bytes = self.ts.view(np.uint8).reshape(num_events, 8)
        ts_bytes[:, :6] = data['timestamp']

    def rebin(self, time_slices=10):
        if hasattr(time_slices, 'size'):
            # then it's an array, treat as bin edges in seconds:
            edges = time_slices / TIMESTAMP_RESOLUTION
        elif np.isscalar(time_slices):
            edges = np.linspace(self.ts.min(), self.ts.max(), time_slices+1)
            #raise NotImplementedError("time slices must be a vector")

        dims = (NUM_TUBE, NUM_PIXEL)
        binned = hist(dims, edges, self.ts, self.tubeID, self.pixel)
        # print(f"{dims=} {edges.shape=} {self.ts.shape=} {self.tubeID.shape=} {self.pixel.shape=} {binned.shape=}")
        detectors = {
            "right": binned[:, 0:48, ::-1].swapaxes(1,2).copy(),
            "left": binned[:, 192:144:-1, :].swapaxes(1,2).copy(),
            "top": binned[:, 48:96, :].copy(),
            "bottom": binned[:, 144:96:-1, ::-1].copy(),
        }

        # returns: detectors data, and bin edges in seconds
        return detectors, edges * TIMESTAMP_RESOLUTION

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

def demo():
    from matplotlib import pyplot as plt

    # cache = "cache/event_files"
    cache = EVENTS_FOLDER
    runs = [
        "20200923065024850",
        "20201008221350744",
        # "20201009143217794",
        # "20201010115802114",
        # "20201011115726522",
    ]
    run = runs[0]
    events = {}
    for position in [0, 1]:
        filename = f"{cache}/{run}_{position}.hst"
        ev = VSANSEvents(filename)

        detectors, edges = ev.rebin(100)
        for name, data in detectors.items():
            integrated = np.sum(data, axis=-1)
            integrated = np.sum(integrated, axis=-1)
            #print(name, 'sum:', integrated, integrated.shape)
            plt.plot(edges[:-1], integrated, label=f"{name}-{position}")

        del ev

    plt.legend()
    plt.show()

    # for position in [0, 1]:
    #     detectors, edges = events[position].rebin(100)
    #     for name, data in detectors.items():
    #         integrated = np.sum(data, axis=-1)
    #         integrated = np.sum(integrated, axis=-1)
    #         #print(name, 'sum:', integrated, integrated.shape)
    #         plt.plot(edges[:-1], integrated, label=f"{name}-{position}")
    # plt.legend()

    if 1:
        # Shared (vmin, vmax) for colormap
        vmin = min(d.min() for d in detectors.values())
        vmax = max(d.max() for d in detectors.values())
        frame = 5
        plt.figure()
        plt.subplot(121)
        plt.pcolor(detectors["right"][frame], vmin=vmin, vmax=vmax)
        nf, ny, nx = detectors["left"].shape
        plt.pcolor(np.arange(-nx, 1), np.arange(ny+1), detectors["left"][frame], vmin=vmin, vmax=vmax)
        plt.axis('equal')
        plt.subplot(122)
        plt.pcolor(detectors["top"][frame], vmin=vmin, vmax=vmax)
        nf, ny, nx = detectors["bottom"].shape
        plt.pcolor(np.arange(nx+1), np.arange(-ny, 1), detectors["bottom"][frame], vmin=vmin, vmax=vmax)
        plt.axis('equal')

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