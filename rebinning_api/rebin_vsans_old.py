from pathlib import Path

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
        self.process_timestamps()

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
        self.data = np.fromfile(self.file, dtype=self.data_dtype, count=-1)
        self.file.close()

    def process_timestamps(self):
        ts = self.data['timestamp']
        self.ts = np.pad(ts, ((0,0), (0, 2)), 'constant').view(np.uint64)[:,0]

    def rebin(self, time_slices=10):
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
        time_sliced_output = np.zeros((192, 128, n_bins + 2))
        # the operation below can be repeated... streaming histograms!
        np.add.at(time_sliced_output, (self.data['tubeID'], self.data['pixel'], time_bins), 1)
        # throw away the data in the outside bins
        time_sliced_output = time_sliced_output[:,:,1:-1]


        detectors = {
            "right": np.fliplr(time_sliced_output[0:48]),
            "left": np.flipud(time_sliced_output[144:192]),
            "top": (time_sliced_output[48:96]).swapaxes(0,1),
            "bottom": np.flipud(np.fliplr((time_sliced_output[96:144]).swapaxes(0,1)))
        }

        # returns: detectors data, and bin edges in seconds
        return detectors, time_edges * TIMESTAMP_RESOLUTION

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