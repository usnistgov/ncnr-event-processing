from pathlib import Path

import requests
import numpy as np

from .binning import hist
from .decoders import decode_ordela_events, parse_ascii_hex_words, ATXYM
from .memlog import log_mem
from . import util

#ev = open("./sample_events/20190211203348712_0.hst", "rb")
# this seems to correspond to regular file:
# https://ncnr.nist.gov/ncnrdata/view/nexus-hdf-viewer.html?pathlist=ncnrdata+vsans+201902+25309+data&filename=sans27334.nxs.ngv

# timestamp resolution seems to be 100 ns:
# https://github.com/sansigormacros/ncnrsansigormacros/blob/ee3680d660331c0748343d24da931169b4984645/NCNR_User_Procedures/Reduction/VSANS/V_EventModeProcessing.ipf#L1231

TIMESTAMP_RESOLUTION = 100e-9

# EVENTS_FOLDER = "cache/event_files"
EVENTFILE_NAME_KEY = "eventFileName"
EVENTS_ENDPOINT = "http://nicedata.ncnr.nist.gov/eventfiles"
CACHE_FOLDER = Path.cwd() / "cache"
EVENTS_FOLDER = CACHE_FOLDER / "event_files"
NEXUS_FOLDER = CACHE_FOLDER / "nexus_files"

NUM_TUBE = 192
NUM_PIXEL = 128

def eventfiles_from_nexus(nexus): #, events_folder=EVENTS_FOLDER):
    instrument = "vsans"
    files = set()
    for name, entry in nexus.items():
        for device, group in entry['instrument'].items():
            if 'eventFileName' in group:
                files.add(group['eventFileName'][0].decode())
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

# One .hst file is recorded per detector carriage (front/middle), and each
# file contains events for all four quadrants on that carriage, distinguished
# by tubeID range.
FRONT_MIDDLE_DETECTORS = ("FL", "ML")
TUBE_QUADRANTS = (
    ("R", 0, 48),
    ("T", 48, 96),
    ("B", 96, 144),
    ("L", 144, 192),
)

def find_event_files(entry, events_folder=EVENTS_FOLDER):
    """
    Look up the event file names for the front and middle detector carriages
    of this nexus entry, and return their expected paths within events_folder.
    """
    events_folder = Path(events_folder)
    paths = set()
    for item in entry.get('DAS_logs', {}).values():
        if hasattr(item, 'get'):
            if EVENTFILE_NAME_KEY in item:
                paths.add(Path(events_folder) / item[EVENTFILE_NAME_KEY][0].decode())
    return paths

_HEX_DIGITS_AND_WHITESPACE = b'0123456789abcdefABCDEF\r\n \t'

def _is_ascii_hex(sample: bytes) -> bool:
    """
    Heuristic check for the ASCII-hex event format from
    NCNR_User_Procedures/Reduction/SANS/EventModeProcessing.ipf: each event
    word is written as hex text, one per line, so the leading bytes of such
    a file are nothing but hex digits, whitespace, and line endings -- a raw
    binary Ordela/VAX event word essentially never looks like that.
    """
    return all(b in _HEX_DIGITS_AND_WHITESPACE for b in sample)

def events_manager_from_files(entry, events_folder=EVENTS_FOLDER):
    """
    Build an EventsManager populated from local vsans .hst event files,
    for use in place of pulling events from the live kafka stream.

    Raises FileNotFoundError if the event files referenced by the nexus
    entry are not present in events_folder.
    """
    from .event_capture import EventsManager

    events_folder = Path(events_folder)
    paths = find_event_files(entry, events_folder=events_folder)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"No event files found: {', '.join(missing)} (looked in {events_folder})"
        )

    db = EventsManager(None)
    # Legacy event timestamps are ticks from the start of the count, so
    # there is no arm/disarm offset to remove during cleanup.
    db.set_times([0], [0], 0, 0)

    detector_index = 0
    instrument = util.lookup_instrument(entry)

    if instrument == "vsans":
        for hst_path in paths:
            hst = VSANSEvents(str(hst_path))
            log_mem(f"loaded {hst_path}")
            # TUBE_QUADRANTS are uniform 48-wide buckets, so tubeID // 48 gives
            # the quadrant index directly in one pass, instead of computing each
            # quadrant's mask with its own (tubeID >= low) & (tubeID < high),
            # which is 3 full-array passes repeated 4 times over the same array.
            quadrant_idx = hst.tubeID // 48
            log_mem(f"computed quadrant index {hst_path}")

            for q, (_, low, high) in enumerate(TUBE_QUADRANTS):
                mask = quadrant_idx == q
                log_mem(f"masked {hst_path} tube {low}-{high}")

                timestamps = hst.ts[mask]  # already int64, no upcast needed
                timestamps *= 100  # 100 ns ticks -> ns, in place
                log_mem(f"converted timestamps {hst_path} tube {low}-{high}")

                # x/y are already separate columns in the .hst file (unlike the
                # live kafka stream, which only ever has a packed pixel_id), so
                # hand them to cleanup() directly instead of packing into a
                # pixel_id here just to have cleanup() unpack it again.
                x = hst.tubeID[mask]
                y = hst.pixel[mask]
                log_mem(f"extracted x/y {hst_path} tube {low}-{high}")

                name = f"detector_{detector_index}"
                detector_index += 1
                db._fields[name] = {"timestamp": [timestamps], "x": [x], "y": [y]}
                del mask
            del hst, quadrant_idx
    else:
        for hst_path in paths:
            with open(hst_path, 'rb') as f:
                magic = f.read(16)
            if magic[:3] == b'NAS':
                hst = SANSEvents(str(hst_path))
            elif _is_ascii_hex(magic):
                hst = OrdelaSANSEventsASCII(str(hst_path))
            else:
                hst = OrdelaSANSEvents(str(hst_path))
            log_mem(f"loaded {hst_path}")
            timestamps = hst.ts
            if not isinstance(hst, SANSEvents):
                # Ordela/VAX ticks are 100 ns; SANSEvents (NAS format) ticks
                # are already 1 ns, per timestamp_frequency in its header.
                timestamps *= 100  # 100 ns ticks -> ns, in place
            log_mem(f"converted timestamps {hst_path}")
            x = hst.x
            y = hst.y
            log_mem(f"extracted x/y {hst_path}")
            name = f"detector_{detector_index}"
            detector_index += 1
            db._fields[name] = {"timestamp": [timestamps], "x": [x], "y": [y]}
            del hst
    return db


class VSANSEvents(object):
    header_dtype = np.dtype([ 
        ('magic_number', 'S5'), # "VSANS"
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
        self.filename = filename
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

    @property
    def simple_header(self):
        keys = self.header_dtype.names
        values = [self.header[k] for k in keys]
        values = [v[0] if len(v) == 1 else v for v in values]
        return dict(zip(keys, values))

    def read_data(self):
        self.file.close()
        # Memory-map the 8-byte events instead of reading them into a single
        # anonymous buffer with np.fromfile. A plain fromfile read of a
        # several-GB file is one big allocation of swap-eligible memory for
        # the whole file; a read-only mmap is backed by the file itself, so
        # under memory pressure the OS can drop clean pages and re-fault them
        # from disk instead of having to page them out to (and back in from)
        # swap.
        raw_data = np.memmap(self.filename, dtype=np.int64, mode='r', offset=int(self.data_offset))

        num_events = len(raw_data)
        ts_bytes = raw_data.view(np.uint8).reshape(num_events, 8)

        # Byte 0 is the tubeID (lowest 8 bits)
        self.tubeID = ts_bytes[:, 0].copy()

        # Byte 1 is the pixel (next 8 bits)
        self.pixel = ts_bytes[:, 1].copy()

        # Bytes 2 through 7 are the 6-byte timestamp (upper 48 bits).
        # raw_data is a read-only mmap of the source file, so this has to be
        # an out-of-place shift (>>=  would try to write through to the file).
        self.ts = raw_data >> 16


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

class SANSEvents(object):
    """
    Reader for 10m SANS-Tubes event-mode files (NAS format), per the layout
    documented in NCNR_User_Procedures/Reduction/SANS_Nexus/EventModeProcessing_Tubes_N.ipf.
    """
    # 23-byte header, little-endian per the NAS format spec.
    header_dtype = np.dtype([
        ('magic_number', 'S3'),           # 'NAS'
        ('revision', '<u2'),
        ('data_offset', '<u2'),           # offset to event data, in bytes
        ('origin_timestamp', '10u1'),     # UTC base timestamp (IEEE1588)
        ('HV_reading', '<u2'),            # high voltage reading, in volts
        ('timestamp_frequency', '<u4'),   # timestamping clock frequency, in Hz
    ])

    # Data section is 10 bytes per event: 1-byte x, 1-byte y, 8-byte timestamp.
    data_dtype = np.dtype([
        ('x', 'u1'),
        ('y', 'u1'),
        ('timestamp', '<u8'),
    ])

    def __init__(self, filename):
        self.filename = filename
        with open(filename, 'rb') as f:
            self.header = np.fromfile(f, dtype=self.header_dtype, count=1, offset=0)
        self.data_offset = int(self.header['data_offset'][0])
        self.timestamp_frequency = int(self.header['timestamp_frequency'][0])
        self.read_data()

    @property
    def simple_header(self):
        keys = self.header_dtype.names
        values = [self.header[k] for k in keys]
        values = [v[0] if len(v) == 1 else v for v in values]
        return dict(zip(keys, values))

    def read_data(self):
        # Memory-map the 10-byte events instead of reading them into a single
        # anonymous buffer, so that under memory pressure the OS can drop
        # clean pages and re-fault them from disk instead of paging out to swap.
        # raw_data = np.memmap(self.filename, dtype=self.data_dtype, mode='r', offset=self.data_offset)
        raw_data = np.fromfile(self.filename, dtype=self.data_dtype, count=-1, offset=self.data_offset)

        self.x = raw_data['x'].copy()
        self.y = raw_data['y'].copy()
        self.ts = raw_data['timestamp'].copy()


class OrdelaSANSEvents(object):
    """
    Reader/decoder for the older 30m SANS Ordela/VAX event-mode (.hst) files,
    per the bit-packed 32-bit event word layout in
    NCNR_User_Procedures/Reduction/SANS_Nexus/EventModeProcessing_OrdelaVAX_N.ipf
    (GetBitsFromEvents/DecodeEvents_New).

    Unlike VSANSEvents/SANSEvents, this format has no header: the file is a
    raw stream of little-endian 32-bit event words, each immediately followed
    by a 0xFFFFFFFF filler word (see LoadEventAsHex/RemoveFFF in the .ipf).
    """

    TIMESTAMP_RESOLUTION = 1e-7  # seconds per tick (ipf: rescaledTime *= 1e-7)

    def __init__(self, filename, remove_bad_events=True):
        self.filename = filename
        self.read_data(remove_bad_events=remove_bad_events)

    def read_data(self, remove_bad_events=True):
        raw = np.fromfile(self.filename, dtype='<u4')
        words = raw[0::2]  # drop the interleaved 0xFFFFFFFF filler words
        self._decode_words(words, remove_bad_events)

    def _decode_words(self, words, remove_bad_events, dims=None):
        # Trim leading words up to the first ATXYM (type==2) event, so that
        # time_msw/n_roll reconstruction starts from a known-good anchor
        # point, instead of from whatever state a partial event left behind
        # (CleanUpBeginning() in the .ipf).
        types = (words >> 30) & 0x3
        anchors = np.flatnonzero(types == ATXYM)
        if anchors.size == 0:
            raise ValueError(f"no ATXYM (type==2) anchor event found in {self.filename}")
        words = words[anchors[0]:]

        if dims is None:
            self.x, self.y, self.ts = decode_ordela_events(words, remove_bad_events)
        else:
            max_x, max_y = dims
            self.x, self.y, self.ts = decode_ordela_events(words, remove_bad_events, max_x, max_y)


class OrdelaSANSEventsASCII(OrdelaSANSEvents):
    """
    Reader/decoder for the older ASCII-text variant of the 30m SANS
    Ordela/VAX event-mode files, per LoadEvents_OLD()/DecodeEvents() in
    NCNR_User_Procedures/Reduction/SANS/EventModeProcessing.ipf.

    Same packed 32-bit event words as OrdelaSANSEvents (see
    decode_ordela_events for the bit layout), but stored as one hexadecimal
    number per text line instead of raw binary.

    Unlike the binary reader, events with x or y outside the 128x128
    detector are dropped rather than wrapped -- see decode_ordela_events for
    why an out-of-range low byte otherwise wraps around to a bogus x.
    """

    DIMS = (128, 128)

    def read_data(self, remove_bad_events=True):
        buffer = np.fromfile(self.filename, dtype=np.uint8)
        words = parse_ascii_hex_words(buffer)
        self._decode_words(words, remove_bad_events, dims=self.DIMS)


def events_from_file(filename):
    """
    Inspect the leading bytes of an event file and construct the matching
    reader: VSANSEvents (magic number 'VSANS'), SANSEvents (magic number
    'NAS'), or OrdelaSANSEvents (no header/magic number at all, so it's the
    fallback when neither magic number matches).
    """
    with open(filename, 'rb') as f:
        magic = f.read(5)

    if magic == b'VSANS':
        return VSANSEvents(filename)
    elif magic[:3] == b'NAS':
        return SANSEvents(filename)
    else:
        return OrdelaSANSEvents(filename)


def demo():
    import logging
    from time import perf_counter
    from matplotlib import pyplot as plt

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('event_processing.rebinning_api.binning').setLevel(level=logging.DEBUG)
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
    positions = [0,1] # or [0,1]
    for position in positions:
        filename = f"{cache}/{run}_{position}.hst"
        load_start_time = perf_counter()
        ev = VSANSEvents(filename)
        load_end_time = perf_counter()
        logging.info(f"loading {filename} took {load_end_time - load_start_time:.3f} seconds")

        rebin_start_time = perf_counter()
        detectors, edges = ev.rebin(100)
        rebin_end_time = perf_counter()
        logging.info(f"rebinning took {rebin_end_time - rebin_start_time:.3f} seconds")
        events[position] = ev
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