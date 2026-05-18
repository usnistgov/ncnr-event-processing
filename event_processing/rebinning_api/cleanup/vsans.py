import numpy as np
import logging
import typing

from .util import travel_time, get_partition, neutron_velocity
if typing.TYPE_CHECKING:
    from ..event_capture import EventsManager

def extra_detector_distance(x: int | np.ndarray, y: int | np.ndarray):
    # TODO: add actual pixel-by-pixel additional distance?
    return 0.0

def partition_to_detector(partition_name: str):
    return "detector"

def to_detector_indices(pixel_ids: np.ndarray):
    y = pixel_ids >> 16
    x = pixel_ids & 0xFFFF
    return x, y

# TODO: make cleanup() function based on _cleanup_vsans
def cleanup(entry, raw_events: "EventsManager", datapath=""):
    raise NotImplementedError("Need to implement cleanup() for VSANS")

def _cleanup_vsans(entry, raw_events, datapath=""):
    make_table = False
    # Table data extracted from sans72110.nxs.ngv
    # det  yrange   events =? integrated
    # 0:FR   0:47     9012 =? 9013    NO!!!
    # 1:FT  48:95     7834 =? 7834    yes
    # 2:FB  96:143    6304 =? 6304    yes
    # 3:FL 144:191    9682 =? 9683    NO!!!
    # 4:MB  96:143    5137 =? 5137    yes
    # 5:MR   0:47    10143 =? 10143   yes
    # 6:ML 144:191   10328 =? 6776    NO!!!
    # 7:MT  48:95     6507 =? 6508    NO!!!

    # TODO: caller has datapath
    cycle = "*"
    proposal = entry["DAS_logs/experiment/proposalId"][0]
    filename = entry["DAS_logs/trajectoryData/fileName"][0]
    datapath = f"vsans/{cycle}/{proposal}/data/{filename}.nxs.ngv" if not datapath else datapath
    # TODO: need to associated redpanda detector number with nexus detector field
    start = raw_events.start
    wavelength = entry["instrument/beam/monochromator/wavelength"][0]
    wavelength_spread = entry["instrument/beam/monochromator/wavelength_spread"][0]
    #print(f"{wavelength=} {wavelength_spread=}")
    detectors = list("FR FT FB FL MB MR ML MT R".split())
    result = {}
    events = {}
    if make_table:
        print(f"    # Table data extracted from {datapath}")
        print(f"    # det  yrange   events =? integrated")
    for k, name in enumerate(detectors):
        nxdetector = entry.get(f"instrument/detector_{name}", None)
        if nxdetector is None:
            logging.warn(f"Missing {entry.name}/instrument/detector_{name} in {datapath}")
            continue
        distance = nxdetector["distance"][0]
        travel_time = int(1e8 * distance / neutron_velocity(wavelength)) # cm / (m/s) * 1e8 = ns
        #print(f"detector_{name}/distance: {distance} travel time {int(travel_time/1e6)} ms")
        DAS = entry[nxdetector["data"].attrs['target']].parent
        dims = tuple(DAS['dimension'][()])
        #print(f"detector_{name}->{DAS.name} {dims=}")
        key = f"detector_{k}"
        if key in raw_events._fields:
            columns = list(zip(*raw_events._fields[key]))
            times, pixels = np.asarray(columns[0]), np.asarray(columns[1])
        else:
            times, pixels = np.zeros(0, dtype='int64'), np.zeros(0, dtype='int64')
        y, x = pixels >> 16, pixels & 0xFFFF
        if make_table:
            num_events = len(pixels)
            counts = nxdetector["integrated_count"][0]
            match = "yes" if counts == num_events else "NO!!!"
            print(f"    # {k}:{name} {y.min():3d}:{y.max():<3d} {num_events:7d} =? {counts:<7d} {match}")
            #print("  x", x)
            #print("  y", y)
        if name == "R":
            pass
        elif name[1] == "R": # offset=0, fliplr
            x, y = x, 47-y
        elif name[1] == "L": # offset=144, flipud
            x, y = 127-x, y-144
        elif name[1] == "T": # swapaxes offset=48
            x, y = y-48, x
        elif name[1] == "B": # swapaxes offset=96, fliplr, flipud
            x, y = 143-y, 127-x
        else:
            raise UnreachableCode
        #print(f"{k}:{name} {dims=} y:{y.min()}-{y.max():<3} x:{x.min()}-{x.max():<3}")
        if not ((x>=0).all() and (x<dims[1]).all() and (y>=0).all() and (y<dims[0]).all()):
            raise RuntimeError(f"Bad pixel id in {datapath}")
        #print(f"times: {times.min()}:{times.max()} relative to {start}")
        #print(f"subtracting {start} from {times[0]} = {times[0]-start}")
        times -= start + travel_time
        #print(f"times: {times.min()/1e9:.3f}:{times.max()/1e9:.3f} relative to {int(start//1e9)}")
        # TODO: correct times for time of flight from wavelength and distance
        events[f"detector_{name}"] = dict(dims=dims, ts=times, x=x, y=y)
    # Treat the monitor as a detector named "monitor" so that we don't need
    # special handling during rebinning.
    monitors = raw_events._fields.get("monitors", [])
    if monitors:
        events['monitor'] = dict(dims=(1,1), ts=np.asarray(monitors, dtype='int64'), x=0, y=0)
    return dict(detectors=events)
