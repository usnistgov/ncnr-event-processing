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
    return "areaDetector"

def to_detector_indices(pixel_ids: np.ndarray):
    y = pixel_ids >> 16
    x = pixel_ids & 0xFFFF
    return x, y

def cleanup(entry, raw_events: "EventsManager", datapath=""):
    make_table = False
    cycle = "*"
    proposal = entry["DAS_logs/experiment/proposalId"][0]
    filename = entry["DAS_logs/trajectoryData/fileName"][0]
    instrument = entry["instrument/name"][0]

    start = raw_events.start
    wavelength = entry["instrument/monochromator/wavelength"][0]
    wavelength_spread = entry["instrument/monochromator/wavelength_error"][0]
    #print(f"{wavelength=} {wavelength_spread=}")
    detector_partitions = raw_events.get_detectors()


    events = {}
    if make_table:
        print(f"    # Table data extracted from {datapath}")
        print(f"    # det  yrange   events =? integrated")
    for k, (name, values) in enumerate(detector_partitions.items()):
        partition = get_partition(name)
        detector_name = partition_to_detector(partition)
        nxdetector = entry.get(f"instrument/{detector_name}", None)
        if nxdetector is None:
            logging.warning(f"Missing {entry.name}/instrument/{detector_name} in {datapath}")
            continue
        DAS = entry[nxdetector["data"].attrs['target']].parent
        dims = tuple(DAS['dimension'][()])

        columns = list(zip(*values))
        times, pixel_ids = np.asarray(columns[0]), np.asarray(columns[1])

        x, y = to_detector_indices(pixel_ids)
        
        base_distance = entry["instrument/detector/distance"] # cm
        extra_distance = extra_detector_distance(x, y)
        distance = base_distance + extra_distance

        time_correction = travel_time(distance, wavelength)

        if not ((x>=0).all() and (x<dims[0]).all() and (y>=0).all() and (y<dims[1]).all()):
            raise RuntimeError(f"Bad pixel id in {datapath}: x = {x.min()}:{x.max()} y = {y.min()}:{y.max()}")
    
        times -= start + time_correction
        events[name] = dict(dims=dims, ts=times, x=x, y=y)

    monitors = raw_events._fields.get("monitors", [])
    if monitors:
        events['monitor'] = dict(dims=(1,1), ts=np.asarray(monitors, dtype='int64'), x=0, y=0)
    return dict(detectors=events)
