import numpy as np
import logging
import typing

from .util import travel_time, get_partition, neutron_velocity
if typing.TYPE_CHECKING:
    from ..event_capture import EventsManager

DEBUG = True

DETECTOR_MEAN_DISTANCE = 400 # cm
DETECTOR_LENGTH = 100 # cm
NUM_DETECTORS = 54

def extra_detector_distance(x: int | np.ndarray, y: int | np.ndarray):
    """ 
    Returns a number between -0.5 * DETECTOR_LENGTH and 0.5 * DETECTOR_LENGTH
    (assumes that recorded detector distance is to the center of the detector)
    """
    return ((x / NUM_DETECTORS) - 0.5) * DETECTOR_LENGTH

def partition_to_detector(partition_name: str):
    return "PSD"

def to_detector_indices(pixel_ids: np.ndarray):
    raw_pixel = pixel_ids >> 16 # detector ID
    x = raw_pixel % NUM_DETECTORS
    y = raw_pixel // NUM_DETECTORS
    if DEBUG:
        p = pixel_ids & 0xFFFF # pixel within detector (only 1 on CANDOR, index == 0)
        assert(np.all(p == 0))

    return x, y

def cleanup(entry, raw_events: "EventsManager", datapath=""):
    make_table = False

    cycle = "*"
    proposal = entry["DAS_logs/experiment/proposalId"][0]
    filename = entry["DAS_logs/trajectoryData/fileName"][0]
    # TODO: need to associated redpanda detector number with nexus detector field
    start = raw_events.start
    wavelength_calibration = entry["DAS_logs/detectorTable/wavelengths"][:]
    wavelength_spread = entry["DAS_logs/detectorTable/wavelengthSpreads"][0]
    detector_partitions = raw_events.get_detectors()

    result = {}
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
        print(f"times: {times}")

        # key = f"detector_{k}"
        # if key in raw_events._fields:
        #     columns = list(zip(*raw_events._fields[key]))
        #     times, pixel_ids = np.asarray(columns[0]), np.asarray(columns[1])
        # else:
        #     times, pixel_ids = np.zeros(0, dtype='int64'), np.zeros(0, dtype='int64')
        x, y = to_detector_indices(pixel_ids)
        wavelength = np.take(wavelength_calibration, x)

        base_distance = DETECTOR_MEAN_DISTANCE
        extra_distance = extra_detector_distance(x, y)
        distance = base_distance + extra_distance

        time_correction = travel_time(distance, wavelength).astype(int)

        if not ((x>=0).all() and (x<dims[0]).all() and (y>=0).all() and (y<dims[1]).all()):
            raise RuntimeError(f"Bad pixel id in {datapath}: x = {x.min()}:{x.max()} y = {y.min()}:{y.max()}")
    
        times -= start + time_correction
        if DEBUG:
            print(f"times: {times}")
            print(f"wavelength: {wavelength}")
            print(f"distance: {distance}")
            print(f"extra_distance: {extra_distance}")
            print(f"time_correction: {time_correction}")
            print(f"x: {x}")
            print(f"y: {y}")
            print(f"dims: {dims}")

        events[name] = dict(dims=dims, ts=times, x=x, y=y)

    # Treat the monitor as a detector named "monitor" so that we don't need
    # special handling during rebinning.
    monitors = raw_events._fields.get("monitors", [])
    if monitors:
        events['monitor'] = dict(dims=(1,1), ts=np.asarray(monitors, dtype='int64'), x=0, y=0)
    return dict(detectors=events)