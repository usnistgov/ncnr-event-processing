from functools import lru_cache
import importlib.resources as pkg_resources
import numpy as np
import logging
import typing

from .util import travel_time, get_partition, neutron_velocity

if typing.TYPE_CHECKING:
    from ..event_capture import EventsManager

DEBUG = False
NUM_DETECTORS = 54

@lru_cache
def load_distances() -> np.ndarray:
    """
    Load a text file containing distances from the package's source directory.
    Returns
    -------
    dict[str, np.ndarray]
        Dict with keys ["index", "mean_distance_m", "sigma_distance_m", "sigma_wavelength_A"]
    """
    data_path = pkg_resources.files(__package__) / "source" / "candor_distances.dat"
    with data_path.open("rb") as source_file:
        data_array = np.genfromtxt(
            source_file,
            names=True,
        )
    return data_array

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
    wavelength_spread = entry["DAS_logs/detectorTable/wavelengthSpreads"][:]
    detector_partitions = raw_events.get_detectors()

    result = {
        "dims": None,
        "x": [],
        "y": [],
        "ts": [],
        "ts_sigma": [],
    }

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

        # key = f"detector_{k}"
        # if key in raw_events._fields:
        #     columns = list(zip(*raw_events._fields[key]))
        #     times, pixel_ids = np.asarray(columns[0]), np.asarray(columns[1])
        # else:
        #     times, pixel_ids = np.zeros(0, dtype='int64'), np.zeros(0, dtype='int64')
        x, y = to_detector_indices(pixel_ids)
        wavelength = wavelength_calibration[x, y]
        wavelength_sigma = wavelength_spread[x, y]

        distance_array = load_distances()
        distance = distance_array["mean_distance_m"][x] * 100 # convert to cm
        distance_sigma = distance_array["sigma_distance_m"][x] * 100 # convert to cm

        time_correction = travel_time(distance, wavelength)
        time_correction_sigma = time_correction * np.sqrt((distance_sigma/distance)**2 + (wavelength_sigma/wavelength)**2)

        if not ((x>=0).all() and (x<dims[0]).all() and (y>=0).all() and (y<dims[1]).all()):
            raise RuntimeError(f"Bad pixel id in {datapath}: x = {x.min()}:{x.max()} y = {y.min()}:{y.max()}")

        # BBM 2026-05-20: don't make relative timestamps here - that is a later step
        times -= time_correction.astype(int)

        result["dims"] = dims
        result["x"].append(x)
        result["y"].append(y)
        result["ts"].append(times)
        result["ts_sigma"].append(time_correction_sigma.astype(int))

        # events[name] = dict(dims=dims, ts=times, ts_sigma=time_correction_sigma, x=x, y=y)

    # Treat the monitor as a detector named "monitor" so that we don't need
    # special handling during rebinning.
    monitors = raw_events._fields.get("monitors", [])
    if monitors:
        events['monitor'] = dict(dims=(1,1), ts=np.asarray(monitors, dtype='int64'), x=0, y=0)
    
    # combine events for all partitions (only one detector in nexus)
    if result["dims"] is not None:
        combined = dict(
            dims=result["dims"],
            ts=np.concatenate(result["ts"]) if result["ts"] else np.zeros(0, dtype='int64'),
            ts_sigma=np.concatenate(result["ts_sigma"]) if result["ts_sigma"] else np.zeros(0, dtype='int64'),
            x=np.concatenate(result["x"]) if result["x"] else np.zeros(0, dtype='int64'),
            y=np.concatenate(result["y"]) if result["y"] else np.zeros(0, dtype='int64'),
        )
        events["multiDetector"] = combined
    raw_events._cleaned_fields = events
    return events