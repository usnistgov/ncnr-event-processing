import numpy as np
import logging
import typing

from .util import travel_time, get_partition, FWHM_to_sigma
if typing.TYPE_CHECKING:
    from ..event_capture import EventsManager

def extra_detector_distance(x: int | np.ndarray, y: int | np.ndarray):
    # TODO: add actual pixel-by-pixel additional distance?
    return 0.0

def partition_to_detector(partition_name: str):
    return "detector"

def to_detector_indices(pixel_ids: np.ndarray):
    x = pixel_ids >> 16
    y = pixel_ids & 0xFFFF
    return x, y


def get_pixel_distances(entry):
    """
    Calculate the total Euclidean distance from the sample to each detector pixel
    for a flat, planar detector, reading dimensions dynamically.

    Args:
        entry: h5py.Group, the HDF5 entry for the measurement.

    Returns:
        np.ndarray: A 2D array (e.g., 128x128) containing the distance from the
                    sample to each pixel in cm.
    """

    def read_cm(path):
        field = entry[path]
        val = field[()]
        if isinstance(val, np.ndarray):
            val = val[0] if val.size > 1 else val.item()

        units = field.attrs.get('units', b'')
        if isinstance(units, bytes):
            units = units.decode('utf-8')

        if units == 'm':
            return val * 100.0
        elif units == 'mm':
            return val / 10.0
        return val

    # 1. Z distance: detector distance + sample position offset
    det_dis = read_cm("DAS_logs/detectorPosition/softPosition")
    sample_pos = read_cm("DAS_logs/geometry/samplePositionOffset")
    Z = det_dis + sample_pos

    # 2. Beam center in pixel coordinates
    x0 = entry["instrument/detector/beam_center_x"][()]
    y0 = entry["instrument/detector/beam_center_y"][()]
    if isinstance(x0, np.ndarray): x0 = x0.item()
    if isinstance(y0, np.ndarray): y0 = y0.item()

    # 3. Pixel sizes (converted to cm)
    sx = read_cm("instrument/detector/x_pixel_size")
    sy = read_cm("instrument/detector/y_pixel_size")

    # 4. Read the dimensions dynamically from the data array
    # Check standard NeXus locations for the detector data
    if "instrument/detector/data" in entry:
        data_shape = entry["instrument/detector/data"].shape
    elif "data/areaDetector" in entry:
        data_shape = entry["data/areaDetector"].shape
    else:
        # Fallback if standard paths are missing
        data_shape = (128, 128)

    # Extract the last two dimensions (e.g., if shape is (1, 128, 128) -> (128, 128))
    ny, nx = data_shape[-2:]

    # 5. Generate the grid (1-based indexing to match center coordinates)
    x_indices = np.arange(1, nx + 1)
    y_indices = np.arange(1, ny + 1)
    x, y = np.meshgrid(x_indices, y_indices, indexing='ij') 

    # 6. Direct linear calculation of physical X and Y coordinates (in cm)
    X = (x - x0) * sx
    Y = (y - y0) * sy

    # 7. Total distance is the hypotenuse of X, Y, and Z
    distances = np.sqrt(X**2 + Y**2 + Z**2)

    return distances

def _concat_or_reuse(arrays: list[np.ndarray]) -> np.ndarray:
    # np.concatenate always makes a copy, even for a single-element list.
    # For large (>1e8 event) detectors avoid that redundant full-size copy
    # in the common case where there is nothing to concatenate.
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)

def cleanup(entry, raw_events: "EventsManager", datapath=""):
    make_table = False
    cycle = "*"
    # proposal = entry["DAS_logs/experiment/proposalId"][0]
    # filename = entry["DAS_logs/trajectoryData/fileName"][0]
    # instrument = entry["instrument/name"][0]

    start = raw_events.start[0]
    wavelength = entry["instrument/monochromator/wavelength"][0]
    wavelength_spread_FWHM = entry["instrument/monochromator/wavelength_error"][0]
    #print(f"{wavelength=} {wavelength_spread=}")
    detector_partitions = raw_events.get_detectors()


    events = raw_events._fields.copy()
    detectors = events.setdefault('detectors', {})
    if make_table:
        print(f"    # Table data extracted from {datapath}")
        print(f"    # det  yrange   events =? integrated")
    for k, (name, event_pairs) in enumerate(detector_partitions.items()):
        partition = get_partition(name)
        detector_name = partition_to_detector(partition)
        nxdetector = entry.get(f"instrument/{detector_name}", None)
        if nxdetector is None:
            logging.warning(f"Missing {entry.name}/instrument/{detector_name} in {datapath}")
            continue
        DAS = entry[nxdetector["data"].attrs['target']].parent
        dims = tuple(DAS['dimension'][()])

        list_of_timestamp_arrays: list[np.ndarray] = event_pairs["timestamp"]
        # Concatenate the arrays into single arrays
        times = _concat_or_reuse(list_of_timestamp_arrays)
        if "x" in event_pairs:
            # Already-split x/y, e.g. from a replayed .hst file where
            # tubeID/pixel are stored as separate columns: use them directly
            # rather than packing into a pixel_id here only to unpack again
            # below.
            x = _concat_or_reuse(event_pairs["x"])
            y = _concat_or_reuse(event_pairs["y"])
            pixel_ids = None
        else:
            # Live-stream format: a single packed pixel_id per event
            # (tubeID << 16 | pixel), matching the kafka wire format.
            pixel_ids = _concat_or_reuse(event_pairs["value"])
            x, y = to_detector_indices(pixel_ids)

        distance_table = get_pixel_distances(entry)
        travel_time_table = np.floor(travel_time(distance_table, wavelength)).astype(np.int64)
        time_correction = travel_time_table[x, y]

        sigma_factor = wavelength_spread_FWHM * FWHM_to_sigma
        ts_sigma_stats = {
            "mean": time_correction.mean() * sigma_factor,
            "max": int(time_correction.max() * sigma_factor),
            "min": int(time_correction.min() * sigma_factor),
        }

        times -= time_correction
        times -= start

        index = np.argsort(times, kind="stable")
        times = times[index]
        x = x[index]
        y = y[index]

        if not ((x>=0).all() and (x<dims[0]).all() and (y>=0).all() and (y<dims[1]).all()):
            raise RuntimeError(f"Bad pixel id in {datapath}: x = {x.min()}:{x.max()} y = {y.min()}:{y.max()}")

        detectors[name] = dict(dims=dims, ts=times, ts_sigma_stats=ts_sigma_stats, x=x, y=y)

    monitors = raw_events._fields.get("monitors", [])
    if monitors:
        detectors['monitor'] = dict(dims=(1,1), ts=np.asarray(monitors, dtype='int64'), x=0, y=0)
    raw_events._cleaned_fields = events
    return events
