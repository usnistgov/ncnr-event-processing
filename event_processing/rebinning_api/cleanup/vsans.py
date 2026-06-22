import numpy as np
import logging
import typing

from .util import travel_time, get_partition, neutron_velocity
if typing.TYPE_CHECKING:
    from ..event_capture import EventsManager, CleanedEvents


logger = logging.getLogger(__name__)

def extra_detector_distance(x: int | np.ndarray, y: int | np.ndarray):
    # TODO: add actual pixel-by-pixel additional distance?
    return 0.0

def to_detector_indices(pixel_ids: np.ndarray):
    y = pixel_ids >> 16
    x = pixel_ids & 0xFFFF
    return x, y

import numpy as np


def calculate_pixel_distances(nxdetector, detector_name: str, dims):
    """
    Calculates the exact sample-to-detector distance for each pixel.
    
    Returns:
        pixel_distances (np.ndarray): A 2D array of distances in **centimeters (cm)** with shape (dim_x, dim_y).
    """
    dimX, dimY = dims[0], dims[1]
    
    # Base Z distance from sample to the center of the detector plane (in cm)
    z = nxdetector['distance'][0]
    if 'setback' in nxdetector:
        z += nxdetector['setback'][0] # setback is also in cm

    if detector_name.endswith("_B"):
        # Back detector special handling
        # cal_x and cal_y are already stored in cm
        x_pixel_size = nxdetector['cal_x'][0]
        y_pixel_size = nxdetector['cal_y'][0]
        
        # Beam center is calculated in cm
        beam_center_x = x_pixel_size * nxdetector['beam_center_x'][0]
        beam_center_y = y_pixel_size * nxdetector['beam_center_y'][0]
        
        realDistX = 0.5 * x_pixel_size # cm
        realDistY = 0.5 * y_pixel_size # cm
    else:
        # Front/Middle detectors (L, R, T, B panels)
        orientation = nxdetector['tube_orientation'][0]
        if isinstance(orientation, bytes):
            orientation = orientation.decode('utf-8')
        orientation = orientation.upper()
        
        # spatial_calibration coefficients (coeffs) are in millimeters (mm)
        coeffs = nxdetector['spatial_calibration'][()]
        
        # beam_center_x and beam_center_y are stored natively in cm
        beam_center_x = nxdetector['beam_center_x'][0]
        beam_center_y = nxdetector['beam_center_y'][0]
        
        # panel_gap is in mm, so we divide by 10.0 to convert to cm
        panel_gap = nxdetector['panel_gap'][0] / 10.0  
        
        lateral_offset = 0.0
        vertical_offset = 0.0
        
        if orientation == "VERTICAL":
            # x_pixel_size is in mm, converted to cm
            x_pixel_size = nxdetector['x_pixel_size'][0] / 10.0 
            # coeffs[1][0] acts as the y pixel size in mm, converted to cm
            y_pixel_size = coeffs[1][0] / 10.0 
            # lateral_offset is natively stored in cm (no conversion needed)
            lateral_offset = nxdetector['lateral_offset'][0] 
        else:
            # coeffs[1][0] acts as the x pixel size in mm, converted to cm
            x_pixel_size = coeffs[1][0] / 10.0
            # y_pixel_size is in mm, converted to cm
            y_pixel_size = nxdetector['y_pixel_size'][0] / 10.0 
            # vertical_offset is natively stored in cm (no conversion needed)
            vertical_offset = nxdetector['vertical_offset'][0] 
            
        pos_key = detector_name[-1]
        
        # Calculate real-space offsets based on panel position (all output variables in cm)
        # coeffs[0][0] is the spatial calibration offset in mm, so divide by 10.0 to get cm
        if pos_key == 'T':
            realDistX = coeffs[0][0] / 10.0 
            realDistY = 0.5 * y_pixel_size + vertical_offset + panel_gap / 2.0
        elif pos_key == 'B':
            realDistX = coeffs[0][0] / 10.0
            realDistY = vertical_offset - (dimY - 0.5) * y_pixel_size - panel_gap / 2.0
        elif pos_key == 'L':
            realDistX = lateral_offset - (dimX - 0.5) * x_pixel_size - panel_gap / 2.0
            realDistY = coeffs[0][0] / 10.0
        elif pos_key == 'R':
            realDistX = x_pixel_size * 0.5 + lateral_offset + panel_gap / 2.0
            realDistY = coeffs[0][0] / 10.0
        else:
            raise ValueError(f"Unknown position key {pos_key} for {detector_shortname}")

    # Relocate distance from the beam center (in cm)
    x0_pos = realDistX - beam_center_x
    y0_pos = realDistY - beam_center_y

    # Generate full 2D panel grid arrays
    X_grid, Y_grid = np.indices((dimX, dimY))
    
    # X and Y absolute coordinate arrays for every pixel (in cm)
    X = X_grid * x_pixel_size + x0_pos
    Y = Y_grid * y_pixel_size + y0_pos
    
    # Calculate total 3D hypotenuse distance for each pixel: sqrt(X^2 + Y^2 + Z^2) (in cm)
    pixel_distances = np.sqrt(X**2 + Y**2 + z**2)
    
    return pixel_distances

def get_detector_ending(pixel_id):
    """
    Determines the detector_name ending ('R', 'T', 'B', 'L')
    from a single encoded 'pixel_id' integer.
    """
    # Extract orig_x (the higher 16 bits)
    orig_x = pixel_id >> 16

    # Divide by 48 to find the offset chunk
    chunk_index = orig_x // 48

    # Map the chunk index to the corresponding suffix
    suffix_map = {
        0: "R",  # orig_x between 0 and 47 (offset 0)
        1: "T",  # orig_x between 48 and 95 (offset 48)
        2: "B",  # orig_x between 96 and 143 (offset 96)
        3: "L"   # orig_x between 144 and 191 (offset 144)
    }

    suffix = suffix_map.get(chunk_index)
    if suffix is None:
        raise ValueError(f"Invalid x pixel value {orig_x}, should be 0 < x < 191")
    return suffix

ALL_DETECTORS = ["FR", "FT", "FB", "FL", "MR", "MT", "MB", "ML", "B"]


def cleanup(entry, raw_events: "EventsManager", datapath=""):
    make_table = False

    # TODO: caller has datapath
    cycle = "*"
    proposal = entry["DAS_logs/experiment/proposalId"][0]
    filename = entry["DAS_logs/trajectoryData/fileName"][0]
    datapath = f"vsans/{cycle}/{proposal}/data/{filename}.nxs.ngv" if not datapath else datapath
    # TODO: need to associated redpanda detector number with nexus detector field
    start = raw_events.start[0]
    wavelength = entry["instrument/beam/monochromator/wavelength"][0]
    wavelength_spread = entry["instrument/beam/monochromator/wavelength_spread"][0]
    detector_partitions = raw_events.get_detectors()

    #print(f"{wavelength=} {wavelength_spread=}")
    # detectors = list("FR FT FB FL MB MR ML MT R".split())

    events = raw_events._fields.copy()
    detectors = events.setdefault('detectors', {})
    if make_table:
        print(f"    # Table data extracted from {datapath}")
        print(f"    # det  yrange   events =? integrated")
    for k, (name, event_pairs) in enumerate(detector_partitions.items()):
        list_of_timestamp_arrays: list[np.ndarray] = event_pairs["timestamp"]
        list_of_pixel_id_arrays: list[np.ndarray] = event_pairs["value"]
        # Concatenate the arrays into single arrays
        times = np.concatenate(list_of_timestamp_arrays)
        pixel_ids = np.concatenate(list_of_pixel_id_arrays)

        # Determine the detector ending based on the first pixel
        if k == 8:
            # last partition is the back detector (detector_B)
            detector_ending = "B"
            detector_name = "detector_B"
        else:
            detector_ending = get_detector_ending(pixel_ids[0])
            detector_group = "F" if k < 4 else "M"
            detector_name = f"detector_{detector_group}{detector_ending}"

        nxdetector = entry.get(f"instrument/{detector_name}", None)
        if nxdetector is None:
            logger.warning(f"Missing {entry.name}/instrument/{detector_name} in {datapath}")
            continue

        dataset = nxdetector.get(f"data", None)
        if dataset is None:
            logger.warning(f"Missing {entry.name}/instrument/{detector_name}/data in {datapath}")
            continue
        DAS = entry[nxdetector["data"].attrs['target']].parent
        dims = tuple(DAS['dimension'][()])

        distance = nxdetector["distance"][0] # cm
        distance_table = calculate_pixel_distances(nxdetector, detector_name, dims)
        
        logger.debug(f"detector {detector_name} distance: {distance}")
        logger.debug(f"distance table: min={distance_table.min()} max={distance_table.max()}")

        orig_x, orig_y = pixel_ids >> 16, pixel_ids & 0xFFFF
        # x, y = pixels >> 16, pixels & 0xFFFF
        if make_table:
            num_events = len(pixel_ids)
            counts = nxdetector["integrated_count"][0]
            match = "yes" if counts == num_events else "NO!!!"
            print(f"    # {k}:{name} {x.min():3d}:{x.max():<3d} {num_events:7d} =? {counts:<7d} {match}")
            #print("  x", x)
            #print("  y", y)
        if detector_name == "detector_B":
            y, x = orig_y, orig_x
        elif detector_name[-1] == "R": # offset=0, flipud
            y, x = 127-orig_y, orig_x
        elif detector_name[-1] == "L": # offset=144, fliplr
            y, x = orig_y, 191-orig_x
        elif detector_name[-1] == "T": # swapaxes offset=48
            y, x = orig_x-48, orig_y
        elif detector_name[-1] == "B": # swapaxes offset=96, fliplr, flipud
            y, x = 143-orig_x, 127-orig_y
        else:
            raise ValueError(f"Unknown detector {name}, should be in FR FT FB FL MB MR ML MT R")
        #print(f"{k}:{name} {dims=} y:{y.min()}-{y.max():<3} x:{x.min()}-{x.max():<3}")
        if not ((y>=0).all() and (y<dims[1]).all() and (x>=0).all() and (x<dims[0]).all()):
            # find bad pixels:
            bad_x = np.where((x < 0) | (x >= dims[0]))[0]
            bad_y = np.where((y < 0) | (y >= dims[1]))[0]
            bad = np.union1d(bad_x, bad_y)
            logging.error(f"Bad pixels in {detector_name}: x={x[bad]}, orig_x={orig_x[bad]}, y={y[bad]}, orig_y={orig_y[bad]}, pixel_ids={pixels[bad]}")
            raise RuntimeError(f"Bad pixel id in {datapath} for detector {detector_name}")
        #print(f"times: {times.min()}:{times.max()} relative to {start}")
        #print(f"subtracting {start} from {times[0]} = {times[0]-start}")

        pixel_dists = distance_table[x, y]
        
        # 3. Calculate corrected times per event using vectorization
        time_correction = travel_time(pixel_dists, wavelength)
        time_correction_sigma = time_correction * wavelength_spread / wavelength

        times -= start + time_correction.astype(int)
        ts_sigma = time_correction_sigma.astype(int)
        result = dict(dims=dims, ts=times, ts_sigma=ts_sigma, x=x, y=y)
        detectors[detector_name] = result

    # backfill with zeros any detectors not found:
    for detector_shortname in ALL_DETECTORS:
        detector_name = f"detector_{detector_shortname}"
        if detector_name not in detectors:
            logging.debug(f"Missing detector {detector_name} in events: setting to zeros array")
            nxdetector = entry.get(f"instrument/{detector_name}", None)
            if nxdetector is None:
                logger.warning(f"Missing {entry.name}/instrument/{detector_name} in {datapath}")
                continue
            data = nxdetector.get(f"data", None)
            if data is None:
                logger.warning(f"Missing {entry.name}/instrument/{detector_name}/data in {datapath}")
                continue
            DAS = entry[data.attrs['target']].parent
            dims = tuple(DAS['dimension'][()])

            detectors[detector_name] = dict(dims=dims, ts=np.zeros(0, dtype='int64'), ts_sigma=np.zeros(0, dtype='int64'), x=np.zeros(0, dtype='int64'), y=np.zeros(0, dtype='int64'))

    # Treat the monitor as a detector named "monitor" so that we don't need
    # special handling during rebinning.
    monitors = raw_events._fields.get("monitors", [])
    if monitors:
        detectors['monitor'] = dict(dims=(1,1), ts=np.asarray(monitors, dtype='int64'), x=0, y=0)

    raw_events._cleaned_fields = events
    # print(f"cleaned_events: {events}")
    return events

""" 
HISTORICAL: This was the original _cleanup code for the old hist files
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
"""