from typing import Dict, Literal, Optional, Sequence, Tuple, Union
from enum import StrEnum

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel
from numpy import ndarray as vector, ndarray as array


class NumpyArray(BaseModel):
    data: bytes
    dtype: str = "float32"
    shape: Sequence[int]

    @classmethod
    def from_ndarray(cls, arr):
        dtype = arr.dtype
        shape = arr.shape
        return cls(data=arr.tostring(), dtype=dtype.str, shape=shape)

    def to_ndarray(self):
        return np.fromstring(self.data, dtype=np.dtype(self.dtype)).reshape(self.shape)

class RebinnedData(BaseModel):
    time_bins: NumpyArray
    detectors: Dict[str, NumpyArray]
    devices: Dict[str, NumpyArray]

class VSANS(BaseModel):
    name: Literal["VSANS"] = "VSANS"
    front_detector_distance: float = 100
    middle_detector_distance: float = 200
    back_detector_distance: float = 500

class RebinUniformWidth(BaseModel):
    delta_t: float

class RebinUniformCount(BaseModel):
    num_bins: int

class InstrumentName(StrEnum):
    VSANS = "vsans"
    CANDOR = "candor"
    MACS = "macs"

@dataclass
class TimeMask:
    # TODO: do we want NaN for empty buckets?
    #: For edges [[start, end], ..., [start, end]] include events if they
    #: appear in one of the intervals. The corresponding buckets for counts,
    #: times, monitors, logs, etc. should be zero.
    edges: array # n x 2 array of bin edges

@dataclass
class TimeBins:
    """
    Binning requires bin edges for the individual bins, as well as a mask
    indicating which parts of the time series should be included. Use this
    also for the "no bin" case where all events are gathered into a single
    time slice.

    The edges are determined by the caller, based on the total duration of the
    measurement and the inteval between T0 pulses (for strobed measurements)
    or the range of values on a device (when binning by device value). This
    leaves the user interface free to design whatever binning schemes are
    desired, including linear, log, fibonacci, or even the user setting and
    dragging bin edges on a graph.

    Masked times (e.g., during pause, or when the sample is bad) are not
    required for simple time bins (you can accomplish the same by setting
    the bin edges to gather bad data into a single point), but including it
    makes the interface more consistent, and allows paused counts to  for a part of
    the time bin.
    when binning strobed measurements and strobing by value. For consistency
    of interface they are included
    """
    #: Bin edges (time bins)
    edges: vector
    #: Mask showing included time windows within the measurement, or None for
    #: the whole interval.
    mask: TimeMask | None
    #: Name of the binning class
    mode: str = "time"

@dataclass
class StrobeBins:
    #: Bin edges (time bins)
    edges: vector
    #: Mask showing included time windows within the measurement, or None for
    #: the whole interval.
    mask: TimeMask | None
    #: True if odd numbered triggers should be binned separately from
    #: even numbered triggers. Even/odd is preserved even through masking.
    hysterisis: bool
    #: If provided, use alternate trigger values for T0
    trigger_override: vector|None
    #: Name of the binning class
    mode: str = "strobe"

@dataclass
class DeviceBins:
    #: Device being binned (motor or environment log)
    device: str
    #: Bin edges in device coordinates
    edges: vector
    mask: TimeMask | None
    #: True if rising values binned separately from falling values
    hysterisis: bool
    #: Name of the binning class
    mode: str = "device"

# TODO: maybe separate calls for the different binning modes
Bins = TimeBins | StrobeBins | DeviceBins

# TODO: Are we going to play at ToF for inelastic measurements?

@dataclass
class Sweep:
    """
    Device being swept plus start and end values. Sweep data is returned as
    part of the metadata reply.
    """
    #: device name
    device: str
    #: starting value
    start: float
    #: stopping value
    stop: float
    #: number of cycles (or zero if indeterminate)
    cycles: int

@dataclass
class DeviceLog:
    mean: float
    std: float
    min: float
    max: float
    #: time relative to the start of the point (may exceed measurement range)
    time: vector
    #: values at the polled times
    value: vector
    # TODO: do we want the interpolated values as well?

@dataclass
class MetadataRequest:

    #: Name of the nexus file.
    filename: str
    #: For scan measurements, which point number in the scan
    point: int = 0

@dataclass
class MetadataReply:
    """
    Metadata available in the NeXus file.
    """
    request: MetadataRequest
    #: Number of points in the scan
    numpoints: int
    # TODO: python datetime object? ISO8601? unix epoch? epics epoch? windows epoch?
    # TODO: Do we need file time separate from point time?
    #: Start time for the measurement of the individual point
    timestamp: datetime
    #: Total measurement time
    duration: float
    #: For strobed measurements, the expected interval between triggers.
    trigger_interval: float
    #: Active detectors
    detectors: list[str]
    #: Environment logs, with values for min, max, mean and std across the measurement
    logs: dict[str, DeviceLog]
    #: Measurement mode: relaxation, strobe, sweep
    mode: str
    #: Sweep device, start, stop, #cycles
    sweep: Sweep|None

@dataclass
class TriggerRequest:
    """
    Trigger data is not available in the nexus file. We need to pull it from
    kafka if not already cached, so this data is not available in the metadata
    request. The {instrument}_{timing} topic should be relatively empty so the
    retrieval shouldn't take too much time.
    """
    filename: str
    point: int = 0

@dataclass
class TriggerReply:
    request: TriggerRequest
    #: Trigger times
    triggers: vector
    # TODO: return pause/resume and fast shutter info as well
    # This information can be used to set the default mask for excluded counts.
    # By adjusting the mask the user can dip into these regions to see what
    # is going on.

@dataclass
class SummaryRequest:
    #: filename
    filename: str
    #: Point number in the file
    point: int
    #: Binning specification
    bins: Bins

@dataclass
class SummaryReply:
    request: SummaryRequest
    #: times for the time bins
    times: vector
    #: counts of roi for each bin
    counts: vector
    #: monitor counts for each bin
    monitor: vector
    # TODO: are device (time, value) logs meaninful for strobed and by-value binning?
    #: Device values (temperature, pressure, motor) with statistics over the time bin.
    devices: dict[str, DeviceLog]
    #: offsets for the trigger signals
    strobe: vector

# Note: we have an implicit assumption that the server is caching request
# info so that for example, subsequent slices can be quickly retrieved given
# the hash of (filename, point, bins)
@dataclass
class FrameRequest:
    #: filename
    filename: str
    #: Point number in the file
    point: int
    #: Binning specification
    bins: Bins
    #: Slice index in histogram
    index: int

@dataclass
class FrameReply:
    request: FrameRequest
    #: requested slice
    frame: int
    #: time offset of the slice (relative to T0 for strobed, N/A for value binned)
    time: float
    #: total time accumulated for slice (including all events for strobed and byvalue)
    duration: float
    #: detector data for the individual detectors at that slice
    detectors: dict[str, array]
    # TODO: are device (time, value) logs meaninful for strobed and by-value binning?
    #: Device values with statistics over the time bin.
    devices: dict[str, DeviceLog]

@dataclass
class NexusRequest:
    #: filename
    filename: str
    #: Binning specification
    bins: Bins

@dataclass
class NexusReply:
    request: NexusRequest
    data: bytes

@dataclass
class LogRequest:
    """
    Return log data (e.g., temperature) over a fixed time interval.
    """
    instrument: str
    device: str
    start: datetime
    end: datetime

@dataclass
class LogReply:
    """
    Return log data (e.g., temperature) over a fixed time interval. This returns
    the value and the time that it was recorded with no interpolation.
    """
    request: LogRequest
    times: vector
    values: vector
