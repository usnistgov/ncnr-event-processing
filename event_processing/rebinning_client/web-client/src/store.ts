import { ref } from 'vue';
import { NumpyArray } from './numpy_array';
import { Meta } from 'quasar';

export const ncnr_metadata_api = "https://ncnr.nist.gov/ncnrdata/metadata/api/v1";
// export const rebinning_api = "http://candorgpu.campus.nist.gov:8080";
export const rebinning_api = "http://localhost:8000";

export const selected_experiment = ref('');
export const selected_filename = ref('');
export const selected_path = ref('');
export const duration = ref(1);

export const datafile_search_state = ref({
    'rows': [],
    'pagination': {
        'rowsPerPage': 10,
        'descending': true,
        'sortBy': 'date',
        'page': 0,
        'rowsNumber': 0,
    },
    'selection': '',
    'selection_path': '',
});

export const binning_state = ref({
    'fetching_summary': false,
    'downloading_nexus': false,
    'duration': null,
    'num_bins': null,
    'bins': null,
    'metadata': null,
    'last_frame': null
});

export async function api_post(base_api: string, endpoint: string, data: object) {
    const response = await fetch(`${base_api}/${endpoint}`, {
        method: "POST", // *GET, POST, PUT, DELETE, etc.
        mode: "cors", // no-cors, *cors, same-origin
        cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
        // credentials: "same-origin", // include, *same-origin, omit
        headers: {
            "Content-Type": "application/json",
            // 'Content-Type': 'application/x-www-form-urlencoded',
        },
        redirect: "follow", // manual, *follow, error
        referrerPolicy: "no-referrer", // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
        body: JSON.stringify(data), // body data type must match "Content-Type" header
    });
    return response.json(); // parses JSON response into native JavaScript objects
}

export async function api_get(base_api: string, endpoint: string, data: object = {}) {
    const url = new URL(`${base_api}/${endpoint}`);
    url.search = new URLSearchParams(data).toString();
    const response = await fetch(url);
    return response.json();
}

export const all_instruments = ref([]);
api_get(ncnr_metadata_api, 'instruments').then((result) => all_instruments.value = result);

interface MetadataRequest {
    // Name of the nexus file.
    filename: string,
    // Path to the data directory relative to data root. If None, then search
    // for the file in the experiment metadata service.
    path?: string
    // Note: entry is specified as an integer rather than a string so that
    // caching works transparently.
    // Entry within the nexus file, if there are multiple. Similar to points
    // within an entry, this defaults to the first entry in the file.
    entry?: number // = 0
    // For scan measurements, which point number in the scan. [summary, frame, metadata, trigger]
    point?: number // = 0
}

interface MetadataReply {
  //
  //  Metadata available in the NeXus file.
  //
  measurement: MetadataRequest
  //: List of entries in the file
  entries: string[]
  //: Number of points in the selected entry
  numpoints: number
  // TODO: python datetime object? ISO8601? unix epoch? epics epoch? windows epoch?
  // TODO: Do we need file time separate from point time?
  //: Start time for the measurement of the individual point
  timestamp: string
  //: Total measurement time
  duration: number
  //: For strobed measurements, the expected interval between triggers.
  trigger_interval: number
  //: Active detectors
  detectors: string[]
  //: Environment logs, with values for min, max, mean and std across the measurement
  logs: { [key: string]: DeviceLog }
  //: Measurement mode: time, strobe, cycle, sweep, device
  event_mode: string
  //: Sweep device, start, stop, cycles
  sweep: Sweep | null
}

interface DeviceLog {
    mean: number,
    std: number,
    min: number,
    max: number,
    //: time relative to the start of the point (may exceed measurement range)
    time: NumpyArray,
    //: values at the polled times
    value: NumpyArray,
    // TODO: do we want the interpolated values as well?
}

interface Sweep {
    // """
    // Device being swept plus start and end values. Sweep data is returned as
    // part of the metadata reply.
    // """
    //: device name
    device: string
    //: starting value
    start: number
    //: stopping value
    stop: number
    //: number of cycles (or zero if indeterminate)
    cycles: number
}

interface TimeBins {
    // """
    // Binning requires bin edges for the individual bins, as well as a mask
    // indicating which parts of the time series should be included. Use this
    // also for the "no bin" case where all events are gathered into a single
    // time slice.

    // The edges are determined by the caller, based on the total duration of the
    // measurement and the inteval between T0 pulses (for strobed measurements)
    // or the range of values on a device (when binning by device value). This
    // leaves the user interface free to design whatever binning schemes are
    // desired, including linear, log, fibonacci, or even the user setting and
    // dragging bin edges on a graph.

    // Masked times (e.g., during pause, or when the sample is bad) are not
    // required for simple time bins (you can accomplish the same by setting
    // the bin edges to gather bad data into a single point), but including it
    // makes the interface more consistent, and allows paused counts to  for a part of
    // the time bin.
    // when binning strobed measurements and strobing by value. For consistency
    // of interface they are included
    // """
    // #: Bin edges (time bins)
    edges: NumpyArray
    // #: Mask showing included time windows within the measurement, or None for
    // #: the whole interval.
    mask: null
    // #: Name of the binning class
    mode: string // = 'time'
}

export const metadata = ref<MetadataReply>({duration: 1});

export async function get_metadata() {
  // do something
  // def get_metadata():
  //               filename = datafile_search_state.get("selection", None)
  //               path = datafile_search_state.get("selection_path", None)
  //               if not filename or not path:
  //                   ui.notify("file or path is undefined... can't get metadata")
  //                   binning_state['metadata'] = None
  //               metadata = binning_client.get_metadata(filename, path=path)
  //               binning_state['metadata'] = metadata
  //               time_bin_settings.duration = metadata.duration
  if (selected_filename.value && selected_path.value) {
    const request: MetadataRequest = {
      filename: selected_filename.value,
      path: selected_path.value,
      entry: 0,
      point: 0
    }
    const metadata_reply: MetadataReply = await api_post(rebinning_api, 'metadata', request);

    metadata.value = metadata_reply;
    console.log(metadata_reply);
  }
}
