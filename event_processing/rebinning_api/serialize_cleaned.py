import typing

if typing.TYPE_CHECKING:
    import h5py
    from event_processing.rebinning_api.event_capture import EventsManager


def write_cleaned_events(group: "h5py.Group", cleaned_fields: dict, meta: dict | None = None):
    """
    Write a `_cleaned_fields`-shaped dict (detectors/monitor/TO_SYNC) into an
    already-open, writable HDF5 group.

    *meta* is an optional dict of scalar/array attrs (e.g. start/stop/arm/disarm)
    recorded alongside the events for provenance; it is not required for binning.
    """
    import numpy as np
    import hdf5plugin

    if meta:
        for key, value in meta.items():
            group.attrs[key] = value

    for key, data in cleaned_fields.items():
        if key == 'detectors':
            det_root = group.create_group(key)
            for det_key in data:
                det_group = det_root.create_group(det_key)
                det_data = data[det_key]
                for subkey in ['ts']:  # was ['ts', 'ts_sigma']
                    subval = np.array(det_data[subkey], dtype='int64') if np.isscalar(det_data[subkey]) else det_data[subkey]
                    chunk_size = min(125000, len(subval))
                    det_group.create_dataset(subkey, chunks=(chunk_size,), dtype='int64', data=subval, **hdf5plugin.Blosc(cname='lz4', shuffle=hdf5plugin.Blosc.BITSHUFFLE))
                for subkey in ['x', 'y']:
                    subval = np.array(det_data[subkey], dtype='int16') if np.isscalar(det_data[subkey]) else det_data[subkey]
                    chunk_size = min(125000, len(subval))
                    det_group.create_dataset(subkey, chunks=(chunk_size,), dtype='int16', data=subval, **hdf5plugin.Blosc(cname='lz4', shuffle=hdf5plugin.Blosc.BITSHUFFLE))
                det_group.attrs['dims'] = det_data['dims']
                if 'ts_sigma_stats' in det_data:
                    stats_group = det_group.create_group('ts_sigma_stats')
                    for stat_key, stat_val in det_data['ts_sigma_stats'].items():
                        # keys are: 'mean', 'std', 'min', 'max'
                        stats_group.create_dataset(stat_key, data=stat_val, shape=())

        elif key in ('monitor', 'TO_SYNC'):
            sub_group = group.create_group(key)
            if data:
                # Unpack list of (timestamp,) tuples
                ts = [d[0] for d in data]
                sub_group.create_dataset('timestamp', data=np.array(ts, dtype='int64'), scaleoffset=0, compression='gzip')
                sub_group.create_dataset('timestamp_sigma', shape=(0,), dtype='int64')
            else:
                sub_group.create_dataset('timestamp', shape=(0,), dtype='int64')
                sub_group.create_dataset('timestamp_sigma', shape=(0,), dtype='int64')


def read_cleaned_events(group: "h5py.Group") -> dict:
    """
    Read a `_cleaned_fields`-shaped dict (detectors/monitor/TO_SYNC) out of an
    already-open, readable HDF5 group written by `write_cleaned_events`.
    """
    import hdf5plugin  # noqa: F401  (registers the lz4/blosc filter needed to read compressed datasets)

    cleaned_fields: dict = {}

    for key in group.keys():
        sub_group = group[key]

        if key == 'detectors':
            cleaned_fields['detectors'] = {}
            for det_key in sub_group.keys():
                det_group = sub_group[det_key]
                det_data = {
                    "ts": det_group['ts'][:],
                    "x": det_group['x'][:],
                    "y": det_group['y'][:],
                    "dims": det_group.attrs['dims'],
                }
                if 'ts_sigma_stats' in det_group:
                    stats_group = det_group['ts_sigma_stats']
                    det_data['ts_sigma_stats'] = {
                        stat_key: stat_val[()]
                        for stat_key, stat_val in stats_group.items()
                    }
                cleaned_fields['detectors'][det_key] = det_data

        elif key in ('monitor', 'TO_SYNC'):
            timestamps = sub_group['timestamp'][:]
            cleaned_fields[key] = [(t,) for t in timestamps]

    return cleaned_fields


def events_to_hdf5(db: "EventsManager", output_hdf5_path: str):
    """
    Converts an EventsManager database to a standalone HDF5 file with the same structure.
    """
    import h5py

    meta = dict(start=db.start, stop=db.stop, arm=db.arm, disarm=db.disarm)
    with h5py.File(output_hdf5_path, 'w') as f:
        write_cleaned_events(f, db._cleaned_fields, meta=meta)


def events_from_hdf5(db: "EventsManager", path):
    """
    Populate the internal _cleaned_fields dictionary from a standalone HDF5 event file.
    """
    import h5py

    with h5py.File(path, 'r') as f:
        db.start = f.attrs.get('start', 0)
        db.stop = f.attrs.get('stop', 0)
        db.arm = f.attrs.get('arm', 0)
        db.disarm = f.attrs.get('disarm', 0)
        db._cleaned_fields = read_cleaned_events(f)
