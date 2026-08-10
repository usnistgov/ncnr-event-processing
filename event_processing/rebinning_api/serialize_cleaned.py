import typing

if typing.TYPE_CHECKING:
    from event_processing.rebinning_api.event_capture import EventsManager


def events_to_hdf5(db: "EventsManager", output_hdf5_path: str):
    """
    Converts an EventsManager database to a HDF5 file with the same structure.
    """
    import h5py
    import hdf5plugin
    import numpy as np
    
    with h5py.File(output_hdf5_path, 'w') as f:
        # Save metadata attributes
        f.attrs['start'] = db.start
        f.attrs['stop'] = db.stop
        f.attrs['arm'] = db.arm
        f.attrs['disarm'] = db.disarm
        
        # Save cleaned (timestamp corrected, reshaped) events
        for key, data in db._cleaned_fields.items():
            if key == 'detectors':
                group = f.create_group(key)
                for det_key in data:
                    det_group = group.create_group(det_key)
                    det_data = data[det_key]
                    for subkey in ['ts']: # was ['ts', 'ts_sigma']
                        subval = np.array(det_data[subkey], dtype='int64') if np.isscalar(det_data[subkey]) else det_data[subkey]
                        chunk_size = min(125000, len(subval))
                        # det_group.create_dataset(subkey, chunks=(chunk_size,), dtype='int64', data=subval, shuffle=True, compression='gzip', compression_opts=4)
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
                            # create scalar dataset:
                            stats_group.create_dataset(stat_key, data=stat_val, shape=())

            elif key in ('monitor', 'TO_SYNC'):
                group = f.create_group(key)
                if data:
                    # Unpack list of (timestamp,) tuples
                    ts = [d[0] for d in data]
                    group.create_dataset('timestamp', data=np.array(ts, dtype='int64'), scaleoffset=0, compression='gzip')
                    group.create_dataset('timestamp_sigma', shape=(0,), dtype='int64')
                else:
                    group.create_dataset('timestamp', shape=(0,), dtype='int64')
                    group.create_dataset('timestamp_sigma', shape=(0,), dtype='int64')

def events_from_hdf5(db: "EventsManager", path):
    """
    Populate the internal _cleaned_fields dictionary from an HDF5 event file.
    """
    import h5py
    import hdf5plugin

    with h5py.File(path, 'r') as f:
        # Load metadata attributes
        db.start = f.attrs.get('start', 0)
        db.stop = f.attrs.get('stop', 0)
        db.arm = f.attrs.get('arm', 0)
        db.disarm = f.attrs.get('disarm', 0)
        
        # Ensure the _cleaned_fields dictionary is initialized
        if not hasattr(db, '_cleaned_fields'):
            db._cleaned_fields = {}
        
        # Load datasets
        for key in f.keys():
            group = f[key]
            
            if key == 'detectors':
                db._cleaned_fields['detectors'] = {}
                
                # Iterate through individual detectors saved inside the 'detectors' group
                for det_key in group.keys():
                    det_group = group[det_key]
                    
                    det_data = {
                        "ts": det_group['ts'][:],
                        "x": det_group['x'][:],
                        "y": det_group['y'][:],
                        "dims": det_group.attrs['dims'],
                    }
                    
                    # Restore ts_sigma_stats if it was saved
                    if 'ts_sigma_stats' in det_group:
                        stats_group = det_group['ts_sigma_stats']
                        det_data['ts_sigma_stats'] = {
                            stat_key: stat_val 
                            for stat_key, stat_val in stats_group.items()
                        }
                        
                    db._cleaned_fields['detectors'][det_key] = det_data
                    
            elif key in ('monitor', 'TO_SYNC'):
                # Reconstruct the original list of (timestamp,) tuples 
                timestamps = group['timestamp'][:]
                db._cleaned_fields[key] = [(t,) for t in timestamps]