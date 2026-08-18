import shutil
import threading
from pathlib import Path

import h5py

from . import data_cache
from . import serialize_cleaned

CACHE_ROOT = Path.cwd() / "cache"
EVENTS_FOLDER = (CACHE_ROOT / "events_cache").absolute()

# Guards writes to the on-disk events-cache files: concurrent requests must
# not open the same HDF5 file for writing at the same time.
_write_lock = threading.Lock()


def configure(cache_root):
    """Set the base cache directory; events-cache copies live in <cache_root>/events_cache."""
    global CACHE_ROOT, EVENTS_FOLDER
    CACHE_ROOT = Path(cache_root)
    EVENTS_FOLDER = (CACHE_ROOT / "events_cache").absolute()


def events_cache_path(filename: str) -> Path:
    return EVENTS_FOLDER / filename


def group_path(entry_name: str, point: int) -> str:
    return f"{entry_name}/events/point_{point}"


def ensure_events_file(measurement) -> Path:
    """
    Make sure a copy of the cached nexus file exists at events_cache_path,
    ready to have an "events" group added to it. Returns the path either way.
    """
    target = events_cache_path(measurement.filename)
    if not target.exists():
        source = data_cache.NEXUS_FOLDER / measurement.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return target


def load_cached_events(measurement, entry_name: str, path: Path | None = None) -> dict | None:
    """
    Load previously-persisted cleaned events for (measurement, entry_name), or
    None if not cached yet. *path* overrides the server's cache directory,
    for loading a user-provided events+nexus file directly.
    """
    target = path if path is not None else events_cache_path(measurement.filename)
    target = Path(target)
    if not target.exists():
        return None
    with h5py.File(target, 'r') as f:
        node = group_path(entry_name, measurement.point)
        if node not in f:
            return None
        return serialize_cleaned.read_cleaned_events(f[node])


def save_cleaned_events(measurement, entry_name: str, cleaned_fields: dict, meta: dict | None = None, path: Path | None = None):
    """
    Persist cleaned events for (measurement, entry_name) into the events-cache
    copy of the nexus file (or an explicit *path*, if given).
    """
    target = Path(path) if path is not None else ensure_events_file(measurement)
    node = group_path(entry_name, measurement.point)
    with _write_lock:
        with h5py.File(target, 'a') as f:
            if node in f:
                del f[node]
            group = f.require_group(node)
            serialize_cleaned.write_cleaned_events(group, cleaned_fields, meta=meta)
    return target
