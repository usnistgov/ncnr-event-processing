import io
import logging
import zipfile
from pathlib import Path

import h5py
from numpy.typing import NDArray

from . import models
from . import data_cache


# Use the logger from uvicorn so we get pretty formatting
logger = logging.getLogger("uvicorn.error")


def open_nexus_entry(measurement: models.Measurement, refresh: bool = False) -> h5py.Group|None:
    point = measurement.point
    path, filename = measurement.path, measurement.filename
    entry_number = measurement.entry

    # TODO: we should only have one cache, not four (nexus files, event files, binned data, live date)
    nexus = data_cache.load_nexus(filename, datapath=path, refresh=refresh)
    entries = nexus_entries(nexus)
    #print("entries", entries)
    entry_name = entries[entry_number]
    return nexus[entry_name]

def nexus_entries(nexus):
    """List of nexus entries"""
    #print({k: list(v.attrs.items()) for k, v in nexus.items()})
    return list(k for k, v in nexus.items() if v.attrs['NX_class'] == 'NXentry')

def nexus_dup(
    entry: h5py.Group,
    binned: dict[str, NDArray],
    bins: models.Bins,
    split: bool = False,
    filename: str|None = None,
    compresslevel: int = 4,
):
    """
    Returns binned events as (data, filename, mimetype).

    The return data is a byte string of mimetype that can be written directly to filename. It may
    be a NeXus file with all frames in a single entry, or a zip file containing the one NeXus file
    per frame.

    entry is the base NeXus file entry for the data
    binned is the data gathered during binning (detector frames, counts, monitors, sample environment)
    bins are the bin edges
    split is True if the returned data should be a zip
    filename is the base name for the zip file entries, or None to use the entry filename.
    compresslevel is the compression level for the zip file.

    The split format is twice as big and takes several times longer to write.
    """
    # Search each detector group for the DASlogs link containing the counts.
    # Record replacement = {link: data}, but only if there are binned events for the detector.
    detector_links = nexus_detector_replacement(entry)
    detectors = binned['detectors']
    replacement = {
        link: detectors[name] for name, link in detector_links.items()
        if name in detectors  # ... only if the detector event data is available
    }

    # count_time – either full vector if no bin number or scalar for a single bin
    field = entry["control/count_time"]
    replacement[field.attrs["target"]] = binned['count_time']

    # optional monitor_counts – same logic as count_time
    if 'monitors' in binned:
        field = entry["control/monitor_counts"]
        replacement[field.attrs["target"]] = binned['monitors']

    # TODO: need to average temperature per frame, etc., from binned['devices']

    # Resolve the base filename from the root group's attribute if it exists
    # Split into stem and suffixes (preserve multi‑part suffixes like .nxs.ngv)
    if filename is None:
        filename = entry["/"].attrs.get("file_name", "data.nxs")
    p = Path(filename)
    suffixes = "".join(p.suffixes)
    stem = p.name[:-len(suffixes)] if suffixes else p.name

    # Write the in‑memory HDF5 file
    fd_mem = io.BytesIO()
    with h5py.File(fd_mem, "w") as h5out:
        entry.copy(entry, h5out, entry.name)
        for name, value in entry.parent.attrs.items():
            h5out.attrs[name] = value
        # print(f"{entry.name} copied to {h5out}")
        record_bin_edges(h5out[entry.name], bins)

        if split:
            # Make a field for the bin number of the current frame
            bin_number = h5out[f"{entry.name}/control"].create_dataset("bin_number", data=[0])
            bin_number.attrs["long_name"] = "0-origin bin number stored in this entry"
            num_bins = len(bins.edges) - 1

            # Store each frame in a zip file as a separate hdf
            # compresslevel=4 adds 10%
            zip_io = io.BytesIO()
            with zipfile.ZipFile(zip_io, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zipf:
                for k in range(num_bins):
                    # Plug in frame number and frame data
                    bin_number[0] = k
                    for target, value in replacement.items():
                        #print(f"setting h5 {target} to {value[k]}")
                        h5out[target][:] = value[k:k+1] if len(value.shape) == 1 else value[k]

                    # Save the hdf file to zip
                    h5out.flush()
                    frame_data = fd_mem.getvalue()
                    zip_name = f"{stem}_rebinned/{stem}_{k:05d}{suffixes}"
                    zipf.writestr(zip_name, frame_data)

            # Capture the entirety of the zip file
            data = zip_io.getvalue()

        else: # not split
            # Plug vectors into the target locations
            for target, value in replacement.items():
                h5_replace_data(h5out[target], value)
            h5out.flush()
            data = fd_mem.getvalue()

    # Done with the in-memory hdf file
    fd_mem.close()

    rebinned_name = f"{stem}_rebinned{suffixes}"
    mimetype = "application/zip" if split else "application/x-hdf5"
    outfile = rebinned_name + ".zip" if split else rebinned_name

    return data, outfile, mimetype

def h5_replace_data(field: h5py.Dataset, data: NDArray):
    # TODO: preserves attributes but not links
    group = field.parent
    name = field.name
    attrs = {name: value for name, value in field.attrs.items()}
    #print(f"{group} {name} {attrs} {field}")
    del group[name]
    field = group.create_dataset(name, data=data)
    for name, value in attrs.items():
        field.attrs[name] = value

def record_bin_edges(entry: h5py.Group, bins: models.Bins):
    """Record bin edges and mode. Optionally store the bin index for per‑frame files.

    Parameters
    ----------
    entry: h5py.Group
        The NeXus entry being written.
    bins: models.Bins
        The binning definition.
    """
    # TODO: Consider storing the binning info in each detector
    # TODO: Add the appropriate NeXus metadata to the fields
    # TODO: Add masking info, etc.
    control = entry["control"]
    field = control.create_dataset("bin_edges", data=bins.edges)
    field.attrs["units"] = "seconds"
    field.attrs["long_name"] = f"bin edges for {bins.mode} binned data"
    field = control.create_dataset("bin_mode", data=bins.mode)
    field.attrs["long_name"] = "binning mode used to create the histogram"

def nexus_detector_replacement(entry):
    """
    Determine the DAS_logs location of each detector data element.
    """
    #print("instrument", sorted(entry["instrument"].items()))
    # TODO: check that the linked detector has stored data
    # that may be enough to verify that it is active
    detectors = {
        name: group["data"].attrs["target"]
        for name, group in sorted(entry["instrument"].items())
        if group.attrs.get('NX_class', None) == 'NXdetector'
        and "data" in group
        and "target" in group["data"].attrs
    }
    #print("detectors", detectors)
    return detectors

# TODO: optimize hdf copy, currently takes > 3 sec for test
# can get a list of all existing links with these functions:
#
# def visititems(group, func):
#     with h5py._hl.base.phil:
#         def proxy(name):
#             """ Call the function with the text name, not bytes """
#             name = group._d(name)
#             return func(name, group[name])
#         return group.id.links.visit(proxy)

# links = []
# def find_links(name, obj):
#     target = obj.attrs.get('target', None)
#     if target is not None and name != target:
#         links.append([name, target])

# visititems(source, find_links)

# takes 0.3 sec to find all links, 0.3 sec to copy all items...
# so should be able to do all replacements and return copy in < 1 sec

def hdf_copy(source, target, replacement=None):
    # type: (h5py.Group, str) -> h5py.File
    """
    Copy an entry and all sub-entries from source to a destination.

    *source* is an open node in an hdf file.

    *target* is an hdf file opened for writing.

    *replacement* is a dictionary of replacement fields.
    """
    links = _hdf_copy_internal(source, target, replacement)
    for link_to, link_from in sorted(links):
        #print("linking", link_from, link_to)
        target[link_to] = target[link_from]

def _hdf_copy_internal(root, h5file, replacement):
    # type: (h5py.Group, h5py.Group, List[Tuple[str, str]]) -> None
    links = []
    
    for item_name, item in sorted(root.items()):
        item_path = f"{root.name}/{item_name}" if root.name != "/" else f"/{item_name}"
        
        # 1. Handle NeXus logical hard links
        if 'target' in item.attrs and item.attrs['target'] != item_path:
            links.append((item_path, item.attrs['target']))
            
        # 2. Handle Datasets
        elif hasattr(item, 'dtype'):
            if item_path in replacement:
                # Modifying this specific dataset
                data = replacement[item_path]
                node = h5file.create_dataset(item.name, data=data, compression=4)
                
                # Copy attributes
                for k, v in item.attrs.items():
                    node.attrs[k] = v
            else:
                # FAST PATH: C-level binary copy for unmodified datasets
                # Bypasses reading into memory (item[()]) and re-compressing
                root.copy(item, h5file, name=item.name)
                
        # 3. Handle Groups
        else:
            node = h5file.create_group(item.name)
            for k, v in item.attrs.items():
                node.attrs[k] = v
            links.extend(_hdf_copy_internal(item, h5file, replacement))
            
    return links