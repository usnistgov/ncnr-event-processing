import io
import h5py

from . import models
from . import data_cache

def open_nexus_entry(measurement: models.Measurement):
    point = measurement.point
    path, filename = measurement.path, measurement.filename
    entry_number = measurement.entry

    # TODO: we should only have one cache, not four (nexus files, event files, binned data, live date)
    nexus = data_cache.load_nexus(filename, datapath=path)
    entries = nexus_entries(nexus)
    #print("entries", entries)
    entry_name = entries[entry_number]
    return nexus[entry_name]

def nexus_entries(nexus):
    """List of nexus entries"""
    #print({k: list(v.attrs.items()) for k, v in nexus.items()})
    return list(k for k, v in nexus.items() if v.attrs['NX_class'] == 'NXentry')

def nexus_dup(entry, counts, duration, bins):
    #print("counts", counts)
    replacement_target = nexus_detector_replacement(entry)
    replacement = {
        v: counts[k] for k, v in replacement_target.items()
    }
    count_time = entry["control/count_time"]
    replacement[count_time.attrs["target"]] = duration

    bio = io.BytesIO()
    with h5py.File(bio, "w") as target:
        hdf_copy(entry.parent, target, replacement)
        record_bins(target[entry.name], bins)
    data = bio.getvalue()
    bio.close()
    return data

def record_bins(entry, bins):
    # TODO: Consider storing the binning info in each detector
    # TODO: Add the appropriate NeXus metadata to the fields
    # TODO: Add masking info, etc.
    control = entry["control"]
    edges = control.create_dataset("bin_edges", data=bins.edges)
    edges.attrs["units"] = "seconds"
    edges.attrs["long_name"] = f"bin edges for {bins.mode} binned data"
    control.create_dataset("bin_mode", data=bins.mode)

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

def hdf_copy(source, target, replacement):
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
    # print(">>> group copy", root.name, "\n   ", "\n    ".join(sorted(root.keys())))
    for item_name, item in sorted(root.items()):
        item_path = f"{root.name}/{item_name}" if root.name != "/" else f"/{item_name}"
        #print("joining", root.name, item_name, "as", item_path)
        #item_path = posixpath.join(root.name, item_name)
        if 'target' in item.attrs and item.attrs['target'] != item_path:
            # print("linking", item_path, item.name)
            links.append((item_path, item.attrs['target']))
        elif hasattr(item, 'dtype'):
            data = replacement.get(item_path, item[()])
            # print("copying", item_path, item.name)
            node = h5file.create_dataset(item.name, data=data)
            attrs = dict(item.attrs)
            node.attrs.update(item.attrs)
        else:
            # print("making", item_path, item.name)
            # Hope that it is a group...
            node = h5file.create_group(item.name)
            node.attrs.update(item.attrs)
            links.extend(_hdf_copy_internal(item, h5file, replacement))
    return links
