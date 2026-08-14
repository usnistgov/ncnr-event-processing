"""
Alternative histogramming backends explored for event_processing.rebinning_api.binning.
The numba backend (binning._hist_numba) is the one actually used by the server; these
torch/numpy variants were benchmarked against it but aren't wired up anywhere.

Not maintained; may not run as-is (e.g. torch is not declared as a project dependency).
"""
import numpy as np


def _get_torch_gpu():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "cpu"
        return "mps"
    return "cpu"

def _hist_torch_addat(dims, edges, ts, x, y):
    import torch
    from torch.nn.functional import pad

    device = _get_torch_gpu()

    # include two extra bins for the timestamps that fall outside the defined bins
    # (those with indices 0 and n_bins + 1); for an array of size n+1 searchsorted
    # returns insertion indices from 0 (below the left edge) to n+1 (past the right edge)
    # To get the indexing to work right for values outside the range, need an
    # extra zero for the left edge.
    n_bins = len(edges) - 1
    nx, ny = dims
    edges = torch.from_numpy(np.asarray(edges, np.int64)).to(device=device)
    ts_edges = pad(edges, (1, 0), "constant", -2**63)
    ts = torch.from_numpy(ts.view(dtype=np.int64)).to(device=device)
    time_index = torch.searchsorted(ts_edges, ts, side='right').to(dtype=torch.int32, device=device)
    x = torch.from_numpy(x).to(dtype=torch.int32, device=device)
    y = torch.from_numpy(y).to(dtype=torch.int32, device=device)
    # Note: uses (int64 * ny + int?)*nx + int? which promotes to int64.
    # Rewriting as int64*ny*nx + int?*nx + int? may lose digits on y if it is int8 or int16.
    bin_index = ((time_index - 1)*nx + x)*ny + y # !!!! careful about type promotion !!!!
    source = torch.ones_like(bin_index)
    binned = torch.zeros((n_bins + 2)*nx*ny, dtype=torch.int32, device=device)
    #print("rebin_torch_index_add", time_index.dtype, tubeID.dtype, pixel.dtype, source.dtype, source.shape, bin_index.dtype, bin_index.shape)
    binned.index_add_(0, bin_index, source)
    binned = binned.reshape((n_bins+2, nx, ny))
    #print(f"events={len(ts)} binned={binned.sum()} trimmed={binned[1:-1].sum()}")

    #print(f"R:{binned[1:-1, 0:48].sum()} T:{binned[1:-1, 48:96].sum()}  B:{binned[1:-1, 96:144].sum()}  L:{binned[1:-1, 144:192].sum()}")
    # throw away the data in the outside bins
    binned = binned[1:-1].cpu().numpy()
    #print(f"trimmed={binned.sum()}")

    # returns: detectors data, and bin edges in seconds
    return binned

def _hist_torch_dd(dims, edges, ts, x, y):
    import torch

    device = _get_torch_gpu()

    # histogramdd is inclusive on the rightmost edge, so need to go one bin further
    edges = np.append(edges, edges[-1]+1)

    nx, ny = dims
    bins = (
        torch.from_numpy(edges.astype('float64')).to(device=device),
        torch.arange(nx+1, dtype=torch.float64, device=device),
        torch.arange(ny+1, dtype=torch.float64, device=device),
    )
    ts = torch.from_numpy(ts).to(dtype=torch.float64, device=device)
    x = torch.from_numpy(x).to(dtype=torch.float64, device=device)
    y = torch.from_numpy(y).to(dtype=torch.float64, device=device)
    data = torch.stack((ts, x, y), dim=1)
    # print("rebin_torch", ts.shape, data.shape, data.dtype, [v.dtype for v in bins])
    binned, edges = torch.histogramdd(data, bins=bins)
    binned = binned[:-1] # trim the right edge
    binned = binned.to('cpu').numpy()
    return np.asarray(binned, dtype='int32')

def _hist_numpy(dims, edges, ts, x, y):
    nx, ny = dims
    n_bins = len(edges) - 1
    # include two extra bins for the timestamps that fall outside the defined bins
    # (those with indices 0 and n_bins + 1); for an array of size n+1 searchsorted
    # returns insertion indices from 0 (below the left edge) to n+1 (past the right edge)
    binned = np.zeros((n_bins + 2, nx, ny), dtype='int32')
    # the operation below can be repeated... streaming histograms!
    time_index = np.searchsorted(edges, ts, side='right')
    np.add.at(binned, (time_index, x, y), 1)
    # throw away the data in the outside bins
    binned = binned[1:-1]
    return binned

def _hist_numpy_dd(dims, edges, ts, x, y):
    edges = np.asarray(edges, 'int64')
    # histogramdd is inclusive on the rightmost edge, so need to go one bin further
    edges = np.append(edges, edges[-1]+1)
    ts = np.asarray(ts, 'int64')
    x = np.asarray(x, 'int64')
    y = np.asarray(y, 'int64')

    nx, ny = dims
    bins = (
        edges,
        np.arange(nx+1, dtype='int64'),
        np.arange(ny+1, dtype='int64'),
    )
    data = np.stack((ts, x, y), axis=1)
    binned, edges = np.histogramdd(data, bins=bins)
    binned = binned[:-1] # trim the right edge
    return np.asarray(binned, dtype='int32')


def test_hist():
    # 3 x 2 detector with 4 timesteps
    dims = (3, 2)
    edges = [2, 5, 12, 18, 24]
    events = [
        (3, 1, 0),  # t=3, x=1, y=0
        (3, 1, 0),  # Another event at the same coordinates
        (5, 2, 1),  # On a left bin boundary
        (12, 2, 1),  # On a right bin boundary
        (20, 1, 1), # Out of order events
        (15, 0, 0), # Out of order events
        (2, 2, 0),  # At the first bin edge
        (24, 2, 0),  # At the last bin edge
        (1, 1, 0),  # Before the first time bin
        (25, 1, 1),  # After the last time bin
    ]
    ts, x, y = zip(*events)
    target = np.zeros((4, 3, 2), dtype='int32')
    target[0,1,0] = 2 # 3,1,0 x 2
    target[1,2,1] = 1 # 5,2,1
    target[2,2,1] = 1 # 12,2,1
    target[3,1,1] = 1 # 20,1,1
    target[2,0,0] = 1 # 15,0,0
    target[0,2,0] = 1 # 2,2,0
    #print(target)

    edges = np.asarray(edges, dtype='int64')
    x = np.asarray(x, dtype='int32')
    y = np.asarray(y, dtype='int32')
    ts = np.asarray(ts, dtype='int64')
    # The remainder are not in the histogram
    for backend in (
            _hist_torch_addat,
            _hist_numpy,
            _hist_numpy_dd,
            _hist_torch_dd,
        ):
        # Using positional arguments to verify that x,y are in the correct order in all backends
        data = backend(dims, edges, ts, x, y)
        # print(f"{backend.__name__}\n{data}")
        assert data.shape == target.shape, f"Shape mismatch for {backend.__name__}: {data.shape}"
        assert (data == target).all(), f"Match fails for {backend.__name__}:\n{data}\n{target}"

if __name__ == "__main__":
    test_hist()
