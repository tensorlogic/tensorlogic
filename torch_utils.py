import torch


def maybe_cast_to_tensor(t, dtype=None):
    if t is not None:
        if isinstance(t, torch.Tensor):
            if dtype is not None:
                t = t.type(dtype)
        else:
            t = torch.tensor(t, dtype=dtype)
    return t


def ravel_idx(idx, stride):
    return torch.sum(idx * torch.tensor(stride, dtype=torch.long).reshape(-1, 1), dim=0)


def unravel_idx(idx, stride, shape):
    stride = torch.tensor(stride, dtype=torch.long).view(-1, 1)
    shape = torch.tensor(shape, dtype=torch.long).view(-1, 1)
    return (idx.view(1, -1) // stride) % shape


def sort_index(idx, strides, shape):

    raveled_idx = ravel_idx(idx, strides)
    sorted_ravel_idx, indices = torch.sort(raveled_idx)
    return unravel_idx(sorted_ravel_idx, strides, shape), indices


def search_int(coords, val):
    return None, coords.eq(val)


def search_slice(coords, sl):

    diff = coords - sl.start
    found = (torch.fmod(diff, sl.step) == 0) * (diff >= 0) * (coords < sl.stop)
    coords = diff // sl.step

    return coords[found], found


def search_set(coords, sorted_set):

    # pad with max int to avoid out of bounds(hack)
    sorted_set = torch.constant_pad_nd(sorted_set, (0, 1), maxsize)
    found_idx = torch.searchsorted(sorted_set, coords)
    found = sorted_set[found_idx].eq(coords)

    return found_idx[found], found


def search_coords(coords, sorted_indices, stride):

    return search_set(ravel_idx(coords, stride), ravel_idx(sorted_indices, stride))


def search_straddle_index(coords, straddle_index_sparse, straddle_index_dense, stride):

    straddle_len = straddle_index_sparse.shape[1]

    straddle_index_sparse = ravel_idx(straddle_index_sparse, stride)
    coords = ravel_idx(coords, stride)

    straddle_index_sparse, inv_index = torch.unique(straddle_index_sparse, sorted=True, return_inverse=True)
    found_idx, found = search_set(coords, straddle_index_sparse)

    select_mask = torch.zeros((found_idx.shape[0], straddle_index_sparse.shape[0]), dtype=torch.bool)
    select_mask[torch.arange(found_idx.shape[0]), found_idx] = True
    nnz_select_mask = select_mask[:, inv_index].flatten().nonzero(as_tuple=True)[0]

    new_coords = nnz_select_mask % straddle_len
    sparse_idx = nnz_select_mask // straddle_len
    dense_idx = straddle_index_dense[:, new_coords]

    return new_coords, sparse_idx, dense_idx, found
