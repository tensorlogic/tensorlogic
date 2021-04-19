import torch


def maybe_cast_to_tensor(t, dtype=None):
    """
    Cast t to a tensor of the specified type if t is not already a tensor of dtype.
    :param t: array_like
        Array like object or tensor to be cast.
    :param dtype: torch.dtype
        Torch type to cast to.
    :return: torch.Tensor
        Cast tensor.
    """
    if t is not None:
        if isinstance(t, torch.Tensor):
            if dtype is not None:
                t = t.type(dtype)
        else:
            t = torch.tensor(t, dtype=dtype)
    return t
