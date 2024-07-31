import torch
import numpy as np
from math import sqrt


def unit_vectors(v, dim=-1):
    return v / torch.sqrt(torch.sum(v ** 2, dim=dim, keepdim=True) + EPSILON)

def Y_2(rij):
    x = rij[:, :, 0]
    y = rij[:, :, 1]
    z = rij[:, :, 2]
    r2 = torch.clamp(torch.sum(rij ** 2, dim=-1), min=EPSILON)
    output = torch.stack([x * y / r2,
                          y * z / r2,
                          (-x ** 2 - y ** 2 + 2. * z ** 2) / (2 * sqrt(3) * r2),
                          z * x / r2,
                          (x ** 2 - y ** 2) / (2. * r2)],
                         dim=-1)
    return output

def get_eijk():
    eijk_ = np.zeros((3, 3, 3))
    eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
    eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
    return torch.tensor(eijk_, dtype=FLOAT_TYPE)


def norm_with_epsilon(input_tensor, dim=None, keepdim=False):
    return torch.sqrt(torch.clamp(torch.sum(input_tensor ** 2, dim=dim, keepdim=keepdim), min=EPSILON))


def ssp(x):
    return torch.log(0.5 * torch.exp(x) + 0.5)


def rotation_equivariant_nonlinearity(x, nonlin=ssp):
    shape = x.shape
    channels = shape[-2]
    representation_index = shape[-1]

    biases = torch.zeros(channels, dtype=FLOAT_TYPE)

    if representation_index == 1:
        return nonlin(x)
    else:
        norm = norm_with_epsilon(x, dim=-1)
        nonlin_out = nonlin(norm + biases)
        factor = nonlin_out / norm
        return x * factor.unsqueeze(-1)


def difference_matrix(geometry):
    ri = geometry.unsqueeze(1)
    rj = geometry.unsqueeze(0)
    rij = ri - rj
    return rij


def distance_matrix(geometry):
    rij = difference_matrix(geometry)
    dij = norm_with_epsilon(rij, dim=-1)
    return dij