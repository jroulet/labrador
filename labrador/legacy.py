"""Deprecated functions kept for backward compatibility."""
import torch
import numpy as np


def _compactify_tanh(value, a, b):
    """
    Compactify a value from an infinite interval to a finite interval
    [a, b] using tanh.

    Parameters
    ----------
    value : float
        Value to be compactified.

    a, b : float
        Bounds of the finite interval.

    Returns
    -------
    float : Compactified value within the interval [a, b].
    """
    return (b - a) / 2 * torch.tanh(value) + (b + a) / 2


def _decompactify_tanh(compact_value, a, b, eps=1e-7):
    """
    Decompactify a value from a finite interval [a, b] to an infinite
    interval using arctanh.

    Parameters
    ----------
    compact_value : float
        Compactified value within the interval [a, b].

    a, b : float
        Bounds of the finite interval.

    eps : float
        Prevents overflow if `compact_value` is close to the edge.

    Returns
    -------
    float : Decompactified value within the infinite interval.
    """
    arg = torch.clamp(2 * (compact_value - (b + a) / 2) / (b - a),
                      -1 + eps, 1 - eps)
    return torch.arctanh(arg)


def _compactify_log_jacobian_determinant_tanh(value, a, b):
    """
    Log of the Jacobian determinant of the ``_compactify`` function.

    That is:

        log |∂{compact_value} / ∂{value}|

    Parameters
    ----------
    value : float
        The value at which to compute the log Jacobian determinant.

    a, b : float
        The bounds of the finite interval.

    Returns
    -------
    float : The log of the Jacobian determinant.
    """
    return torch.log(torch.as_tensor(b - a) / 2) - 2 * _log_cosh(value)


def _log_cosh(x):
    """Numerically stable log(cosh(x))."""
    abs_x = torch.abs(x)
    return abs_x + torch.log1p(torch.exp(-2 * abs_x)) - np.log(2)
