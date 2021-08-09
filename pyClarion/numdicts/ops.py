"""Ops on numerical dictionaries with automdiff support."""


# TODO: GradientOps may not handle defaults correctly! Check and correct. - Can


__all__ = ["log", "exp", "sigmoid", "tanh", "set_by",
           "sum_by", "max_by", "threshold", "clip", "boltzmann", "keep", "drop", "transform_keys"]


from .numdicts import (
    D, NumDict, MutableNumDict, record_call, register_op, register_grad
)
from .funcs import by, val_max, val_sum, with_default

from typing import TypeVar, Union, Callable, Hashable, Container, Any
import math


def log(d: D) -> NumDict:
    """Compute the elementwise natural logarithm of d."""

    return d.log()


def exp(d: D) -> NumDict:
    """Compute the base-e exponential of d."""

    return d.exp()


def sigmoid(d: D) -> NumDict:
    """Apply the logistic function elementwise to d."""

    return 1 / (1 + (-d).exp())


def tanh(d: D) -> NumDict:
    """Apply the tanh function elementwise to d."""

    return (2 * sigmoid(d)) - 1


@register_op
def set_by(
    target: D, source: D, *, keyfunc: Callable[..., Hashable]
) -> NumDict:
    """
    Construct a numdict mapping target keys to matching values in source.

    For each key in source, output[key] = source[keyfunc(key)]. Defaults are
    discarded.
    """

    value = NumDict({k: source[keyfunc(k)] for k in target}, None)
    record_call(set_by, value, (target, source), {"keyfunc": keyfunc})

    return value


@register_grad(set_by)
def _grad_set_by(grads, target, source, *, keyfunc):

    return (grads * 0, sum_by(grads, keyfunc=keyfunc))


@register_op
def sum_by(d: D, *, keyfunc: Callable[..., Hashable], **kwds: Any) -> NumDict:
    """
    Sum the values of d grouped by keyfunc.

    Maps each l in the range of keyfunc to the sum of all d[k] such that
    keyfunc(k) is equal to l. See by() for further details.
    """

    value = by(d, sum, keyfunc, **kwds)
    _kwds = {"keyfunc": keyfunc}
    _kwds.update(kwds)
    record_call(sum_by, value, (d,), _kwds)

    return value


@register_grad(sum_by)
def _grad_sum_by(grads, d, *, keyfunc):
    return (set_by(d, grads, keyfunc=keyfunc),)


@register_op
def max_by(d: D, *, keyfunc: Callable[..., Hashable], **kwds: Any) -> NumDict:
    """
    Find maximum values in d grouped by keyfunc.

    Maps each l in the range of keyfunc to the max of all d[k] such that
    keyfunc(k) is equal to l. See by() for further details.
    """

    value = by(d, max, keyfunc, **kwds)
    _kwds = {"keyfunc": keyfunc}
    _kwds.update(kwds)
    record_call(max_by, value, (d,), _kwds)

    return value

# TODO: _grad_max_by is not differentiable, should probably be made so. - Can


@register_grad(max_by)
def _grad_max_by(grads, d, *, keyfunc):

    _isclose = math.isclose
    y = max_by(d, keyfunc=keyfunc)
    arg_max = {k for k, v in d.items() if _isclose(v, y[keyfunc(k)])}
    grad_max = NumDict(
        {k: grads[keyfunc(k)] if k in arg_max else 0 for k in d})

    return (grad_max,)


@register_op
def threshold(
        d: D, *, th: Union[float, int], keep_default: bool = False) -> NumDict:
    """
    Return a copy of d containing only values above theshold.

    If the default is below or equal threshold it is set to None in the output, unless
    keep default is True.
    """

    mapping = {k: d[k] for k in d if th < d[k]}
    if d.default is not None:
        default = d.default if keep_default or th < d.default else None
    else:  # added this to prevent errors when d.default was none as default was undefined
        default = None
    value = NumDict(mapping, default)
    _kwds = {"th": th}
    record_call(threshold, value, (d,), _kwds)
    return value


@register_grad(threshold)
def _grad_threshold(grads, d, *, th):
    mapping = {k: (th < d[k]) * grads[k] for k in d}
    return (NumDict(mapping, default=0),)


@register_op
def clip(d: D, low: float = None, high: float = None) -> NumDict:
    """
    Return a copy of d with values clipped.

    dtype must define +/- inf values.
    """

    low = low or float("-inf")
    high = high or float("inf")

    mapping = {k: max(low, min(high, d[k])) for k in d}
    value = NumDict(mapping, d.default)
    _kwds = {"low": low, "high": high}
    record_call(clip, value, (d,), _kwds)
    return value


@register_grad(clip)
def _grad_clip(grads, d, *, low, high):
    mapping = {k: (low < d[k] < high)*grads[k] for k in d}
    return (NumDict(mapping, default=0),)


@register_op
def boltzmann(d: D, t: Union[float, int]) -> NumDict:
    """(low < d[k] < high)
    Construct a boltzmann distribution from d with temperature t.

    If d has a default, the returned value will have a default of 0, and, if d
    is empty, the return value will also be empty.
    """
    default = 0 if d.default is not None else None
    _kwds = {"t": t}
    if len(d) > 0:
        x = d / t
        x = x - val_max(x)  # softmax(x) = softmax(x + c)
        numerators = x.exp()
        denominator = val_sum(numerators)
        value = with_default(numerators / denominator, default=default)
        record_call(boltzmann, value, (d,), _kwds)
        return value
    else:
        value = NumDict(default=default)
        record_call(boltzmann, value, (d,), _kwds)
        return NumDict(default=default)


@ register_grad(boltzmann)  # LOOK UP SOFTMAX
def _grad_boltzmann(grads, d, *, t):
    if(len(d) > 0):
        return (None,)
    else:
        return (NumDict(default=0),)


def keep(
    d: D,
    func: Callable[..., bool] = None,
    keys: Container = None,
    **kwds: Any


) -> NumDict:
    """
    Return a copy of d keeping only the desired keys.

    Keys are kept iff func(key, **kwds) or key in container is True.
    """

    if func is None and keys is None:
        raise ValueError("At least one of func or keys must not be None.")

    mapping = {
        k: d[k] for k in d
        if (
            (func is not None and func(k, **kwds)) or
            (keys is not None and k in keys)
        )
    }
    _kwds = {"func": func, "keys": keys, **kwds}

    value = NumDict(mapping, d.default)
    record_call(keep, value, (d,), _kwds)
    return value


@ register_grad(keep)
def _grad_keep(grads, d, *, func, keys, **kwds):
    mapping = {
        k: (
            # KWDS FAILS HERE TODO FIGURE OUT WHY
            (func is not None and func(k, **kwds)) or
            (keys is not None and k in keys)
        )*grads[k]
        for k in d
    }
    return (NumDict(mapping, grads.default),)  # default prob incorrect


def drop(
    d: D,
    func: Callable[..., bool] = None,
    keys: Container = None,
    **kwds: Any
) -> NumDict:
    """
    Return a copy of d dropping unwanted keys.

    Keys are dropped iff func(key, **kwds) or key in container is True.
    """

    if func is None and keys is None:
        raise ValueError("At least one of func or keys must not be None.")
    mapping = {
        k: d[k] for k in d
        if ((func is not None and not func(k, **kwds)) and
            (keys is not None and k not in keys))
    }
    _kwds = {"func": func, "keys": keys, **kwds}  # TODO TEST THIS
    value = NumDict(mapping, d.default)
    record_call(drop, value, (d,), _kwds)
    return value


@ register_grad(drop)
def _grad_drop(grads, d, *, func, keys, **kwds):
    mapping = {
        k: ((func is not None and not func(k, **kwds)) and
            (keys is not None and k not in keys))*grads[k] for k in d
    }
    return (NumDict(mapping, grads.default),)  # default prob incorrect


def transform_keys(d: D, func: Callable[..., Hashable], **kwds) -> NumDict:
    """
    Return a copy of d where each key is mapped to func(key, **kwds).

    Warning: If function is not one-to-one wrt keys, will raise ValueError.
    """

    mapping = {func(k, **kwds): d[k] for k in d}

    if len(d) != len(mapping):
        raise ValueError("Func must be one-to-one on keys of arg d.")
    value = NumDict(mapping, d.default)
    _kwds = {"func": func, **kwds}  # TODO TEST THIS
    record_call(transform_keys, value, (d,), _kwds)
    return value


@ register_grad(transform_keys)
# ASK ABOUT THIS IMPLEMENTATION BC IT'S WHACKY
def _grad_transform_keys(grads, d, *, func, **kwds):
    mapping = {func(k,**kwds): grads[k]/func(grads[k], **kwds) for k in d}
    return (NumDict(mapping, 1*grads.default),)  # default prob incorrect
