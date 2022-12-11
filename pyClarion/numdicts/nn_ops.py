from __future__ import annotations

from . import numdict as nd
from . import gradient_tape as gt
from . import basic_ops as bops
from . import dict_ops as dops
from . import vec_ops as vops

from .utils import eltwise, op1, by
from .utils import sigmoid as _sigmoid
from .utils import tanh as _tanh

from math import exp as _exp

from typing import Tuple, Callable, Iterable, TypeVar, cast
from random import choices, normalvariate


__all__ = ["sigmoid", "tanh", "boltzmann", "sample", "cam_by", "eltwise_cam", 
    "add_normal_noise"]


T = TypeVar("T")
T1, T2 = TypeVar("T1"), TypeVar("T2")


@gt.GradientTape.op()
def sigmoid(d: nd.NumDict) -> nd.NumDict:
    return op1(_sigmoid, d)

@gt.GradientTape.grad(sigmoid)
def _grad_sigmoid(
    grads: nd.NumDict, result: nd.NumDict, d:nd.NumDict
) -> Tuple[nd.NumDict]:
    return (grads * result * (1 - result),)


@gt.GradientTape.op()
def tanh(d: nd.NumDict) -> nd.NumDict:
    """Apply the tanh function elementwise to d."""
    return op1(_tanh, d)

@gt.GradientTape.grad(tanh)
def _grad_tanh(
    grads: nd.NumDict, result: nd.NumDict, d:nd.NumDict
) -> Tuple[nd.NumDict]:
    return (grads * (1 - result * result),)


@gt.GradientTape.op()
def boltzmann(d: nd.NumDict, t: nd.NumDict) -> nd.NumDict:
    """Construct a boltzmann distribution from d with temperature t."""
    if not len(d):
        raise ValueError("Arg d should not be empty.")
    if len(t):
        raise ValueError("Temperature should be a scalar.")
    ks, vs = zip(*d.items())
    vmax, _t = max(vs), t._c
    # v - vmax is a stability trick; softmax(x) = softmax(x + c)
    exp_v = [_exp((v - vmax) / _t) for v in vs]
    assert len(exp_v)
    sum_exp_v = sum(exp_v)
    vals = [v / sum_exp_v for v in exp_v]
    return nd.NumDict._new(m={k: v for k, v in zip(ks, vals)})

@gt.GradientTape.grad(boltzmann)
def _grad_boltzmann(
    grads: nd.NumDict, result: nd.NumDict, d: nd.NumDict, t: nd.NumDict
) -> Tuple[nd.NumDict]:
    raise NotImplementedError()


@gt.GradientTape.op()
def sample(d: nd.NumDict) -> nd.NumDict:
    """Sample a key from d according to its strength."""
    if not len(d):
        raise ValueError("NumDict must be non-empty.")
    cs, ws = tuple(zip(*d.items()))
    s = choices(cs, weights=cast(Tuple[float], ws))
    return nd.NumDict._new(m={k: 1.0 if k in s else 0.0 for k in d})

@gt.GradientTape.grad(sample)
def _grad_sample(
    grads: nd.NumDict, result: nd.NumDict, d: nd.NumDict
) -> Tuple[nd.NumDict]:
    return (grads * result,)


@gt.GradientTape.op()
def cam_by(d: nd.NumDict[T1], *, kf: Callable[[T1], T2]) -> nd.NumDict[T2]:
    return by(d, f=_cam, kf=kf)

@gt.GradientTape.grad(cam_by)
def _grad_cam_by(
    grads: nd.NumDict, result: nd.NumDict, d: nd.NumDict
) -> Tuple[nd.NumDict]:
    raise NotImplementedError()

def _cam(xs: Iterable[float]) -> float:
    _xs = [0.0]; _xs.extend(xs)
    return max(_xs) + min(_xs)


@gt.GradientTape.op()
def eltwise_cam(*ds: nd.NumDict[T]) -> nd.NumDict[T]:
    return eltwise(*ds, f=_cam)

@gt.GradientTape.grad(eltwise_cam)
def _grad_eltwise_cam(
    grads: nd.NumDict[T], result: nd.NumDict[T], *ds: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], ...]:
    raise NotImplementedError()


@gt.GradientTape.op()
def add_normal_noise(
    d: nd.NumDict[T], *, mu: float = 0, sigma: float = 1
) -> nd.NumDict[T]:
    """Sample a key from d according to its strength."""
    if sigma < 0:
        raise ValueError("Std must be non-negative.")
    return nd.NumDict._new(m={k: v + normalvariate(mu, sigma) 
        for k, v in d.items()})

@gt.GradientTape.grad(add_normal_noise)
def _grad_add_normal_noise(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T]
) -> Tuple[nd.NumDict[T]]:
    raise NotImplementedError()