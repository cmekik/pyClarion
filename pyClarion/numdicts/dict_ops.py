from __future__ import annotations

from . import numdict as nd
from . import gradient_tape as gt

from typing import Any, Tuple, Callable, TypeVar, Iterable, overload


__all__ = ["mask", "isolate", "keep", "drop", "with_keys", "transform_keys", 
    "merge"]


T = TypeVar("T")
T1, T2 = TypeVar("T1"), TypeVar("T2")


@gt.GradientTape.op(no_grad=True)
def mask(d: nd.NumDict[T]) -> nd.NumDict[T]:
    return nd.NumDict._new(m={k: 1.0 for k in d}, c=0.0)


@gt.GradientTape.op()
def set_c(d: nd.NumDict[T], c: Any) -> nd.NumDict[T]:
    return nd.NumDict._new(m={k: v for k, v in d.items()}, c=float(c))

@gt.GradientTape.grad(set_c)
def _grad_set_c(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T], *, c: Any
) -> Tuple[nd.NumDict[T]]:
    return (mask(grads) * grads,)


@overload
def isolate(d: nd.NumDict[Any]) -> nd.NumDict[Any]:
    ...

@overload
def isolate(d: nd.NumDict[T], *, key: T) -> nd.NumDict[Any]:
    ...

@gt.GradientTape.op()
def isolate(d, *, key=None):
    """
    Return a constant NumDict isolating a value from d.
    
    If key is None, the constant is set to d.c. Otherwise, it is set to d[key].
    """
    if key is None: 
        return nd.NumDict._new(c=d._c)
    else: 
        return nd.NumDict._new(c=d[key])

@gt.GradientTape.grad(isolate)
def _grad_isolate(grads, result, d, *, key):
    raise NotImplementedError()


@gt.GradientTape.op()
def keep(d: nd.NumDict[T], *, sf: Callable[[T], bool]) -> nd.NumDict[T]:
    return nd.NumDict._new(m={k: v for k, v in d.items() if sf(k)}, c=d._c)

@gt.GradientTape.grad(keep)
def _grad_keep(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T], *, 
    sf: Callable[[T], bool]
) -> Tuple[nd.NumDict[T]]:
    raise NotImplementedError()


@gt.GradientTape.op()
def drop(d: nd.NumDict[T], *, sf: Callable[[T], bool]) -> nd.NumDict[T]:
    return nd.NumDict._new(m={k: v for k, v in d.items() if not sf(k)}, c=d._c)

@gt.GradientTape.grad(drop)
def _grad_drop(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T], *, 
    sf: Callable[[T], bool]
) -> Tuple[nd.NumDict[T]]:
    raise NotImplementedError()


@gt.GradientTape.op()
def keep_less(d: nd.NumDict[T], ref: nd.NumDict[T]) -> nd.NumDict[T]:
    return nd.NumDict._new(
        m={k: v for k, v in d.items() if v < ref[k]}, 
        c=d._c)

@gt.GradientTape.grad(keep_less)
def _grad_keep_less(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T], 
    ref: nd.NumDict[T]
) -> Tuple[nd.NumDict[T]]:
    raise NotImplementedError()


@gt.GradientTape.op()
def keep_greater(d: nd.NumDict[T], ref: nd.NumDict[T]) -> nd.NumDict[T]:
    return nd.NumDict._new(
        m={k: v for k, v in d.items() if v > ref[k]}, 
        c=d._c)

@gt.GradientTape.grad(keep_greater)
def _grad_keep_greater(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T], 
    ref: nd.NumDict[T]
) -> Tuple[nd.NumDict[T]]:
    raise NotImplementedError()


@gt.GradientTape.op()
def keep_if(d: nd.NumDict[T], cond: nd.NumDict[T]) -> nd.NumDict[T]:
    return nd.NumDict._new(
        m={k: v for k, v in d.items() if cond[k] != 0.0}, 
        c=d._c)

@gt.GradientTape.grad(keep_if)
def _grad_keep_if(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T], 
    ref: nd.NumDict[T]
) -> Tuple[nd.NumDict[T]]:
    raise NotImplementedError()


@gt.GradientTape.op()
def squeeze(d: nd.NumDict[T]) -> nd.NumDict[T]:
    return nd.NumDict._new(m={k: v for k, v in d.items() if v != d._c}, c=d._c)

@gt.GradientTape.grad(squeeze)
def _grad_squeeze(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T]
) -> Tuple[nd.NumDict[T]]:
    raise NotImplementedError()


@gt.GradientTape.op()
def with_keys(d: nd.NumDict[T], *, ks: Iterable[T]) -> nd.NumDict[T]:
    return nd.NumDict._new(m={k: d[k] for k in ks}, c=d._c)

@gt.GradientTape.grad(with_keys)
def _grad_with_keys(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T], 
    *, ks: Iterable[T]
) -> Tuple[nd.NumDict[T]]:
    raise NotImplementedError()


@gt.GradientTape.op()
def transform_keys(
    d: nd.NumDict[T1], *, kf: Callable[[T1], T2]
) -> nd.NumDict[T2]:
    """
    Transform the keys of d using kf and return the result.
    
    :param d: The NumDict to be transformed.
    :param kf: A function taking keys of d to a new keys space.
    """
    new = nd.NumDict._new(m={kf(k): d[k] for k in d}, c=d._c)
    if len(d) != len(new):
        raise ValueError("Function must be one-to-one on keys of arg d.")
    return new

@gt.GradientTape.grad(transform_keys)
def _grad_transform_keys(
    grads: nd.NumDict[T2], result: nd.NumDict[T2], d: nd.NumDict[T1], *, 
    kf: Callable[[T1], T2]
) -> Tuple[nd.NumDict[T1]]:
    return (transform_keys(grads, kf={kf(k): k for k in d}.__getitem__),)


@gt.GradientTape.op()
def merge(*ds: nd.NumDict[T]) -> nd.NumDict[T]:
    if len(ds) == 0:
        raise ValueError("Merge must be provided with at least one argument.")
    d = nd.NumDict[T]._new(c=0.0)
    for _d in ds: 
        d.update({k: v for k, v in _d.items()}, strict=True)
    return d

@gt.GradientTape.grad(merge)
def _grad_merge(
    grads: nd.NumDict[T], result: nd.NumDict[T], *ds: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], ...]:
    return tuple(grads * mask(d) for d in ds)
