from __future__ import annotations

from . import numdict as nd
from . import gradient_tape as gt

from . import basic_ops as bops
from . import dict_ops as dops 
from .utils import reduce, by, eltwise

from itertools import product
from typing import TypeVar, Tuple, Callable, overload, Any


__all__ = [
    "reduce_sum", "reduce_max", "reduce_min", "put", "sum_by", "max_by", 
    "min_by"
]

T = TypeVar("T")
T1, T2 = TypeVar("T1"), TypeVar("T2")


@overload
def reduce_sum(d: nd.NumDict[Any]) -> nd.NumDict[Any]:
    ...

@overload
def reduce_sum(d: nd.NumDict[Any], *, key: T) -> nd.NumDict[T]:
    ...

@gt.GradientTape.op()
def reduce_sum(d, *, key=None):
    return reduce(d, f=sum, initial=0.0, key=key)

@overload
def _grad_reduce_sum(
    grads: nd.NumDict[T2], result: nd.NumDict[T2], d: nd.NumDict[T1], *, 
    key: T2
) -> Tuple[nd.NumDict[T1]]:
    ...

@overload
def _grad_reduce_sum(
    grads: nd.NumDict[Any], result: nd.NumDict[Any], d: nd.NumDict[T1], *, 
    key: None
) -> Tuple[nd.NumDict[T1]]:
    ...

@gt.GradientTape.grad(reduce_sum)
def _grad_reduce_sum(grads, result, d, *, key):
    return (dops.isolate(grads, key=key) * dops.mask(d),)


def matmul(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[Any]:
    return reduce_sum(d1 * d2)


@overload
def reduce_max(d: nd.NumDict[Any]) -> nd.NumDict[Any]:
    ...

@overload
def reduce_max(d: nd.NumDict[Any], *, key: T) -> nd.NumDict[T]:
    ...

@gt.GradientTape.op()
def reduce_max(d, *, key=None):
    return reduce(d, f=max, initial=float("-inf"), key=key)

@overload
def _grad_reduce_max(
    grads: nd.NumDict[T2], result: nd.NumDict[T2], d: nd.NumDict[T1], *, 
    key: T2
) -> Tuple[nd.NumDict[T1]]:
    ...

@overload
def _grad_reduce_max(
    grads: nd.NumDict[Any], result: nd.NumDict[Any], d: nd.NumDict[T1], *, 
    key: None
) -> Tuple[nd.NumDict[T1]]:
    ...

@gt.GradientTape.grad(reduce_max)
def _grad_reduce_max(grads, result, d, *, key):
    return (dops.isolate(grads, key=key) * bops.isclose(d, result) * dops.mask(d),)


@overload
def reduce_min(d: nd.NumDict[Any]) -> nd.NumDict[Any]:
    ...

@overload
def reduce_min(d: nd.NumDict[Any], *, key: T) -> nd.NumDict[T]:
    ...

@gt.GradientTape.op()
def reduce_min(d, *, key=None):
    return reduce(d, f=min, initial=float("+inf"), key=key)

@overload
def _grad_reduce_min(
    grads: nd.NumDict[T2], result: nd.NumDict[T2], d: nd.NumDict[T1], *, 
    key: T2
) -> Tuple[nd.NumDict[T1]]:
    ...

@overload
def _grad_reduce_min(
    grads: nd.NumDict[Any], result: nd.NumDict[Any], d: nd.NumDict[T1], *, 
    key: None
) -> Tuple[nd.NumDict[T1]]:
    ...

@gt.GradientTape.grad(reduce_min)
def _grad_reduce_min(grads, result, d, *, key):
    return (dops.isolate(grads, key=key) * bops.isclose(d, result) * dops.mask(d),)


@gt.GradientTape.op()
def put(
    d: nd.NumDict[T1], source: nd.NumDict[T2], *, 
    kf: Callable[[T1], T2], strict: bool = False
) -> nd.NumDict[T1]:
    """
    Map keys of d to values from source according to kf.
    
    Constructs a new mapping such that its members are as follows: 
        {k: source[kf(k)] for k in d}
    """
    return nd.NumDict._new(
        m={k: source[kf(k)] for k in d if not strict or kf(k) in source})

@gt.GradientTape.grad(put)
def _grad_put(
    grads: nd.NumDict[T1], 
    result: nd.NumDict[T1], 
    d: nd.NumDict[T1], 
    source: nd.NumDict[T2],
    *, 
    kf: Callable[[T1], T2],
    strict: bool
) -> Tuple[nd.NumDict[T1], nd.NumDict[T2]]:

    return (nd.NumDict(c=0), sum_by(grads * dops.mask(result), kf=kf))


@gt.GradientTape.op()
def mul_from(
    d: nd.NumDict[T1], source: nd.NumDict[T2], *, 
    kf: Callable[[T1], T2], strict: bool = False
) -> nd.NumDict[T1]:
    """
    Map keys of d to values from source according to kf.
    
    Constructs a new mapping such that its members are as follows: 
        {k: v * source[kf(k)] for k, v in d.items()}
    """
    return nd.NumDict._new(
        m={k: v * source[kf(k)] for k, v in d.items() 
        if not strict or kf(k) in source})

@gt.GradientTape.grad(mul_from)
def _grad_mul_from(
    grads: nd.NumDict[T1], 
    result: nd.NumDict[T1], 
    d: nd.NumDict[T1], 
    source: nd.NumDict[T2],
    *, 
    kf: Callable[[T1], T2],
    strict: bool
) -> Tuple[nd.NumDict[T1], nd.NumDict[T2]]:
    raise NotImplementedError()


@gt.GradientTape.op()
def div_from(
    d: nd.NumDict[T1], source: nd.NumDict[T2], *, 
    kf: Callable[[T1], T2], strict: bool = False
) -> nd.NumDict[T1]:
    """
    Map keys of d to values from source according to kf.
    
    Constructs a new mapping such that its members are as follows: 
        {k: v * source[kf(k)] for k, v in d.items()}
    """
    return nd.NumDict._new(
        m={k: v / source[kf(k)] for k, v in d.items() 
        if not strict or kf(k) in source})

@gt.GradientTape.grad(div_from)
def _grad_div_from(
    grads: nd.NumDict[T1], 
    result: nd.NumDict[T1], 
    d: nd.NumDict[T1], 
    source: nd.NumDict[T2],
    *, 
    kf: Callable[[T1], T2],
    strict: bool
) -> Tuple[nd.NumDict[T1], nd.NumDict[T2]]:
    raise NotImplementedError()


@gt.GradientTape.op()
def sum_by(d: nd.NumDict[T1], *, kf: Callable[[T1], T2]) -> nd.NumDict[T2]:
    """
    Sum the values of d grouped by kf.
    
    The resulting numdict contains a mapping of the following form. 
        {k_out: sum(d[k] for k in d if kf(k) == k_out)}
    """
    return by(d, f=sum, kf=kf)

@gt.GradientTape.grad(sum_by)
def _grad_sum_by(
    grads: nd.NumDict[T2], result: nd.NumDict[T2], d: nd.NumDict[T1], *, 
    kf: Callable[[T1], T2]
) -> Tuple[nd.NumDict]:
    return (put(d, grads, kf=kf),)


@gt.GradientTape.op()
def max_by(d: nd.NumDict[T1], *, kf: Callable[[T1], T2]) -> nd.NumDict[T2]:
    """
    Take the maximum of the values of d grouped by kf.
    
    The resulting numdict contains a mapping of the following form. 
        {k_out: max(d[k] for k in d if kf(k) == k_out)}
    """
    return by(d, f=max, kf=kf)

@gt.GradientTape.grad(max_by)
def _grad_max_by(
    grads: nd.NumDict[T2], result: nd.NumDict[T2], d: nd.NumDict[T1], *, 
    kf: Callable[[T1], T2]
) -> Tuple[nd.NumDict]:
    return (put(d, grads, kf=kf) * bops.isclose(d, put(d, result, kf=kf)),)


@gt.GradientTape.op()
def min_by(d: nd.NumDict[T1], *, kf: Callable[[T1], T2]) -> nd.NumDict[T2]:
    """
    Take the minimum of the values of d grouped by kf.
    
    The resulting numdict contains a mapping of the following form. 
        {k_out: min(d[k] for k in d if kf(k) == k_out)}
    """
    return by(d, f=min, kf=kf)

@gt.GradientTape.grad(min_by)
def _grad_min_by(
    grads: nd.NumDict[T2], result: nd.NumDict[T2], d: nd.NumDict[T1], *, 
    kf: Callable[[T1], T2]
) -> Tuple[nd.NumDict]:
    return (put(d, grads, kf=kf) * bops.isclose(d, put(d, result, kf=kf)),)


@gt.GradientTape.op()
def eltwise_max(*ds: nd.NumDict[T]) -> nd.NumDict[T]:
    return eltwise(*ds, f=max)

@gt.GradientTape.grad(eltwise_max)
def _grad_eltwise_max(
    grads: nd.NumDict[T], result: nd.NumDict[T], *ds: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], ...]:
    raise NotImplementedError()


@gt.GradientTape.op()
def eltwise_min(*ds: nd.NumDict[T]) -> nd.NumDict[T]:
    return eltwise(*ds, f=min)

@gt.GradientTape.grad(eltwise_min)
def _grad_eltwise_min(
    grads: nd.NumDict[T], result: nd.NumDict[T], *ds: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], ...]:
    raise NotImplementedError()


@gt.GradientTape.op()
def outer(d1: nd.NumDict[T1], d2: nd.NumDict[T2]) -> nd.NumDict[Tuple[T1, T2]]:
    return nd.NumDict._new(
        m={(k1, k2): d1[k1] * d2[k2] for k1, k2 in product(d1, d2)})

@gt.GradientTape.grad(outer)
def _grad_outer(
    grads: nd.NumDict[Tuple[T1, T2]], result: nd.NumDict[Tuple[T1, T2]], 
    d1: nd.NumDict[T1], d2: nd.NumDict[T2]
) -> Tuple[nd.NumDict[T1], nd.NumDict[T2]]:
    raise NotImplementedError()

