from __future__ import annotations

from . import numdict as nd
from . import gradient_tape as gt
from .utils import coerce2, op1, op2 
from .utils import sign as _sign
from .utils import isclose as _isclose
from .utils import isnan as _isnan
from .utils import isinf as _isinf
from .utils import isfinite as _isfinite
from .utils import lt as _lt
from .utils import gt as _gt
from .utils import le as _le
from .utils import ge as _ge 

from typing import Tuple, TypeVar, Any 
from math import log as _log
from math import exp as _exp


__all__ = [
    "isfinite", "isnan", "isinf", "replace_inf", "neg", "sign", "log", "exp", 
    "isclose", "less", "greater", "less_equal", "greater_equal", "maximum", 
    "minimum"
]


T = TypeVar("T")


### BASIC UNARY OPS ###


@gt.GradientTape.op(no_grad=True)
def isfinite(d: nd.NumDict[T]) -> nd.NumDict[T]:
    return op1(_isfinite, d)


@gt.GradientTape.op(no_grad=True)
def isnan(d: nd.NumDict[T]) -> nd.NumDict[T]:
    return op1(_isnan, d)


@gt.GradientTape.op(no_grad=True)
def isinf(d: nd.NumDict[T]) -> nd.NumDict[T]:
    return op1(_isinf, d)


@gt.GradientTape.op(no_grad=True)
def replace_inf(d: nd.NumDict[T], val: Any) -> nd.NumDict[T]:
    return nd.NumDict._new(
        m={k: v if _isfinite(v) else float(val) for k, v in d.items()}, 
        c=d._c)


@gt.GradientTape.op()
def neg(d: nd.NumDict[T]) -> nd.NumDict[T]:
    return op1(float.__neg__, d)

@gt.GradientTape.grad(neg)
def _grad_neg(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T]
) -> Tuple[nd.NumDict[T]]:
    return (-grads,)


@gt.GradientTape.op()
def sign(d: nd.NumDict[T]) -> nd.NumDict[T]:
    return op1(_sign, d)


@gt.GradientTape.op()
def absolute(d: nd.NumDict[T]) -> nd.NumDict[T]:
    return op1(abs, d)

@gt.GradientTape.grad(absolute)
def _grad_absolute(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T]
) -> Tuple[nd.NumDict[T]]:
    return (grads * sign(d),)


@gt.GradientTape.op()
def log(d: nd.NumDict[T]) -> nd.NumDict[T]:    
    return op1(_log, d)

@gt.GradientTape.grad(log)
def _grad_log(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T]
) -> Tuple[nd.NumDict[T]]:
    return (grads / d,)


@gt.GradientTape.op()
def exp(d: nd.NumDict[T]) -> nd.NumDict[T]:
    return op1(_exp, d)

@gt.GradientTape.grad(exp)
def _grad_exp(
    grads: nd.NumDict[T], result: nd.NumDict[T], d: nd.NumDict[T]
) -> Tuple[nd.NumDict[T]]:
    return (grads * result,)


### BINARY ARITHMETIC OPS ###


@coerce2
@gt.GradientTape.op()
def add(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(float.__add__, d1, d2)

@gt.GradientTape.grad(add)
def _grad_add(
    grads: nd.NumDict[T], result: nd.NumDict[T], 
    d1: nd.NumDict[T], d2: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], nd.NumDict[T]]:
    return (grads, grads)


@coerce2
@gt.GradientTape.op()
def mul(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(float.__mul__, d1, d2)

@gt.GradientTape.grad(mul)
def _grad_mul(
    grads: nd.NumDict[T], result: nd.NumDict[T], 
    d1: nd.NumDict[T], d2: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], nd.NumDict[T]]:
    return (grads * d2, grads * d1)


@coerce2
@gt.GradientTape.op()
def sub(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(float.__sub__, d1, d2)

@gt.GradientTape.grad(sub)
def _grad_sub(
    grads: nd.NumDict[T], result: nd.NumDict[T], 
    d1: nd.NumDict[T], d2: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], nd.NumDict[T]]:
    return (grads, -grads)


@coerce2
@gt.GradientTape.op()
def rsub(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(float.__rsub__, d1, d2)

@gt.GradientTape.grad(rsub)
def _grad_rsub(
    grads: nd.NumDict[T], result: nd.NumDict[T], 
    d1: nd.NumDict[T], d2: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], nd.NumDict[T]]:
    return (-grads, grads)


@coerce2
@gt.GradientTape.op()
def div(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(float.__truediv__, d1, d2)

@gt.GradientTape.grad(div)
def _grad_div(
    grads: nd.NumDict[T], result: nd.NumDict[T], 
    d1: nd.NumDict[T], d2: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], nd.NumDict[T]]:
    return (grads / d2, -(grads * d1) / (d2 * d2))


@coerce2
@gt.GradientTape.op()
def rdiv(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(float.__rtruediv__, d1, d2)

@gt.GradientTape.grad(rdiv)
def _grad_rdiv(
    grads: nd.NumDict[T], result: nd.NumDict[T], 
    d1: nd.NumDict[T], d2: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], nd.NumDict[T]]:
    return (-(grads * d2) / (d1 * d1), grads / d1)


@coerce2
@gt.GradientTape.op()
def power(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(float.__pow__, d1, d2)

@gt.GradientTape.grad(power)
def _grad_power(
    grads: nd.NumDict[T], result: nd.NumDict[T], 
    d1: nd.NumDict[T], d2: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], nd.NumDict[T]]:
    return (grads * d2 * d1 ** (d2 - 1), grads * log(d1) * d1 ** d2)


@coerce2
@gt.GradientTape.op()
def rpow(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(float.__rpow__, d1, d2)

@gt.GradientTape.grad(rpow)
def _grad_rpow(
    grads: nd.NumDict[T], result: nd.NumDict[T], 
    d1: nd.NumDict[T], d2: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], nd.NumDict[T]]:
    return (grads * log(d2) * d2 ** d1, grads * d1 * d2 ** (d1 - 1))


### BINARY COMPARISON OPS ###


@coerce2
@gt.GradientTape.op(no_grad=True)
def isclose(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(_isclose, d1, d2)


@coerce2
@gt.GradientTape.op(no_grad=True)
def less(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(_lt, d1, d2)


@coerce2
@gt.GradientTape.op(no_grad=True)
def greater(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(_gt, d1, d2)


@coerce2
@gt.GradientTape.op(no_grad=True)
def less_equal(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(_le, d1, d2)


@coerce2
@gt.GradientTape.op(no_grad=True)
def greater_equal(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(_ge, d1, d2)


@coerce2
@gt.GradientTape.op()
def maximum(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(max, d1, d2)

@gt.GradientTape.grad(maximum)
def _grad_maximum(
    grads: nd.NumDict[T], result: nd.NumDict[T], 
    d1: nd.NumDict[T], d2: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], nd.NumDict[T]]:
    return (grads * less_equal(d2, d1), grads * less_equal(d1, d2))


@coerce2
@gt.GradientTape.op()
def minimum(d1: nd.NumDict[T], d2: nd.NumDict[T]) -> nd.NumDict[T]:
    return op2(min, d1, d2)

@gt.GradientTape.grad(minimum)
def _grad_minimum(
    grads: nd.NumDict[T], result: nd.NumDict[T], 
    d1: nd.NumDict[T], d2: nd.NumDict[T]
) -> Tuple[nd.NumDict[T], nd.NumDict[T]]:
    return (grads * less_equal(d1, d2), grads * less_equal(d2, d1))
