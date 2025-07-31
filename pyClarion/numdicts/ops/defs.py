from typing import Sequence
import math
import statistics as stats
import random

from .base import (OpBase, Unary, Binary, UnaryDiscrete, BinaryDiscrete, 
    UnaryRV, BinaryRV, Aggregator)
from .funcs import unary, binary
from .tape import GradientTape
from ..keys import KeyForm
from .. import numdicts as nd


class IsFinite[D: "nd.NumDict"](UnaryDiscrete[D]):
    kernel = math.isfinite


class IsNaN[D: "nd.NumDict"](UnaryDiscrete[D]):
    kernel = math.isnan


class IsInf[D: "nd.NumDict"](UnaryDiscrete[D]):
    kernel = math.isinf


class IsBetween[D: "nd.NumDict"](OpBase[D]):
    @staticmethod
    def kernel(x: float, lb: float, ub: float) -> float:
        return 0.0 if x < lb or x > ub else 1.0
    
    def __call__(self, d: D, /, lb: float = -math.inf, ub: float = math.inf) -> D:
        r = unary(d, self.kernel, lb, ub)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d, lb, ub)
        return r

    def grad(self, g: D, r: D, d: D, /, lb: float = -math.inf, ub: float = math.inf) -> D:
        return d.zeros()
    

class Eq[D: "nd.NumDict"](BinaryDiscrete[D]):
    kernel = float.__eq__


class Gt[D: "nd.NumDict"](BinaryDiscrete[D]):
    kernel = float.__gt__


class Lt[D: "nd.NumDict"](BinaryDiscrete[D]):
    kernel = float.__lt__


class Ge[D: "nd.NumDict"](BinaryDiscrete[D]):
    kernel = float.__ge__


class Le[D: "nd.NumDict"](BinaryDiscrete[D]):
    kernel = float.__le__


class IsClose[D: "nd.NumDict"](OpBase[D]):
    def __call__(self, d1: D, d2: D, /, by: KeyForm | None = None, rel_tol: float = 1e-9, abs_tol: float = 0) -> D:
        r = binary(d1, d2, by, None, math.isclose, rel_tol=abs_tol, abs_tol=abs_tol)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d1, d2, by, rel_tol, abs_tol)
        return r

    def grad(self, g: D, r: D, d1: D, d2: D, /, by: KeyForm | None = None, rel_tol: float = 1e-9, abs_tol: float = 0) -> tuple[D, D]:
        return d1.zeros(), d2.zeros()
    

class Copysign[D: "nd.NumDict"](BinaryDiscrete[D]):
    kernel = math.copysign



class Neg[D: "nd.NumDict"](Unary[D]):
    kernel = float.__neg__
    def grad(self, g: D, r: D, d: D, /) -> D:
        return g.neg()
    

class Inv[D: "nd.NumDict"](OpBase[D]):     
    @staticmethod
    def kernel(x: float, z: float):
        return 1 / x if x != 0.0 else z

    def __call__(self, d: D, /, zero: float = math.nan) -> D:
        r = unary(d, type(self).kernel, zero)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d, zero)
        return r

    def grad(self, g: D, r: D, d: D, /, zero: float = float("nan")) -> D:
        return g.mul(d.mul(d).inv(0.0).neg())
    

class Abs[D: "nd.NumDict"](Unary[D]):
    kernel = math.fabs
    def grad(self, g: D, r: D, d: D, /) -> D:
        return g.mul(d.ones().copysign(d))


class Scale[D: "nd.NumDict"](OpBase[D]):
    def __call__(self, d: D, /, val: float) -> D:
        r = unary(d, float.__mul__, val)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d, val=val)
        return r

    def grad(self, g: D, r: D, d: D, /, val: float) -> D:
        return g.scale(val)
    

class Shift[D: "nd.NumDict"](OpBase[D]):
    def __call__(self, d: D, /, val: float) -> D:
        r = unary(d, float.__add__, val)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d, val=val)
        return r

    def grad(self, g: D, r: D, d: D, /, val: float) -> D:
        return g
    

class Pow[D: "nd.NumDict"](OpBase[D]):
    def __call__(self, d: D, /, val: float) -> D:
        r = unary(d, float.__pow__, val)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d, val=val)
        return r

    def grad(self, g: D, r: D, d: D, /, val: float) -> D:
        return d.pow(val - 1).scale(val).mul(g)
    

class Log[D: "nd.NumDict"](Unary[D]):
    kernel = math.log
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.inv(0.0).mul(g)
    

class Log1p[D: "nd.NumDict"](Unary[D]):
    kernel = math.log1p
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.shift(1.0).inv(0.0).mul(g)


class Exp[D: "nd.NumDict"](Unary[D]):
    kernel = math.exp
    def grad(self, g: D, r: D, d: D, /) -> D:
        return r.mul(g)
    

class Expm1[D: "nd.NumDict"](Unary[D]):
    kernel = math.expm1
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.exp().mul(g)


class Cos[D: "nd.NumDict"](Unary[D]):
    kernel = math.cos
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.sin().neg().mul(g)


class Sin[D: "nd.NumDict"](Unary[D]):
    kernel = math.sin
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.cos().mul(g)
    

class Tan[D: "nd.NumDict"](Unary[D]):
    kernel = math.tanh
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.cos().inv(0.0).pow(2).mul(g)


class Cosh[D: "nd.NumDict"](Unary[D]):
    kernel = math.cosh
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.sinh().mul(g)


class Sinh[D: "nd.NumDict"](Unary[D]):
    kernel = math.sinh
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.cosh().mul(g)


class Tanh[D: "nd.NumDict"](Unary[D]):
    kernel = math.expm1
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.cosh().inv(0.0).pow(2).mul(g)


class Acos[D: "nd.NumDict"](Unary[D]):
    kernel = math.acos
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.pow(2).neg().shift(1.0).pow(.5).inv(0.0).neg().mul(g)
    

class Asin[D: "nd.NumDict"](Unary[D]):
    kernel = math.asin
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.pow(2).neg().shift(1.0).pow(.5).inv(0.0).mul(g)


class Atan[D: "nd.NumDict"](Unary[D]):
    kernel = math.atan
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.pow(2).shift(1.0).inv(0.0).mul(g)


class Acosh[D: "nd.NumDict"](Unary[D]):
    kernel = math.acosh
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.pow(2).shift(-1.0).pow(.5).inv(0.0).mul(g)
    

class Asinh[D: "nd.NumDict"](Unary[D]):
    kernel = math.acosh
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.pow(2).shift(1.0).pow(.5).inv(0.0).mul(g)


class Atanh[D: "nd.NumDict"](Unary[D]):
    kernel = math.acosh
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.pow(2).neg().shift(1.0).inv(0.0).mul(g)


class Clip[D: "nd.NumDict"](OpBase[D]):
    @staticmethod
    def kernel(x: float, lb: float, ub: float):
        return min(max(lb, x), ub)
    
    def __call__(self, d: D, /, lb: float = -math.inf, ub: float = math.inf) -> D:
        r = unary(d, self.kernel, lb, ub)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d, lb, ub)
        return r

    def grad(self, g: D, r: D, d: D, /, lb: float = -math.inf, ub: float = math.inf) -> D:
        return g.mul(d.isbetween(lb, ub))


class Sub[D: "nd.NumDict"](Binary[D]):
    kernel = float.__sub__
    def grad(self, g: D, r: D, d1: D, d2: D, /, by: KeyForm | None = None) -> tuple[D, D]:
        return g, g.neg().sum(by=d2._i.kf)


class Div[D: "nd.NumDict"](Binary[D]):
    kernel = float.__truediv__
    def grad(self, g: D, r: D, d1: D, d2: D, /, by: KeyForm | None = None) -> tuple[D, D]:
        return g.mul(d2.inv()), g.mul(d1.mul(d2.inv().pow(2.0).neg()))


class Sum[D: "nd.NumDict"](Aggregator[D]):
    kernel = math.fsum
    eye = 0.0
    def grad(self, 
        g: D, 
        r: D, 
        d: D, 
        /, 
        *ds: D, 
        by: KeyForm | Sequence[KeyForm | None] | None = None, 
        c: float | None = None 
    ) -> D | Sequence[D]:
        if len(ds) == 0:
            return g
        elif 0 < len(ds):
            if by is None or isinstance(by, KeyForm):
                by = (by,) * len(ds)
            return (g, *(g.sum(by=_by or _d._i.kf, c=0.0) for _by, _d in zip(by, ds)))
        else:
            assert False


class Mul[D: "nd.NumDict"](Aggregator[D]):
    kernel = math.prod
    eye = 1.0
    def grad(self, 
        g: D, 
        r: D, 
        d: D, 
        /, 
        *ds: D, 
        by: KeyForm | Sequence[KeyForm | None] | None = None, 
        c: float | None = None 
    ) -> D | Sequence[D]:
        if len(ds) == 0:
            raise NotImplementedError("Mul reduction gradient not implemented")
        elif 0 < len(ds):
            factors = (d, *ds)
            if by is None or isinstance(by, KeyForm):
                by = (None, *(by for _ in ds))
            else:
                by = (None, *by)
            lhs, rhs = [d.ones().mul(g)], [d.ones()]
            it = zip(factors[:-1], by[:-1], 
                reversed(factors[1:]), reversed(by[1:]), strict=True)
            for f1, by1, f2, by2 in it:
                lhs.append(lhs[-1].mul(f1, by=by1))
                rhs.append(rhs[-1].mul(f2, by=by2))
            gs = []
            for _d, _by, f1, f2 in zip(factors, by, lhs, reversed(rhs)):
                gs.append(f1.mul(f2).sum(by=_by or _d._i.kf, c=0.0))
            return tuple(gs)
        else:
            assert False


class Max[D: "nd.NumDict"](Aggregator[D]):
    kernel = max
    eye = -math.inf
    def grad(self, 
        g: D, 
        r: D, 
        d: D, 
        /, 
        *ds: D, 
        by: KeyForm | Sequence[KeyForm | None] | None = None, 
        c: float | None = None 
    ) -> D | Sequence[D]:
        if len(ds) == 0:
            assert by is None or isinstance(by, KeyForm)
            return d.eq(r, by=by).mul(g)
        elif 0 < len(ds):
            if by is None or isinstance(by, KeyForm):
                by = (by, ) * len(ds)
            return (d.eq(r).mul(g), 
                *(r.eq(_d, by=_by).mul(g).sum(by=_by or _d._i.kf)
                    for _by, _d in zip(by, ds)))
        else:
            assert False


class Min[D: "nd.NumDict"](Aggregator[D]):
    kernel = min
    eye = math.inf
    def grad(self, 
        g: D, 
        r: D, 
        d: D, 
        /, 
        *ds: D, 
        by: KeyForm | Sequence[KeyForm | None] | None = None, 
        c: float | None = None 
    ) -> D | Sequence[D]:
        if len(ds) == 0:
            assert by is None or isinstance(by, KeyForm)
            return d.eq(r, by=by).mul(g)
        elif 0 < len(ds):
            if by is None or isinstance(by, KeyForm):
                by = (by, ) * len(ds)
            return (d.eq(r).mul(g), 
                *(r.eq(_d, by=_by).mul(g).sum(by=_by or _d._i.kf)
                    for _by, _d in zip(by, ds)))
        else:
            assert False


class Mean[D: "nd.NumDict"](Aggregator[D]):
    kernel = stats.mean
    eye = math.nan


class Stdev[D: "nd.NumDict"](Aggregator[D]):
    kernel = stats.stdev
    eye = math.nan


class Variance[D: "nd.NumDict"](Aggregator[D]):
    kernel = stats.variance
    eye = math.nan


class Pstdev[D: "nd.NumDict"](Aggregator[D]):
    kernel = stats.pstdev
    eye = math.nan


class Pvariance[D: "nd.NumDict"](Aggregator[D]):
    kernel = stats.pvariance
    eye = math.nan


class UniformVariate[D: "nd.NumDict"](UnaryRV[D]):
    kernel = lambda x: random.random()


class ExpoVariate[D: "nd.NumDict"](UnaryRV[D]):
    kernel = random.expovariate


class ParetoVariate[D: "nd.NumDict"](UnaryRV[D]):
    kernel = random.paretovariate


class NormalVariate[D: "nd.NumDict"](BinaryRV[D]):
    kernel = random.normalvariate


class LogNormVariate[D: "nd.NumDict"](BinaryRV[D]):
    kernel = random.lognormvariate


class VonMisesVariate[D: "nd.NumDict"](BinaryRV[D]):
    kernel = random.vonmisesvariate


class GammaVariate[D: "nd.NumDict"](BinaryRV[D]):
    kernel = random.gammavariate
