import math
from typing import TypeVar, Sequence, overload

from ..keys import Key, KeyForm
from ..indices import Index
from .. import numdicts as nd


D = TypeVar("D", bound="nd.NumDict")


_max = max
_min = min


def eye(self: D) -> D:
    return self


def neg(self: D) -> D:
    d = {k: -v for k, v in self._d.items()} 
    c = -self._c
    return type(self)(self._i, d, c, False)


def abs(self: D) -> D:
    d = {k: math.fabs(v) for k, v in self._d.items()} 
    c = math.fabs(self._c)    
    return type(self)(self._i, d, c, False)


def inv(self: D) -> D:
    d = {k: 1 / v for k, v in self._d.items()} 
    c = 1 / self._c    
    return type(self)(self._i, d, c, False)


def log(self: D) -> D:
    d = {k: math.log(v) for k, v in self._d.items()}
    c = math.log(self._c)    
    return type(self)(self._i, d, c, False)


def log1p(self: D) -> D:
    d = {k: math.log1p(v) for k, v in self._d.items()}
    c = math.log1p(self._c)    
    return type(self)(self._i, d, c, False)


def logit(self: D) -> D:
    d = {k: math.log(v) - math.log1p(-v) for k, v in self._d.items()}
    c = math.log(self._c) - math.log1p(-self.c)
    return type(self)(self._i, d, c, False)


def exp(self: D) -> D:
    d = {k: math.exp(v) for k, v in self._d.items()}
    c = math.exp(self._c)    
    return type(self)(self._i, d, c, False)


def expm1(self: D) -> D:
    d = {k: math.expm1(v) for k, v in self._d.items()}
    c = math.expm1(self._c)    
    return type(self)(self._i, d, c, False)


def expit(self: D) -> D:
    d = {k: (exp_v := math.exp(v)) / (1 + exp_v) if v < 0 
        else 1 / (1 + math.exp(-v)) for k, v in self._d.items()}
    c = ((exp_c := math.exp(self._c)) / (1 + exp_c) if self._c < 0 
        else 1 / (1 + math.exp(-self._c)))
    return type(self)(self._i, d, c, False)


def cosh(self: D) -> D:
    d = {k: math.cosh(v) for k, v in self._d.items()}
    c = math.cosh(self._c)
    return type(self)(self._i, d, c, False)


def sinh(self: D) -> D:
    d = {k: math.sinh(v) for k, v in self._d.items()}
    c = math.sinh(self._c)
    return type(self)(self._i, d, c, False)


def tanh(self: D) -> D:
    d = {k: math.tanh(v) for k, v in self._d.items()}
    c = math.tanh(self._c)
    return type(self)(self._i, d, c, False)


def const(self: D, *, c: float = 1.0) -> D:
    return type(self)(self._i, {}, c, False)


def shift(self: D, *, x: float) -> D:
    d = {k: v + x for k, v in self._d.items()}
    c = self._c + x  
    return type(self)(self._i, d, c, False)


def scale(self: D, *, x: float) -> D:
    d = {k: v * x for k, v in self._d.items()}
    c = self._c * x
    return type(self)(self._i, d, c, False)


def pow(self: D, *, x: float) -> D:
    d = {k: math.pow(v, x) for k, v in self._d.items()}
    c = math.pow(self._c, x)
    return type(self)(self._i, d, c, False)


def bound_max(self: D, *, x: float) -> D:
    d = {k: _min(v, x) for k, v in self._d.items()}
    c = _min(self._c, x)  
    return type(self)(self._i, d, c, False)


def bound_min(self: D, *, x: float) -> D:
    d = {k: _max(v, x) for k, v in self._d.items()}
    c = _max(self._c, x)  
    return type(self)(self._i, d, c, False)


@overload
def sum(self: D) -> D:
    ...

@overload
def sum(self: D, *, by: KeyForm) -> D:
    ...

@overload
def sum(self: D, *others: D) -> D:
    ...

@overload
def sum(self: D, *others: D, by: KeyForm | Sequence[KeyForm | None]) -> D:
    ...

def sum(
    self: D, 
    *others: D, 
    by: KeyForm | Sequence[KeyForm | None] | None = None,
) -> D:
    match (others, by):
        case ((), None):
            by = self._i.kf.agg
            mode = "self" if self._c == 0. or math.isnan(self._c) else "full"
            it = ()
            i = Index(self._i.root, by)
            c = math.fsum(self.group(by, mode=mode).get(Key(), (self._c,)))
        case ((), by) if isinstance(by, KeyForm):
            if not by < self._i.kf:
                raise ValueError(f"Keyform {by.as_key()} cannot reduce "
                    f"{self._i.kf.as_key()}")
            mode = "self" if self._c == 0. or math.isnan(self._c) else "full"
            it = self.group(by, mode=mode).items()
            i = Index(self._i.root, by)
            c = self._c if mode == "self" else float("nan")
        case ((), by):
            raise TypeError("Expected arg by of type KeyForm")
        case (others, by):
            mode = "self" if math.isnan(self._c) else "match"
            it = self.collect(*others, branches=by, mode=mode)
            i = self._i
            c = math.fsum((self._c, *(other._c for other in others)))
        case _:
            assert False
    d = {k: math.fsum(vs) for k, vs in it}
    return type(self)(i, d, c, False)


def sub(self: D, other: D, *, by: KeyForm | None = None) -> D:
    if not other._i.kf <= self._i.kf:
        raise ValueError()
    mode = "self" if math.isnan(self._c) else "match"
    it = self.collect(other, branches=(by,), mode=mode)
    d = {k: v1 - v2 for k, (v1, v2) in it}
    c = self._c - other._c
    return type(self)(self._i, d, c, False)


@overload
def mul(self: D) -> D:
    ...

@overload
def mul(self: D, *, by: KeyForm) -> D:
    ...

@overload
def mul(self: D, *others: D) -> D:
    ...

@overload
def mul(self: D, *others: D, by: KeyForm | Sequence[KeyForm | None]) -> D:
    ...

def mul(
    self: D, 
    *others: D, 
    by: KeyForm | Sequence[KeyForm | None] | None = None,
) -> D:
    match (others, by):
        case ((), None):
            by = self._i.kf.agg
            mode = "self" if self._c == 1. or math.isnan(self._c) else "full"
            it = ()
            i = Index(self._i.root, by)
            c = math.prod(self.group(by, mode=mode).get(Key(), (self.c,)))
        case ((), by) if isinstance(by, KeyForm):
            if not by < self._i.kf:
                raise ValueError(f"Keyform {by.as_key()} cannot reduce "
                    f"{self._i.kf.as_key()}")
            mode = "self" if self._c == 1. or math.isnan(self._c) else "full"
            it = self.group(by, mode=mode).items()
            i = Index(self._i.root, by)
            c = self._c if mode == "self" else float("nan")
        case ((), by):
            raise TypeError("Expected arg by of type KeyForm")
        case (others, by):
            mode = "self" if math.isnan(self._c) else "match"
            it = self.collect(*others, branches=by, mode=mode)
            i = self._i
            c = math.prod((self._c, *(other._c for other in others)))
        case _:
            assert False
    d = {k: math.prod(vs) for k, vs in it}
    return type(self)(i, d, c, False)


def div(self: D, other: D, *, by: KeyForm | None = None) -> D:
    mode = "self" if math.isnan(self._c) else "match"
    it = self.collect(other, branches=(by,), mode=mode)
    d = {k: v1 / v2 for k, (v1, v2) in it}
    c = self._c / other._c
    return type(self)(self._i, d, c, False)


@overload
def max(self: D) -> D:
    ...

@overload
def max(self: D, *, by: KeyForm) -> D:
    ...

@overload
def max(self: D, *others: D) -> D:
    ...

@overload
def max(self: D, *others: D, by: KeyForm | Sequence[KeyForm | None]) -> D:
    ...

def max(
    self: D, 
    *others: D, 
    by: KeyForm | Sequence[KeyForm | None] | None = None,
) -> D:
    match (others, by):
        case ((), None):
            by = self._i.kf.agg
            mode = "self" if self._c == float("-inf") or math.isnan(self._c) else "full"
            it = ()
            i = Index(self._i.root, by)
            c = _max(self.group(by, mode=mode).get(Key(), (self.c,)))
        case ((), by) if isinstance(by, KeyForm):
            if not by < self._i.kf:
                raise ValueError(f"Keyform {by.as_key()} cannot reduce "
                    f"{self._i.kf.as_key()}")
            mode = "self" if self._c == float("-inf") or math.isnan(self._c) else "full"
            it = self.group(by, mode=mode).items()
            i = Index(self._i.root, by)
            c = self._c if mode == "self" else float("nan")
        case ((), by):
            raise TypeError("Expected arg by of type KeyForm")
        case (others, by):
            mode = "self" if math.isnan(self._c) else "match"
            it = self.collect(*others, branches=by, mode=mode)
            i = self._i
            c = _max((self._c, *(other._c for other in others)))
        case _:
            assert False
    d = {k: _max(vs) for k, vs in it}
    return type(self)(i, d, c, False)


@overload
def min(self: D) -> D:
    ...

@overload
def min(self: D, *, by: KeyForm) -> D:
    ...

@overload
def min(self: D, *others: D) -> D:
    ...

@overload
def min(self: D, *others: D, by: KeyForm | Sequence[KeyForm | None]) -> D:
    ...

def min(
    self: D, 
    *others: D, 
    by: KeyForm | Sequence[KeyForm | None] | None = None,
) -> D:
    match (others, by):
        case ((), None):
            by = self._i.kf.agg
            mode = "self" if self._c == float("inf") or math.isnan(self._c) else "full"
            it = ()
            i = Index(self._i.root, by)
            c = _min(self.group(by, mode=mode).get(Key(), (self.c,)))
        case ((), by) if by is None or isinstance(by, KeyForm):
            if by is None:
                by = self._i.kf.agg
            if not by < self._i.kf:
                raise ValueError(f"Keyform {by.as_key()} cannot reduce "
                    f"{self._i.kf.as_key()}")
            mode = "self" if self._c == float("inf") or math.isnan(self._c) else "full"
            it = self.group(by, mode=mode).items()
            i = Index(self._i.root, by)
            c = self._c if mode == "self" else float("nan")
        case ((), by):
            raise TypeError("Expected arg by of type KeyForm")
        case (others, by):
            mode = "self" if math.isnan(self._c) else "match"
            it = self.collect(*others, branches=by, mode=mode)
            i = self._i
            c = _min((self._c, *(other._c for other in others)))
        case _:
            assert False
    d = {k: _min(vs) for k, vs in it}
    return type(self)(i, d, c, False)
