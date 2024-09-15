import math
from typing import TypeVar, Sequence, overload
from ..keyspaces import Index
from .. import numdicts as nd


D = TypeVar("D", bound="nd.NumDict")


_max = max
_min = min


def neg(self: D) -> D:
    d = {k: -v for k, v in self._d.items()} 
    c = -self._c
    return type(self)(self._i, d, c, False)


def abs(self: D) -> D:
    d = {k: math.fabs(v) for k, v in self._d.items()} 
    c = math.fabs(self._c)    
    return type(self)(self._i, d, c, False)


def log(self: D) -> D:
    d = {k: math.log(v) for k, v in self._d.items()}
    c = math.log(self._c)    
    return type(self)(self._i, d, c, False)


def log1p(self: D) -> D:
    d = {k: math.log1p(v) for k, v in self._d.items()}
    c = math.log(self._c)    
    return type(self)(self._i, d, c, False)


def exp(self: D) -> D:
    d = {k: math.exp(v) for k, v in self._d.items()}
    c = math.exp(self._c)    
    return type(self)(self._i, d, c, False)


def expm1(self: D) -> D:
    d = {k: math.exp(v) for k, v in self._d.items()}
    c = math.exp(self._c)    
    return type(self)(self._i, d, c, False)


def shift(self: D, *, x: float) -> D:
    d = {k: v + x for k, v in self._d.items()}
    c = self._c + x  
    return type(self)(self._i, d, c, False)


def scale(self: D, *, x: float) -> D:
    d = {k: v * x for k, v in self._d.items()}
    c = self._c * x
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
def sum(self: D, *, by: nd.KeyForm) -> D:
    ...

@overload
def sum(self: D, *, by: nd.KeyForm, b: int) -> D:
    ...

@overload
def sum(self: D, *others: D) -> D:
    ...

@overload
def sum(self: D, *others: D, bs: Sequence[int | None]) -> D:
    ...

def sum(
    self: D, 
    *others: D, 
    by: nd.KeyForm | None = None,
    b: int | None = None,
    bs: Sequence[int | None] | None = None, 
) -> D:
    match (others, by, b, bs):
        case ((), by, b, None):
            if by is None or not by < self._i.keyform:
                raise ValueError()
            mode = "self" if self._c == 0. or math.isnan(self._c) else "full"
            it = self.group(by, branch=b, mode=mode).items()
            i = Index(self._i.keyspace, by)
            c = self._c if mode == "self" else float("nan")
        case (others, None, None, bs):
            mode = "self" if math.isnan(self._c) else "match"
            it = self.collect(*others, branches=bs, mode=mode)
            i = self._i
            c = math.fsum((self._c, *(other._c for other in others)))
        case _:
            raise ValueError()
    d = {k: math.fsum(vs) for k, vs in it}
    return type(self)(i, d, c, False)


def sub(self: D, other: D, b: int | None = None) -> D:
    if not other._i.keyform <= self._i.keyform:
        raise ValueError()
    mode = "self" if math.isnan(self._c) else "match"
    it = self.collect(other, branches=(b,), mode=mode)
    d = {k: v1 - v2 for k, (v1, v2) in it}
    c = self._c - other._c
    return type(self)(self._i, d, c, False)


@overload
def mul(self: D, *, by: nd.KeyForm) -> D:
    ...

@overload
def mul(self: D, *, by: nd.KeyForm, b: int) -> D:
    ...

@overload
def mul(self: D, *others: D) -> D:
    ...

@overload
def mul(self: D, *others: D, bs: Sequence[int | None]) -> D:
    ...

def mul(
    self: D, 
    *others: D, 
    by: nd.KeyForm | None = None,
    b: int | None = None,
    bs: Sequence[int | None] | None = None
) -> D:
    match (others, by, b, bs):
        case ((), by, b, None):
            if by is None or not by < self._i.keyform:
                raise ValueError()
            mode = "self" if self._c == 1. or math.isnan(self._c) else "full"
            it = self.group(by, branch=b, mode=mode).items()
            i = Index(self._i.keyspace, by)
            c = self._c if mode == "self" else float("nan")
        case (others, None, None, bs):
            mode = "self" if math.isnan(self._c) else "match"
            it = self.collect(*others, branches=bs, mode=mode)
            i = self._i
            c = math.prod((self._c, *(other._c for other in others)))
        case _:
            raise ValueError()
    d = {k: math.prod(vs) for k, vs in it}
    return type(self)(i, d, c, False)


def div(self: D, other: D, b: int | None = None) -> D:
    mode = "self" if math.isnan(self._c) else "match"
    it = self.collect(other, branches=(b,), mode=mode)
    d = {k: v1 / v2 for k, (v1, v2) in it}
    c = self._c / other._c
    return type(self)(self._i, d, c, False)


@overload
def max(self: D, *, by: nd.KeyForm) -> D:
    ...

@overload
def max(self: D, *, by: nd.KeyForm, b: int) -> D:
    ...

@overload
def max(self: D, *others: D) -> D:
    ...

@overload
def max(self: D, *others: D, bs: Sequence[int | None]) -> D:
    ...

def max(
    self: D, 
    *others: D, 
    by: nd.KeyForm | None = None,
    b: int | None = None,
    bs: Sequence[int | None] | None = None
) -> D:
    match (others, by, b, bs):
        case ((), by, b, None):
            if by is None or not by < self._i.keyform:
                raise ValueError()
            mode = "self" if math.isnan(self._c) else "full"
            it = self.group(by, branch=b, mode=mode).items()
            i = Index(self._i.keyspace, by)
            c = self._c if mode == "self" else float("nan")
        case (others, None, None, bs):
            mode = "self" if math.isnan(self._c) else "match"
            it = self.collect(*others, branches=bs, mode=mode)
            i = self._i
            c = _max((self._c, *(other._c for other in others)))
        case _:
            raise ValueError()
    d = {k: _max(vs) for k, vs in it}
    return type(self)(i, d, c, False)


@overload
def min(self: D, *, by: nd.KeyForm) -> D:
    ...

@overload
def min(self: D, *, by: nd.KeyForm, b: int) -> D:
    ...

@overload
def min(self: D, *others: D) -> D:
    ...

@overload
def min(self: D, *others: D, bs: Sequence[int | None]) -> D:
    ...

def min(
    self: D, 
    *others: D, 
    by: nd.KeyForm | None = None,
    b: int | None = None,
    bs: Sequence[int | None] | None = None
) -> D:
    match (others, by, b, bs):
        case ((), by, b, None):
            if by is None or not by < self._i.keyform:
                raise ValueError()
            mode = "self" if math.isnan(self._c) else "full"
            it = self.group(by, branch=b, mode=mode).items()
            i = Index(self._i.keyspace, by)
            c = self._c if mode == "self" else float("nan")
        case (others, None, None, bs):
            mode = "self" if math.isnan(self._c) else "match"
            it = self.collect(*others, branches=bs, mode=mode)
            i = self._i
            c = _min((self._c, *(other._c for other in others)))
        case _:
            raise ValueError()
    d = {k: _min(vs) for k, vs in it}
    return type(self)(i, d, c, False)
