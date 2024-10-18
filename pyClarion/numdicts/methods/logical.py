import math
from typing import TypeVar, overload
from ..keys import KeyForm
from .. import numdicts as nd


D = TypeVar("D", bound="nd.NumDict")


def isfinite(self: D) -> D:
    d = {k: float(math.isfinite(v)) for k, v in self._d.items()} 
    c = float(math.isfinite(self._c))
    return type(self)(self._i, d, c, False)


def isnan(self: D) -> D:
    d = {k: float(math.isnan(v)) for k, v in self._d.items()} 
    c = float(math.isnan(self._c))
    return type(self)(self._i, d, c, False)


def isinf(self: D) -> D:
    d = {k: float(math.isinf(v)) for k, v in self._d.items()} 
    c = float(math.isinf(self._c))
    return type(self)(self._i, d, c, False)


def isclose(self: D, other: D) -> D:
    mode = "self" if math.isnan(self._c) else "match"
    d = {k: float(math.isclose(v1, v2)) 
        for k, (v1, v2) in self.collect(other, mode=mode)}
    c = float(math.isclose(self._c, other._c))
    return type(self)(self._i, d, c, False)


def gt(self: D, other: D) -> D:
    mode = "self" if math.isnan(self._c) else "match"
    d = {k: float(v1 > v2) for k, (v1, v2) in self.collect(other, mode=mode)}
    c = float(self._c > other._c)
    return type(self)(self._i, d, c, False)


def gte(self: D, other: D) -> D:
    mode = "self" if math.isnan(self._c) else "match"
    d = {k: float(v1 >= v2) for k, (v1, v2) in self.collect(other, mode=mode)}
    c = float(self._c >= other._c)
    return type(self)(self._i, d, c, False)


def lt(self: D, other: D) -> D:
    mode = "self" if math.isnan(self._c) else "match"
    d = {k: float(v1 < v2) for k, (v1, v2) in self.collect(other, mode=mode)}
    c = float(self._c < other._c)
    return type(self)(self._i, d, c, False)


def lte(self: D, other: D) -> D:
    mode = "self" if math.isnan(self._c) else "match"
    d = {k: float(v1 <= v2) for k, (v1, v2) in self.collect(other, mode=mode)}
    c = float(self._c <= other._c)
    return type(self)(self._i, d, c, False)


def with_default(self: D, *, c: bool | int | float) -> D:
    return type(self)(self._i, self._d.copy(), c, False)


def valmax(self) -> float:
    kmax, vmax = None, -math.inf
    for k in self:
        if self[k] > vmax:
            kmax, vmax = k, self[k] 
    assert kmax is not None 
    return vmax


def valmin(self) -> float:
    kmin, vmin = None, math.inf
    for k in self:
        if self[k] < vmin:
            kmin, vmin = k, self[k] 
    assert kmin is not None
    return vmin


@overload
def argmax(self) -> nd.Key:
    ...

@overload
def argmax(self, *, by: KeyForm) -> dict[nd.Key, nd.Key]:
    ...

@overload
def argmax(self, *, by: KeyForm, b: int) -> dict[nd.Key, nd.Key]:
    ...

def argmax(
    self, *, by: KeyForm | None = None, b: int | None = None
) -> nd.Key | dict[nd.Key, nd.Key]:
    it = self._d if math.isnan(self._c) else self._i
    match (by, b):
        case (None, None):
            kmax, vmax = None, -math.inf
            for k in self:
                if self[k] > vmax:
                    kmax, vmax = k, self[k] 
            assert kmax is not None 
            return kmax
        case (by, b):
            assert by is not None
            kmax, vmax = {}, {}
            for k in it:
                group, v = by.reduce(k, b), self[k]
                if vmax.setdefault(group, -math.inf) < v:
                    kmax[group] = k
                    vmax[group] = v
            return {k: v for k, v in kmax.items()}


@overload
def argmin(self) -> nd.Key:
    ...

@overload
def argmin(self, *, by: KeyForm) -> dict[nd.Key, nd.Key]:
    ...

@overload
def argmin(self, *, by: KeyForm, b: int) -> dict[nd.Key, nd.Key]:
    ...

def argmin(
    self, *, by: KeyForm | None = None, b: int | None = None
) -> nd.Key | dict[nd.Key, nd.Key]:
    it = self._d if math.isnan(self._c) else self._i
    match (by, b):
        case (None, None):
            kmin, vmin = None, math.inf
            for k in it:
                if self[k] < vmin:
                    kmin, vmin = k, self[k] 
            assert kmin is not None
            return kmin
        case (by, b):
            assert by is not None
            kmin, vmin = {}, {}
            for k in it:
                group, v = by.reduce(k, b), self[k]
                if v < vmin.setdefault(group, math.inf):
                    kmin[group] = k
                    vmin[group] = v
            return {k: v for k, v in kmin.items()}