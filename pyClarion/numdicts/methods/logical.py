import math
from typing import TypeVar, overload
from ..keys import KeyForm, Key
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
def argmax(self) -> Key:
    ...

@overload
def argmax(self, *, by: str | Key | KeyForm) -> dict[Key, Key]:
    ...

def argmax(
    self, *, by: str | Key | KeyForm | None = None
) -> Key | dict[Key, Key]:
    it = self._d if math.isnan(self._c) else self._i
    match by:
        case None:
            kmax, vmax = None, -math.inf
            for k in self:
                if self[k] > vmax:
                    kmax, vmax = k, self[k] 
            assert kmax is not None 
            return kmax
        case by:
            if isinstance(by, (str, nd.Key)):
                by = nd.KeyForm.from_key(nd.Key(by))
            reduce = by.reductor(self.i.keyform)
            kmax, vmax = {}, {}
            for k in it:
                group, v = reduce(k), self[k]
                if vmax.setdefault(group, -math.inf) < v:
                    kmax[group] = k
                    vmax[group] = v
            return {k: v for k, v in kmax.items()}


@overload
def argmin(self) -> nd.Key:
    ...

@overload
def argmin(self, *, by: str | Key | KeyForm) -> dict[Key, Key]:
    ...

def argmin(
    self, *, by: str | Key | KeyForm | None = None
) -> Key | dict[Key, Key]:
    it = self._d if math.isnan(self._c) else self._i
    match by:
        case None:
            kmin, vmin = None, math.inf
            for k in it:
                if self[k] < vmin:
                    kmin, vmin = k, self[k] 
            assert kmin is not None
            return kmin
        case by:
            if isinstance(by, (str, nd.Key)):
                by = nd.KeyForm.from_key(nd.Key(by))
            reduce = by.reductor(self.i.keyform)
            kmin, vmin = {}, {}
            for k in it:
                group, v = reduce(k), self[k]
                if v < vmin.setdefault(group, math.inf):
                    kmin[group] = k
                    vmin[group] = v
            return {k: v for k, v in kmin.items()}