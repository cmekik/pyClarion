from typing import (Mapping, Iterator, Callable, Concatenate, Sequence, 
    Literal, Self, SupportsFloat, Any, overload)
from itertools import chain
from functools import wraps
from contextlib import contextmanager
import math

from .keys import KeyForm, Key
from .indices import Index, IndexObserver


def inplace[D: "NumDictBase", **P, R](
    f: Callable[Concatenate[D, P], R]
) -> Callable[Concatenate[D, P], R]:
    @wraps(f)
    def wrapper(d: D, *args: P.args, **kwargs: P.kwargs) -> R:
        if d._p: 
            raise RuntimeError("Cannot mutate protected NumDict data.")
        return f(d, *args, **kwargs)
    return wrapper


class NumDictBase(IndexObserver):
    __slots__ = ("_i", "_d", "_c", "_p")

    _i: Index
    _d: dict[Key, float]
    _c: float
    _p: bool

    def __init__(
        self, 
        i: Index,
        d: dict[Key, float], 
        c: float,
        _v: bool = True
    ) -> None:
        if _v: 
            for key in d:
                if key not in i:
                    raise ValueError(f"Key {key} not a member of index")
        self._i = i
        self._d = d 
        self._c = c
        self._p = True
        self.register(i)

    # def _iter_d(self) -> Iterator[Key]:
    #     yield from self._d

    @property
    def i(self) -> Index:
        return self._i

    @property
    def d(self) -> dict[Key, float]:
        return {k: self._d[k] for k in self._d}

    @property
    def c(self) -> float:
        return self._c

    def __len__(self) -> int:
        return len(tuple(self._i))

    def __iter__(self) -> Iterator[Key]:
        yield from self._i

    def __contains__(self, key: str | Key) -> bool:
        k = Key(key)
        return k in self._i
    
    def __getitem__(self, key: str | Key) -> float:
        k = Key(key)
        try:
            return self._d[k]
        except KeyError as e:
            if k in self:
                return self._c
            else:
                raise KeyError(f"Key '{k}' not a member") from e

    def __repr__(self) -> str:
        return (f"<{type(self).__qualname__} '{self.i.kf.as_key()}' "
            f"c={self.c} at {hex(id(self))}>")
    
    def __str__(self) -> str:
        data = [f"{type(self).__qualname__} '{self.i.kf.as_key()}' c={self.c}"]
        width = 0
        for k in self.d:
            width = max(width, len(str(k)))
        for k, v in self.d.items():
            data.append(f"{str(k):<{width}} {v}")
        return "\n    ".join(data)

    def copy(self: Self) -> Self:
        return type(self)(self._i, self.d, self._c)

    def pipe[**P](
        self: Self, 
        f: Callable[Concatenate[Self, P], Self], 
        *args: P.args, 
        **kwdargs: P.kwargs
    ) -> Self:
        """Call a custom function as part of a NumDict method chain."""
        return f(self, *args, **kwdargs)

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
                if isinstance(by, (str, Key)):
                    by = KeyForm.from_key(Key(by))
                reduce = by.reductor(self.i.kf)
                kmax, vmax = {}, {}
                for k in it:
                    group, v = reduce(k), self[k]
                    if vmax.setdefault(group, -math.inf) < v:
                        kmax[group] = k
                        vmax[group] = v
                return {k: v for k, v in kmax.items()}

    @overload
    def argmin(self) -> Key:
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
                if isinstance(by, (str, Key)):
                    by = KeyForm.from_key(Key(by))
                reduce = by.reductor(self.i.kf)
                kmin, vmin = {}, {}
                for k in it:
                    group, v = reduce(k), self[k]
                    if v < vmin.setdefault(group, math.inf):
                        kmin[group] = k
                        vmin[group] = v
                return {k: v for k, v in kmin.items()}

    @contextmanager
    def mutable(self):
        self._p = False
        yield self
        self._p = True

    @inplace
    def __setitem__(self, key: str | Key, value: float) -> None:
        k = Key(key)
        if k in self:
            self._d[k] = float(value)
        else:
            raise ValueError(f"Key '{key}' not a member")

    @c.setter
    @inplace
    def c(self, c: float) -> None:
        self._c = c

    @inplace
    def reset(self) -> None:
        self._d.clear()
    
    @inplace
    def update(
        self: Self, 
        data: Mapping[Key, SupportsFloat] | Mapping[str, SupportsFloat]
    ) -> None:
        for k in data:
            if k not in self:
                raise ValueError(f"Key '{k}' not a member")
        self._d.update({Key(k): float(v) for k, v in data.items()})
    
    def on_del(self, index: Index, key: Key) -> None:
        self._d.pop(key, None)


def nd_iter[D: NumDictBase](self: D, *others: D) -> Iterator[Key]:
    if not others:
        yield from self._d
    else:
        seen = set()
        for obj in chain(nd_iter(self), *(nd_iter(d) for d in others)):
            if obj not in seen and obj in self:
                yield obj
            seen.add(obj)


def collect[D: NumDictBase](
    self: D, 
    *others: D,
    mode: Literal["self", "match", "full"],
    branches: Sequence[KeyForm | None] | KeyForm | None = None
) -> Iterator[tuple[Key, list[float]]]:
    if mode not in ("self", "match", "full"):
        raise ValueError(f"Invalid mode flag: '{mode}'")    
    if (branches is not None and not isinstance(branches, KeyForm) 
        and len(branches) != len(others)):
        raise ValueError(f"len(branches) != len(others)")
    invariant = True; reductors = []
    for i, d in enumerate(others):
        if d._i.root != self._i.root:
            raise ValueError(f"Mismatched keyspaces")
        if d._i.kf != self._i.kf:
            invariant = False
        kf = self._i.kf
        if branches is not None:
            branch = branches if isinstance(branches, KeyForm) else branches[i]
            kf = branch if branch is not None else kf 
        reductors.append(d._i.kf.reductor(kf))
    match mode:
        case "self":
            it = nd_iter(self)
        case "match" if invariant:
            it = nd_iter(self, *others)
        case "match" if not invariant:
            it = self._i
        case "full":
            it = self._i
        case _:
            raise ValueError()
    for k in it:
        data = [self[k]]
        for d, reduce in zip(others, reductors):
            data.append(d[reduce(k)])
        yield k, data


def group(
    self,
    kf: KeyForm,
    mode: Literal["self", "full"] = "full"
) -> dict[Key, list[float]]:
    if mode not in ("self", "full"):
        raise ValueError(f"Invalid mode flag: '{mode}'")
    reduce = kf.reductor(self._i.kf)
    items: dict[Key, list[float]] = {}
    match mode:
        case "self":
            it = self._iter()
        case "full":
            it = self._i
    for k in it:
        items.setdefault(reduce(k), []).append(self[k])
    return items


def unary[D: NumDictBase](
    d: D, kernel: Callable, *args: Any, **kwargs: Any
) -> D:
    new_c = kernel(d._c, *args, **kwargs)
    new_d = {k: new_v for k, v in d._d.items() if (new_v := kernel(v, *args, **kwargs)) != new_c} 
    return type(d)(d._i, new_d, new_c, False)


def binary[D: NumDictBase](
    d1: D, d2: D, kernel: Callable, by: KeyForm | None, *args: Any, **kwargs: Any
) -> D:
    it = collect(d1, d2, mode="match", branches=(by,))
    new_c = kernel(d1._c, d2._c, *args, **kwargs)
    new_d = {k: v for k, (v1, v2) in it if (v := kernel(v1, v2, *args, **kwargs)) != new_c}
    return type(d1)(d1._i, new_d, new_c, False)


def random_variate[D: NumDictBase](
    d: D, 
    *ds: D, 
    kernel: Callable, 
    by: KeyForm | None = None, 
    c: float | None = None
) -> D:
    c = c or d._c
    it = collect(d, *ds, branches=by, mode="full")
    new_d = {k: v for k, vs in it if (v := kernel(*vs)) != c}
    return type(d)(d._i, new_d, c, False)


def aggregator[D: NumDictBase](
    d: D, 
    *others: D,
    kernel: Callable,
    eye: float, 
    by: KeyForm | Sequence[KeyForm | None] | None = None,
    c: float | None = None
) -> D:
    mode = "match" if others else "self" if d._c == eye else "full"
    c = c or eye
    if len(others) == 0 and by is None:
        by = d._i.kf.agg
        it = ()
        i = Index(d._i.root, by)
        assert mode != "match"
        new_c = kernel(group(d, by, mode=mode).get(Key(), (c,)))
    elif len(others) == 0 and isinstance(by, KeyForm):
        if not by < d._i.kf:
            raise ValueError(f"Keyform {by.as_key()} cannot "
                f"reduce {d._i.kf.as_key()}")
        assert mode != "match"
        it = group(d, by, mode=mode).items()
        i = Index(d._i.root, by)
        new_c = d._c if mode == "self" else c
    elif 0 < len(others):
        if c is not None:
            ValueError("Unexpected float value for arg c")
        mode = "match"
        it = collect(d, *others, branches=by, mode=mode)
        i = d._i
        new_c = kernel((d._c, *(other._c for other in others)))
    else:
        assert False
    new_d = {k: v for k, vs in it if (v := kernel(vs)) != new_c}
    return type(d)(i, new_d, new_c, False)