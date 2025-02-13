from typing import (Mapping, Iterator, Callable, Concatenate, Sequence, 
    Literal, Self, SupportsFloat)
from itertools import chain
from functools import wraps
from contextlib import contextmanager

from .keys import KeyForm, Key
from .keyspaces import Index


def numdict(
    i: Index, 
    d: dict[Key, float] | dict[str, SupportsFloat], 
    c: SupportsFloat
) -> "NumDict":
    d = {Key(k): float(v) for k, v in d.items()}
    c = float(c)
    return NumDict(i, d, c)


class NumDict:
    from .methods.logical import (isfinite, isnan, isinf, isclose, gt, gte, lt, 
        lte, with_default, valmax, valmin, argmax, argmin)
    from .methods.arithmetic import (eye, neg, inv, abs, log, log1p, logit, exp, 
        expm1, expit, tanh, bound_max, bound_min, const, shift, scale, pow, sum, 
        sub, mul, div, max, min)
    from .methods.stochastic import (stduniformvariate, normalvariate, 
        lognormvariate, vonmisesvariate, expovariate, gammavariate, 
        paretovariate, logisticvariate, gumbelvariate)

    __slots__ = ("_i", "_d", "_c", "_p", "_r", "__weakref__")

    _i: Index
    _d: dict[Key, float]
    _c: float
    _r: int
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
        self._r = self._i.deletions

    def _iter_d(self) -> Iterator[Key]:
        if self._r == self._i.deletions:
            yield from self._d
            return
        rm = set()
        for k in self._d:
            if k in self._i:
                yield k
            else:
                rm.add(k)
        for k in rm:
            del self._d[k]
        self._r = self._i.deletions

    @property
    def i(self) -> Index:
        return self._i

    @property
    def d(self) -> dict[Key, float]:
        return {k: self._d[k] for k in self._iter_d()}

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
        return (f"<{type(self).__qualname__} {self.i.keyform.as_key()} "
            f"c={self.c} at {hex(id(self))}>")
    
    def __str__(self) -> str:
        data = [f"{type(self).__qualname__} {self.i.keyform.as_key()} c={self.c}"]
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

    def collect(
        self: Self, 
        *others: Self,
        mode: Literal["self", "match", "full"],
        branches: Sequence[int | None] | None = None
    ) -> Iterator[tuple[Key, list[float]]]:
        if mode not in ("self", "match", "full"):
            raise ValueError(f"Invalid mode flag: '{mode}'")
        invariant = True; reductors = []
        for i, d in enumerate(others):
            if d._i.root != self._i.root:
                raise ValueError(f"Mismatched keyspaces")
            if d._i.keyform != self._i.keyform:
                invariant = False
            b = branches[i] if branches is not None else None
            reductors.append(d._i.keyform.reductor(self._i.keyform, b))
        if branches is None:
            branches = tuple(None for _ in others)
        match mode:
            case "self":
                it = self._iter()
            case "match" if invariant:
                it = self._iter(*others)
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
        branch: int | None = None, 
        mode: Literal["self", "full"] = "full"
    ) -> dict[Key, list[float]]:
        if mode not in ("self", "full"):
            raise ValueError(f"Invalid mode flag: '{mode}'")
        reduce = kf.reductor(self._i.keyform, branch)
        items: dict[Key, list[float]] = {}
        match mode:
            case "self":
                it = self._iter()
            case "full":
                it = self._i
        for k in it:
            items.setdefault(reduce(k), []).append(self[k])
        return items

    def _iter(self: Self, *others: Self) -> Iterator[Key]:
        if not others:
            yield from self._iter_d()
        else:
            seen = set()
            for obj in chain(self._iter(), *(d._iter() for d in others)):
                if obj not in seen and obj in self:
                    yield obj
                seen.add(obj)
    
    @contextmanager
    def mutable(self):
        self._p = False
        yield self
        self._p = True

    @staticmethod
    def inplace[D: "NumDict", **P, R](
        f: Callable[Concatenate[D, P], R]
    ) -> Callable[Concatenate[D, P], R]:
        @wraps(f)
        def wrapper(d: D, *args: P.args, **kwargs: P.kwargs) -> R:
            if d._p: 
                raise RuntimeError("Cannot mutate protected NumDict data.")
            return f(d, *args, **kwargs)
        return wrapper

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
