from typing import Literal, Iterator, Sequence, Callable, Concatenate, cast
from itertools import chain

from ..keys import Key, KeyForm
from ..indices import Index
from ..undefined import _Undefined, Undefined
from .. import numdicts as nd


def nd_iter(d: "nd.NumDict", *others: "nd.NumDict") -> Iterator[Key]:
    if not others:
        yield from d._d
    else:
        seen = set()
        for obj in chain(nd_iter(d), *(nd_iter(d) for d in others)):
            if obj not in seen and obj in d:
                yield obj
            seen.add(obj)


def collect[D: "nd.NumDict"](
    d: D, 
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
    for i, oth in enumerate(others):
        if oth._i.root != d._i.root:
            raise ValueError(f"Mismatched keyspaces")
        if oth._i.kf != d._i.kf:
            invariant = False
        kf = d._i.kf
        if branches is not None:
            branch = branches if isinstance(branches, KeyForm) else branches[i]
            kf = branch if branch is not None else kf 
        reductors.append(oth._i.kf.reductor(kf))
    match mode:
        case "self":
            it = nd_iter(d)
        case "match" if invariant:
            it = nd_iter(d, *others)
        case "match" if not invariant:
            it = d._i
        case "full":
            it = d._i
        case _:
            raise ValueError()
    for k in it:
        data = [d[k]]
        for oth, reduce in zip(others, reductors):
            data.append(oth[reduce(k)])
        yield k, data


def group(
    d: "nd.NumDict",
    kf: KeyForm,
    mode: Literal["self", "full"] = "full"
) -> dict[Key, list[float]]:
    if mode not in ("self", "full"):
        raise ValueError(f"Invalid mode flag: '{mode}'")
    reduce = kf.reductor(d._i.kf)
    items: dict[Key, list[float]] = {}
    match mode:
        case "self":
            it = nd_iter(d)
        case "full":
            it = d._i
    for k in it:
        items.setdefault(reduce(k), []).append(d[k])
    return items


def unary[**P, D: "nd.NumDict"](
    d: D, 
    kernel: Callable[Concatenate[float, P], float], 
    *args: P.args, 
    **kwargs: P.kwargs
) -> D:
    new_c = (d._c if isinstance(d._c, _Undefined) 
        else float(kernel(d._c, *args, **kwargs)))
    new_d = {k: float(new_v) for k, v in d._d.items() 
        if (new_v := kernel(v, *args, **kwargs)) != new_c}
    return type(d)(d._i, new_d, new_c, False)
    

def binary[**P, D: "nd.NumDict"](
    d1: D, 
    d2: D, 
    by: KeyForm | None, 
    c: float | _Undefined | None,
    kernel: Callable[Concatenate[float, float, P], float], 
    *args: P.args, 
    **kwargs: P.kwargs
) -> D:
    mode = "self" if isinstance(d1._c, _Undefined) else "match"
    it = collect(d1, d2, mode=mode, branches=(by,))
    if c is not None:
        new_c = c
    elif isinstance(d1._c, _Undefined) or isinstance(d2._c, _Undefined):
        new_c = Undefined
    else:
        new_c = kernel(d1._c, d2._c, *args, **kwargs)
    new_d = {k: v for k, (v1, v2) in it 
        if (v := kernel(v1, v2, *args, **kwargs)) != new_c}
    return type(d1)(d1._i, new_d, new_c, False)
    

def variadic[D: "nd.NumDict"](d: D, *ds: D, by: KeyForm | Sequence[KeyForm | None] | None, c: float | _Undefined | None, kernel: Callable[[Sequence[float]], float], eye: float) -> D:
    if isinstance(d._c, _Undefined):
        mode = "self"
    elif 0 < len(ds):
        mode = "match"
    elif d._c == eye:
        mode = "self"
    else:
        mode = "full"
    if len(ds) == 0 and by is None:
        by = d._i.kf.agg
        it = ()
        i = Index(d._i.root, by)
        assert mode != "match"
        new_c = kernel(group(d, by, mode=mode).get(Key(), (eye,)))
    elif len(ds) == 0 and isinstance(by, KeyForm):
        if not by <= d._i.kf:
            raise ValueError(f"Keyform {by.as_key()} cannot "
                f"reduce {d._i.kf.as_key()}")
        assert mode != "match"
        it = group(d, by, mode=mode).items()
        i = Index(d._i.root, by)
        new_c = d._c if mode == "self" and c is None else eye if c is None else c
    elif 0 < len(ds):
        if c is not None:
            ValueError("Unexpected float value for arg c")
        it = collect(d, *ds, branches=by, mode=mode)
        i = d._i
        cs = (d._c, *(other._c for other in ds))
        if any(isinstance(c, _Undefined) for c in cs):
            new_c = Undefined
        else:
            new_c = kernel(cast(tuple[float, ...], cs))
    else:
        assert False
    new_d = {k: v for k, vs in it if (v := kernel(vs)) != new_c}
    return type(d)(i, new_d, new_c, False)
