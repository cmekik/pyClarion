from __future__ import annotations
from typing import Tuple, List, Callable

from ..base import Structure, feature, uris


def links(s: Structure) -> List[Tuple[str, str]]:
    return [(module.path, uris.join(module.path, input)) # type: ignore
        for module in s.modules() for input in module.inputs]


def _fspace_key(f: feature):
    vkey = ((type(None), int, str).index(type(f.v)), f.v)
    return (f.d, f.l, vkey)


def _get_fspace(s: Structure, getter: Callable) -> List[feature]:
    fspace = []
    for module in s.modules():
        try: 
            _fspace = getter(module.process)
        except NotImplementedError: 
            pass
        else:
            for f in _fspace: fspace.append(f)
    return sorted(fspace, key=_fspace_key)


def fspace(s: Structure) -> List[feature]:
    return reprs(s) + flags(s) + params(s) + cmds(s)


def reprs(s: Structure) -> List[feature]:
    return _get_fspace(s, lambda p: p.reprs)


def flags(s: Structure) -> List[feature]:
    return _get_fspace(s, lambda p: p.flags)


def params(s: Structure) -> List[feature]:
    return _get_fspace(s, lambda p: p.params)


def cmds(s: Structure) -> List[feature]:
    return _get_fspace(s, lambda p: p.cmds)


def nops(s: Structure) -> List[feature]:
    return _get_fspace(s, lambda p: p.nops)
