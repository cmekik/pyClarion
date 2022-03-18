"""URI manipulation functions."""

from __future__ import annotations
import re
from urllib import parse
from typing import Tuple, List, Dict, TypeVar, Any, overload


ID, SEP, FSEP, SUP = re.compile(r"[\w-]+"), "/", "#", ".." # type: ignore


join = parse.urljoin
split = parse.urlsplit


def ispath(s: str) -> bool:
    """Return True if s is a path."""
    nodes = remove_prefix(s, SEP).split(SEP)
    nodes = nodes[slice(s.startswith(SEP), len(nodes) - s.endswith(SEP))]
    return all(ID.fullmatch(_s) is not None or _s == SUP for _s in nodes)


def split_head(path: str) -> Tuple[str, str]:
    """Separate the first identifier in path from the rest."""
    head, _, tail = path.partition(SEP)
    return head, tail


def commonprefix(path1: str, path2: str) -> str:
    "Return the prefix common to path1 and path2."
    if not ispath(path1): 
        raise ValueError(f"Invalid realizer path '{path1}'.")
    if not ispath(path2): 
        raise ValueError(f"Invalid realizer path '{path2}'.")
    parts1 = path1.split(SEP)
    parts2 = path2.split(SEP)
    common_parts = []
    for part1, part2 in zip(parts1, parts2):
        if part1 != part2: 
            break
        common_parts.append(part1)
    return SEP.join(common_parts)


def remove_prefix(path: str, prefix: str) -> str:
    """Remove prefix from path, if it is present."""
    if path.startswith(prefix): 
        return path[len(prefix):]
    else: 
        return path 


def relativize(target: str, source: str) -> str:
    """Return path from source to target, where target is a child of source."""
    common = commonprefix(target, source)
    if common == source:
        return remove_prefix(target, common).lstrip(SEP)
    raise ValueError(f"'{target}' is not subordinate to '{source}'.")


T, T2 = TypeVar("T"), TypeVar("T2", str, list, tuple, dict)

@overload
def prefix(f: str, p: str) -> str:
    ...

@overload
def prefix(f: Dict[str, T], p: str) -> Dict[str, T]:
    ...

@overload
def prefix(f: List[str], p: str) -> List[str]:
    ...

@overload
def prefix(f: Tuple[str, ...], p: str) -> Tuple[str, ...]:
    ...

def prefix(f: T2, p: str) -> T2:
    """Prefix fragment string or collection f with path p."""
    if isinstance(f, str):
        return FSEP.join([p, f]).strip(FSEP)
    elif isinstance(f, dict):
        return {FSEP.join([p, k]).strip(FSEP): v for k, v in f.items()}
    elif isinstance(f, list):
        return list(FSEP.join([p, x]).strip(FSEP) for x in f)
    elif isinstance(f, tuple):
        return tuple(FSEP.join([p, x]).strip(FSEP) for x in f)
    else:
        raise TypeError(f"Unexpected type for arg 'f': {type(f)}")
