"""URI manipulation functions."""

from __future__ import annotations
import re
from urllib import parse
from typing import Tuple


ID, SEP, FSEP, SUP = re.compile("\w+"), "/", "#", ".." # type: ignore


join = parse.urljoin
split = parse.urlsplit


def ispath(path: str) -> bool:
    nodes = remove_prefix(path, SEP).split(SEP)
    nodes = nodes[slice(path.startswith(SEP), len(nodes) - path.endswith(SEP))]
    return all(ID.fullmatch(s) is not None or s == SUP for s in nodes)


def split_head(path: str) -> Tuple[str, str]:
    head, _, tail = path.partition(SEP)
    return head, tail


def commonprefix(path1: str, path2: str) -> str:
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


def remove_prefix(path, prefix):
    if path.startswith(prefix):
        return path[len(prefix):]
    return path 


def relativize(target: str, source: str) -> str:
    common = commonprefix(target, source)
    if common == source:
        return remove_prefix(target, common).lstrip(SEP)
    raise ValueError(f"'{target}' is not subordinate to '{source}'.")
