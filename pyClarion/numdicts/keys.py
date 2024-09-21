from typing import Self, Type, Sequence, Callable, cast
from dataclasses import dataclass
from collections import deque
from functools import cache
import re

from .exc import ValidationError


def sig_cache[T: Callable](f: T) -> T:
    return cast(T, cache(f))


class Key(tuple[tuple[str, int], ...]):
    __slots__ = ()
    
    @sig_cache
    def __new__(cls: Type[Self], s: Self | str = "") -> Self:
        if isinstance(s, cls):
            return s
        assert isinstance(s, str)
        if any(c.isspace() for c in s):
            raise ValidationError("No spaces allowed")
        template = "{}" 
        atomic, compound = r"[^(),]+", r"[^(),]+(?:,[^(),]+)+"
        outer = fr"((?:)|(?:{atomic})|(?:\({compound}\)))"
        pat_o = re.compile(template.format(outer))
        pat_i = re.compile(atomic)
        cur, res = [["", 0]], []
        for segment in s.split(":"):
            m = pat_o.fullmatch(segment)
            if m is None:
                raise ValidationError("Invalid key string")
            groups = m.groups()
            if len(groups) != len(cur):
                raise ValidationError("Invalid key string")
            new_cur, new_template = [], []
            for p, g in zip(cur, groups):
                children = pat_i.findall(g)
                p[1] = len(children)
                res.append(p)
                new_cur.extend([[child, 0] for child in children])
                new_template.append(
                    r"" if not children else 
                    r"{}" if len(children) == 1 else
                    r"\({}\)".format(','.join(["{}"] * len(children))))
            template = template.format(*new_template)
            pat_o = re.compile(template.format(*([outer] * len(new_cur))))
            cur = new_cur
        else:
            res.extend(cur)
        return super().__new__(cls, [(label, degree) for label, degree in res])
    
    def __str__(self) -> str:
        S, cur, nxt, fmt, lvs, res = 0, "{}", [], [], [], []
        for i, (label, degree) in enumerate(self):
            if S == 0 or i == lvs[-1]:
                lvs.append(S + 1)
            S += degree
            nxt.append(
                "" if degree == 0 else 
                "{}" if degree == 1 else 
                f"({','.join(["{}" for _ in range(degree)])})")
            fmt.append(label)
            if i + 1 in lvs:
                res.append(cur.format(*fmt))
                cur = cur.format(*nxt)      
                nxt.clear() 
                fmt.clear()
        return ":".join(res[1:])

    def __repr__(self) -> str:
        return f"Key({repr(str(self))})"
    
    @sig_cache
    def __bool__(self) -> bool:
        return self != Key()

    @sig_cache
    def __le__(self: Self, other) -> bool:
        if not isinstance(other, Key):
            return NotImplemented
        return bool(self.find_in(other))

    @property
    @sig_cache
    def height(self) -> int:
        S, LVs = 1, []
        for i, (label, degree) in enumerate(self):
            if S == 1 or i == LVs[-1]:
                LVs.append(S)
            S += degree
        return len(LVs) - 1

    @property
    def size(self) -> int:
        return len(self) - 1

    @sig_cache
    def find_in(self: Self, other: "Key") -> Sequence[tuple[int, ...]]:
        N_s, N_o = len(self), len(other)
        if N_o < N_s:
            return []
        Z_o, matches, Zs = 0, deque(), deque()
        for i_o, (label_o, degree_o) in enumerate(reversed(other)):
            i_o = N_o - i_o - 1
            Z_o += degree_o
            children = {N_o - Z_o + j for j in range(degree_o)}
            for i_m, (match, Z_s) in enumerate(zip(matches, Zs)):
                i_s = N_s - len(match) - 1
                label_s, degree_s = self[i_s]
                if label_o == label_s:
                    Z_s += degree_s 
                    req = {match[N_s - Z_s + j] for j in range(degree_s)}
                    if req.issubset(children):
                        match[i_s] = i_o
                        Zs[i_m] = Z_s
            if len(self) - 1 <= i_o and label_o == self[N_s - 1][0]:
                matches.appendleft({N_s - 1: i_o})
                Zs.appendleft(0)
        result = []
        for m in matches:
            if len(m) == N_s:
                result.append(tuple(m[i] for i in range(N_s)))
        return tuple(result)

    @sig_cache
    def cut(self: Self, n: int, m: Sequence[int] = ()) -> tuple[Self, Self]:
        if n == 0 and not m:
            return super().__new__(type(self), [("", 0)]), self
        if n < 0 or len(self) <= n:
            raise ValidationError(f"Invalid node index '{n}' for key '{self}'")
        label, degree = self[n]
        for i in m:
            if i < 0 or degree <= i:
                raise ValidationError(f"Invalid child index '{i}'")
        if not m:
            m = list(range(degree))
        S = sum(d for _, d in self[:n])
        initial = [S + j + 1 for j in m]
        S, indices = S + degree, [*initial]
        l, r = [], []
        l.extend([*self[:n], (label, degree - len(initial))])
        r.append(("", len(initial)))
        for i, (label, degree) in enumerate(self[n + 1:], start=n + 1):
            seq = r if i in indices else l
            seq.append((label, degree))
            if i in indices:
                indices.extend([S + j + 1 for j in range(degree)])
            S += degree
        cls = type(self)
        return super().__new__(cls, l), super().__new__(cls, r)

    @sig_cache
    def link(self: Self, other: Self, n: int, m: Sequence[int] = ()) -> Self:
        if n < 0 or len(self) <= n:
            raise ValidationError(f"Invalid node index '{n}'")
        (label_s, degree_s), (_, degree_o) = self[n], other[0]
        label_n, degree_n = label_s, degree_s + degree_o
        if m and len(m) != degree_o:
            raise ValidationError(f"Invalid child index list")
        for i in m:
            if i < 0 or degree_n <= i:
                raise ValidationError(f"Invalid child index '{i}'")
        S, result = 0, []
        for label, degree in self[:n]:
            S += degree
            result.append((label, degree))
        if not m:
            m = [degree_s + j for j in range(degree_o)]
        indices = [S + j + 1 for j in m]
        S += degree_n
        result.append((label_n, degree_n))
        l, r = deque(self[n + 1:]), deque(other[1:])
        for i in range(n + 1, len(self) + len(other) - 1):
            node = r.popleft() if i in indices else l.popleft()
            _, degree = node
            if i in indices:
                indices.extend([S + j + 1 for j in range(degree)])
            result.append(node)
            S += degree
        return super().__new__(type(self), result)


@dataclass(frozen=True)
class KeyForm:
    __slots__ = ("k", "h")

    k: Key
    h: tuple[int, ...]

    def __post_init__(self):
        h = deque(self.h)
        for (_, degree) in self.k:
            if 0 < degree:
                continue
            try:
                h.popleft()
            except IndexError as e:
                raise ValueError("Height vector too short") from e

    def __contains__(self: Self, key: Key) -> bool:
        try:
            return key == self.reduce(key, None)
        except ValueError:
            return False
        
    def __eq__(self: Self, other) -> bool:
        if not isinstance(other, KeyForm):
            return NotImplemented
        return self.k == other.k and self.h == other.h

    def __le__(self: Self, other: Self) -> bool:
        matches = self.k.find_in(other.k)
        if len(matches) < 1:
            return False
        match, it_h_s, it_h_o = matches[0], iter(self.h), iter(other.h)
        S, depths, refs = 1, {}, {}
        for i, (label, degree) in enumerate(other.k):
            try:
                i_s = match.index(i)
            except ValueError:
                pass
            else:
                label_s, degree_s = self.k[i_s]
                assert label_s == label
                if degree_s == 0:
                    depths[i] = 0
                    refs[i] = next(it_h_s)
                elif degree < degree_s:
                    return False
            if i in depths:
                depths.update({S + j: depths[i] + 1 for j in range(degree)})
                refs.update({S + j: refs[i] for j in range(degree)})
                if degree == 0 and depths[i] + next(it_h_o) < refs[i]:
                    return False
            S += degree
        return True
    
    def __lt__(self: Self, other) -> bool:
        if not isinstance(other, KeyForm):
            return NotImplemented
        return self != other and self <= other

    @sig_cache
    def reduce(self, key: Key, branch: int | None) -> Key:
        matches = self.k.find_in(key)
        if not matches:
            raise ValueError(f"Key '{key}' does not match")
        if branch is None and 1 < len(matches):
            raise ValueError(
                f"Key '{key}' contains multiple matches but no branch given")
        match = matches[0 if branch is None else branch]
        S, result, it_h, trim, heights = 1, [], iter(self.h), set(), {}
        for i, (label, degree) in enumerate(key):
            try:
                i_s = match.index(i)
            except ValueError:
                pass
            else:
                degree_s = self.k[i_s][1]
                result.append((label, degree_s or degree))
                if degree_s == 0:
                    h = next(it_h) - 1
                    heights.update({S + j: h for j in range(degree)})
                    trim.update({S + j for j in range(degree)})
            if i in trim:
                h = heights[i]
                if h == 0:
                    result.append((label, 0))
                else:
                    if degree == 0:
                        raise ValueError(f"Expected node {i} to have height")
                    result.append((label, degree))
                    heights.update({S + j: h - 1 for j in range(degree)})
                    trim.update({S + j for j in range(degree)})
            S += degree
        return tuple.__new__(Key, result)
