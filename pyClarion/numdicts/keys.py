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
        if not isinstance(s, str):
            raise TypeError(f"Expected str or {cls.__name__}, got "
                f"{type(s).__name__} instead""")
        if any(c.isspace() for c in s):
            raise ValidationError("No spaces allowed")
        if s == "":
            return super().__new__(cls, (("", 0),))
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
            if len(groups) == 1 and m.string == "":
                raise ValidationError("Invalid key string")
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
    
    def __bool__(self) -> bool:
        return self != Key()

    def __lt__(self, other) -> bool:
        if not isinstance(other, Key):
            return NotImplemented
        if tuple(self) == tuple(other):
            return False
        return self <= other

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
    def find_in(self: Self, other: "Key", wc: str | None = None) \
        -> Sequence[tuple[int, ...]]:
        N_s, N_o = len(self), len(other)
        if N_o < N_s:
            return ()
        if other.height < self.height:
            return ()
        matches = []
        for i_s in range(N_s - 1, -1, -1):
            new_matches = []
            for i_o in range(N_o - N_s + i_s, i_s - 1, -1):
                (l_s, d_s), (l_o, d_o) = self[i_s], other[i_o]
                if l_s != wc and l_o != l_s:
                    continue
                if i_s == N_s - 1:
                    new_matches.append({i_s: i_o})
                    continue
                S = sum(deg for _, deg in other[:i_o])
                children = {S + j + 1 for j in range(d_o)}
                for m in matches:
                    if m[i_s + 1] <= i_o:
                        continue
                    if len(children.intersection(m.values())) != d_s:
                        continue
                    new_matches.append({i_s: i_o, **m})
            matches = new_matches
        result = []
        for m in matches:
            if m[0] == 0:
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
            
    def __contains__(self, obj) -> bool:
        if not isinstance(obj, Key):
            return NotImplemented
        ref = self.as_key()
        if len(obj) != len(ref):
            return False
        return bool(ref.find_in(obj, wc="?"))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, KeyForm):
            return NotImplemented
        return self.k == other.k and self.h == other.h

    def __lt__(self, other) -> bool:
        return self != other and self <= other

    def __le__(self, other) -> bool:
        if not isinstance(other, KeyForm):
            return NotImplemented
        k1 = self.as_key(); k2 = other.as_key()
        return bool(k1.find_in(k2, wc="?"))

    def reductor(self, other: "KeyForm", b: int | None = None) \
        -> Callable[[Key], Key]:
        k1 = self.as_key(); k2 = other.as_key()
        if not self <= other:
            raise ValueError(f"Keyform {k1} cannot match keys from {k2}")
        matches = k1.find_in(k2, wc="?")
        if 1 < len(matches) and b is None:
            raise ValueError(f"Keyform {k1} has multiple matches to {k2}")
        indices = matches[0] if b is None else matches[b]
        cuts = []; S = 1
        for i, (_, deg) in enumerate(k2):
            m = tuple(j for j in range(deg) if S + j not in indices)
            if m:
                cuts.append((i, m))
            S += deg 
        def reduce(key: Key) -> Key:
            for i, m in reversed(cuts):
                key, _ = key.cut(i, m)
            return key
        return reduce

    @sig_cache
    def as_key(self) -> Key:
        k = self.k; it = reversed(self.h)
        for i in range(len(k) - 1, -1, -1):
            if k[i][1] == 0:
                k = k.link(Key(":".join(["?"] * next(it))), i)
        return k
    
    @classmethod
    @sig_cache
    def from_key(cls: Type[Self], key: Key) -> Self:
        leaves, indices, hs, S = [], {}, {}, 1
        for i, (label, deg) in enumerate(key):
            children = [key[S + j] for j in range(deg)]
            if not (i == 0 or label.isidentifier() or label == "?"):
                raise ValidationError(f"Unexpected label {repr(label)} in key, "
                    "label must be a valid python identifier or '?'.")
            elif label.isidentifier() and deg == 0:
                leaves.append(i)
                indices[i] = i
                hs[i] = 0
            elif label.isidentifier() and any(lb == "?" for lb, _ in children):
                if 1 < len(children):
                    raise ValidationError("Wildcard '?' cannot have siblings")
                leaves.append(i)
                indices[i] = i
                hs[i] = 0
                for j in range(deg):
                    indices[S + j] = indices[i]
            elif label == "?":
                if 1 < len(children):
                    raise ValidationError("Wildcard '?' can have at most one "
                        "child node")
                if any(lb != "?" for lb, _ in children):
                    raise ValueError("Children of wildcard nodes must also be "
                        "wildcard nodes.")
                hs[indices[i]] += 1
                for j in range(deg):
                    indices[S + j] = indices[i]
            S += deg
        ref = key
        for i in reversed(leaves):
            ref, _ = ref.cut(i)
        return cls(ref, tuple(hs[indices[i]] for i in leaves))