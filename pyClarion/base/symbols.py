from typing import NamedTuple, Tuple, TypeVar, Generic, Dict, Union, cast
from typing_extensions import Literal
import re


spc = r"\s+"
tok = r"[a-zA-Z_]+\w*"
dig = r"[0-9]+"
itg = fr"[-+]?{dig}"
frk = fr"(?:{itg}(?:[/]{dig})?)"
adr = fr"(?:{tok}(?:/{tok})*)"
var = fr"(?:@{tok})"
seq = fr"(?:\w+(?:[-]\w+)*)"
val = fr"(?:{seq}|{frk})"
vls = fr"(?:{val}|{{{val}(?:,{val})*}})"
dvs = fr"(?:{seq}(?:[:]{vls})?)"
mrk = fr"(?:|{adr}(?:[.](?:{dig}|{seq}))?(?:[:]{adr}(?:[.](?:{dig}|{seq}))?)*)"
pth = fr"(?:|{adr})"

loc = fr"(?P<p>{adr})(?:[.](?P<s>{dig}))?"
dvp = fr"(?P<d>{seq})(?:[:](?P<v>{val}))?"

RE = {
    "spc": re.compile(spc),
    "tok": re.compile(tok),
    "dig": re.compile(dig),
    "itg": re.compile(itg),
    "frk": re.compile(frk),
    "adr": re.compile(adr),
    "var": re.compile(var),
    "seq": re.compile(seq),
    "val": re.compile(val),
    "vls": re.compile(vls),
    "dvs": re.compile(dvs),
    "mrk": re.compile(mrk),
    "pth": re.compile(pth),
    "loc": re.compile(loc),
    "dvp": re.compile(dvp)
}


Pat = Literal["spc", "tok", "dig", "itg", "frk", "adr", "var", "seq", "val", 
    "vls", "dvs", "mrk", "pth"]
Sym = Literal["dvp", "loc"]


def match(s: str, ref: Union[Pat, Sym]) -> bool:
    """Return True if s is a path."""
    return bool(RE[ref].fullmatch(s))


def parse(s: str, ref: Sym) -> Dict[str, str]:
    m = RE[ref].fullmatch(s)
    if m is None: 
        raise ValueError(f"Str '{s}' is not a valid {ref}.")
    return m.groupdict()


def make_loc(s: int, p: str) -> str:
    if not match(p, "adr"):
        raise ValueError(f"Invalid path '{p}'")
    else:
        return f"{p}.{s}"
    

T = TypeVar("T")
X = TypeVar("X", bound="Symbol")
class Symbol(NamedTuple, Generic[T]):
    i: T
    l: int = 0
    m: str = ""
    p: str = ""

    def lag(self: X, val: int = 1) -> X:
        return type(self)(self.i, self.l + val, self.m, self.p)

    def mark(self: X, mrk: str) -> X:
        return type(self)(self.i, self.l, ".".join([self.m, mrk]), self.p)

    @classmethod
    def validate(cls, i: T, l: int = 0, m: str = "", p: str = "") -> None:
        if m and not match(m, "mrk"):
            raise ValueError(f"Value '{m}' is not a valid marker.")
        if p and not match(p, "adr"):
            raise ValueError(f"Value '{p}' is not a valid path.")


DV = TypeVar("DV", str, Tuple[str, str])
class F(Symbol, Generic[DV]): 
    """A feature symbol."""  
    i: DV

    @property
    def dim(self) -> "D":
        if isinstance(self.i, str):
            return cast(D, self)
        else:
            return F(self.i[0], self.l, self.m, self.p)

    @classmethod
    def validate(cls, i: DV, l: int = 0, m: str = "", p: str = "") -> None:
        _i = ":".join(i) if isinstance(i, tuple) else i
        if not match(_i, ref="dvp"):
            raise ValueError(f"Value '{_i}' is not a valid dim or "
                "dim-val pair.")
        super().validate(i, l, m, p)

D = F[str]
V = F[Tuple[str, str]]


class C(Symbol): 
    """A chunk symbol."""
    i: str

    @classmethod
    def validate(cls, i: str, l: int = 0, m: str = "", p: str = "") -> None:
        if not match(i, ref="seq"):
            raise ValueError(f"Value '{i}' is not a valid token sequence.")
        super().validate(i, l, m, p)


class R(Symbol):
    """A rule symbol.""" 
    i: str

    @classmethod
    def validate(cls, i: str, l: int = 0, m: str = "", p: str = "") -> None:
        if not match(i, ref="seq"):
            raise ValueError(f"Value '{i}' is not a valid token sequence.")
        super().validate(i, l, m, p)
