from typing import (List, Type, IO, Optional, ClassVar, TypeVar, NamedTuple, 
    Dict, Tuple)
from dataclasses import dataclass, field
import re

from .exc import CCMLSyntaxError
from .primitives import AST
from .semantics import ROOT


@dataclass
class Line:
    text: str
    lineno: int
    indentation: int
    continued: bool
    body: str
    comment: Optional[str]

    pattern: ClassVar[str] = (
        r"^( *)(?:([^\\\s#][^\\#]*?)\s*(?:(\\)\s*)?)?(?:[#](.*))?$\n?")
    _pattern_re: ClassVar[re.Pattern] = re.compile(pattern)

    @classmethod
    def from_text(cls, lineno: int, text: str) -> "Line":
        m = cls._pattern_re.fullmatch(text)
        if m is None: 
            raise CCMLSyntaxError(f"Invalid '\\' on line {lineno}.")
        spaces, body, linebreak, comment = m.groups()
        indentation, continued = len(spaces), linebreak is not None
        return cls(text, lineno, indentation, continued, body, comment)


E = TypeVar("E", bound="Expr")
class Expr(NamedTuple):
    body: str 
    lineno: int
    indentation: int 
    lines: List[Line]
    
    @classmethod
    def from_lines(cls: Type[E], first: Line, *rest: Line) -> E:
        body = " ".join([first.body, *(line.body for line in rest)])
        return cls(body, first.lineno, first.indentation, [first, *rest])


@dataclass
class Parser:
    lineno : int = 0
    indentation: int = 0
    delta: int = 0
    node: AST = field(init=False)
    lines: List[Line] = field(default_factory=list)
    ops: List[Type[AST]] = field(default_factory=AST.discover)

    _ops: Dict[str, Tuple[re.Pattern, Type[AST]]] = field(init=False)
    _kw_re: ClassVar[re.Pattern] = re.compile(r"(\S+).*")
    
    def __post_init__(self):
        self._ops = {op.__name__.lower(): 
            (re.compile(self._data_pat_gen(op)), op) 
            for op in self.ops}

    def parse(self, stream: IO) -> ROOT:
        root = ROOT(0, "", "")
        self.node = root
        for s in stream: 
            self._step(s)
        self._finalize()
        assert self.node is root
        return self.node

    def _step(self, text: str) -> None:
        self.lineno = self.lineno + 1
        line = Line.from_text(self.lineno, text)
        if line.body is not None: 
            self.lines.append(line)
            if not line.continued: 
                lines, self.lines = self.lines, []
                expr = Expr.from_lines(*lines)
                self._process_indentation(expr)
                self._update_node_pointer()
                self._add_new_ast_node(expr)

    def _finalize(self) -> None:
        for _ in range(-self.indentation // 2, 0):
            assert self.node.parent is not None
            self.node = self.node.parent

    def _process_indentation(self, expr: Expr):
        delta = expr.indentation - self.indentation
        self.indentation = expr.indentation
        if (2 < delta or delta % 2 != 0 
            or (delta == 2 and len(self.node.children) == 0)): 
            raise CCMLSyntaxError(f"Indentation error on line {expr.lineno}")
        self.delta = delta

    def _update_node_pointer(self):
        if self.delta == 0:
            pass
        elif self.delta == 2:
            assert 0 < len(self.node.children)
            self.node = self.node.children[-1]
        else:
            assert self.delta < 0 and self.delta % 2 == 0
            for _ in range(self.delta // 2 + 1, 0):
                assert self.node.parent is not None
                self.node = self.node.parent
            assert self.node.parent is not None
            self.node = self.node.parent

    def _add_new_ast_node(self, expr: Expr) -> None:
        kw = self._kw_re.fullmatch(expr.body)
        assert kw is not None
        try:
            pat, op = self._ops[kw.group(1)]
        except KeyError:
            raise CCMLSyntaxError(f"Unexpected func name '{kw.group(1)}' "
               f"on line {expr.lineno}") from None
        data = pat.fullmatch(expr.body)
        if data is None:
            raise CCMLSyntaxError(f"Invalid expr on line {expr.lineno}")
        self.node.spawn(op, expr.lineno, *data.groups()[1:])

    def _data_pat_gen(self, t: Type[AST]) -> str:
        return (fr"({t.__name__.lower()})"
            r"((?:\s+[^\s=]+(?!=))*)((?:\s+[^\s=]+=[^\s=]+)*)")
