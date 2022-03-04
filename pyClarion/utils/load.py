"""Interpreter for the pyClarion Explicit Knowledge Language."""


from contextlib import contextmanager
from dataclasses import dataclass, field
import re
from typing import Dict, List, Tuple, Iterable, Optional, Set, IO, Generator
from collections import OrderedDict, ChainMap, deque
from itertools import combinations

import pyClarion as cl
from pyClarion.components.utils import first


sign = r"\+|\-"
uint = r"\d+"
int_ = r"(?:\+|\-)?\d+"
float_ = r"(?:\+|\-)?(?:\d+|\.\d+|\d+.\d+)(?:e(?:\+|\-)?\d+)?"
term = r"\w+"
ref = r"{\*?\w+(?:#(?:\+|\-)?\d+)?}"
key = r"\w+="
pdelim, fdelim, dd = r"\/", r"#", r"\.\."
ellipsis = r"\.\.\."
name = fr"(?:{term}|{ref}|{sign})(?:\.?(?:{term}|{ref}|{sign}))*"
path = fr"(?:{name})(?:{pdelim}{name})*"
purepath = fr"(?:{term})(?:{pdelim}{term})*"
uri = fr"(?:{dd}{pdelim})?{path}(?:{fdelim}{path})?"
param = fr"{key}({float_}|{ref})"
literal = fr"(?:{uri}|{float_}|{param})" 
indent = r"'INDENT"
dedent = r"'DEDENT"


class CCMLError(RuntimeError):
    pass


@dataclass
class Token:
    t: str = ""
    i: str = ""
    l: int = 1
    d: Dict[str, str] = field(default_factory=dict)
    elts: List["Token"] = field(default_factory=list)


class IndentAnalyzer:
    def __init__(self):
        self.level = 0
        self.lineno = None
    
    def __call__(self, mo: re.Match) -> str:
        spaces, exprs = mo.groups()
        num = len(spaces)
        if num % 4: 
            raise CCMLError(f"Line {self.lineno}: Indent must be a multiple "
                "of 4 spaces")
        delta = (num // 4) - self.level
        self.level += delta
        if delta == 0:
            return exprs
        if delta == 1:
            return "\n".join([indent, exprs])
        if delta < 0:
            return "\n".join([dedent] * -delta + [exprs])
        else:
            raise CCMLError(f"Line {self.lineno}: Indent too deep")

    @contextmanager
    def at_line(self, lineno: int) -> Generator[None, None, None]:
        self.lineno = lineno
        yield
        self.lineno = None

    def close(self):
        while self.level:
            self.level -= 1
            yield dedent


class Tokenizer:
    # preprocessor patterns
    p_comment = r"(?:^|\s+)#(.*)"
    p_cmd_expr = ": (?! *\n)"
    p_cmd_trailing_space = r":\s*\Z"
    p_composite = "(.*:)(\n)(.+)"
    p_indent = r"^( *)(\S.*)"
    p_empty = r"^\s*"

    tokens = OrderedDict([
        ("INDENT", indent),
        ("DEDENT", dedent),
        ("ELLIPSIS", ellipsis),
        ("DATA", fr"(?P<data>{literal}(?: +{literal})*)"),
        ("CHUNK", fr"chunk(?: +(?P<chunk_id>{term}|{ref}))?:"),
        ("CTX", fr"ctx:"),
        ("SIG", fr"sig:"),
        ("FOR", fr"(?:for +(?P<mode>each|rotations|combinations(?= +k={uint}))(?:(?<=combinations) +k=(?P<k>{uint}))?):"),
        ("VAR", fr"var(?: +(?P<var_id>{term}))?:"),
        ("RULESET", fr"ruleset(?: +(?P<ruleset_id>{term}|{ref}))?:"),
        ("RULE", fr"rule(?: +(?P<rule_id>{term}|{ref}))?:"),
        ("CONC", fr"conc(?: +(?P<conc_id>{term}|{ref}))?:"),
        ("COND", fr"cond(?: +(?P<cond_id>{term}|{ref}))?:"),
        ("STORE", fr"store +(?P<store_id>{purepath}):")
    ])
    tok_re = re.compile(r"|".join(fr"(?P<{s}>{p})" for s, p in tokens.items()))

    def __call__(self, stream: IO) -> Generator[Token, None, None]:
        for lineno, logical_line in self.preprocess(stream):
            mo = self.tok_re.fullmatch(logical_line)
            if mo:
                t = mo.lastgroup
                args = re.fullmatch(self.tokens[t], mo[t]).groupdict() # type: ignore
                yield Token(t=f"'{t}", d=args, l=lineno) # type: ignore
            else:
                raise CCMLError(f"Line {lineno}: Invalid expression")

    def preprocess(self, stream: IO) -> Generator[Tuple[int, str], None, None]:
        indenter = IndentAnalyzer()
        for lineno, line in enumerate(stream, start=1):
            line = re.sub(self.p_comment, "", line) # type: ignore 
            line = re.sub(self.p_cmd_expr, ":\n", line)
            line = re.sub(self.p_cmd_trailing_space, ":", line)
            line = re.sub(self.p_composite, self.delimit_composite_line, line)
            with indenter.at_line(lineno):
                line = re.sub(self.p_indent, indenter, line)
            line = re.sub(self.p_empty, "", line)
            for logical_line in line.splitlines():
                yield lineno, logical_line.strip()
        else:
            for dedent in indenter.close(): 
                yield lineno, dedent # type: ignore 

    def delimit_composite_line(self, mo: re.Match) -> str:
        pre, delim, post = mo.groups()
        repr(post)
        if delim:
            res = "\n".join([fr"{pre}", r"'INDENT", post, r"'DEDENT"])
            return res
        else:
            return pre


class Parser:
    terminals = ["'DATA", "'ELLIPSIS"]
    grammar = {
        r"'ROOT": r"(?:'STORE|'VAR)*",
        r"'STORE": r"(?:'CHUNK|'RULE|'RULESET|'CTX|'SIG|'VAR|'FOR)*",
        r"'CHUNK": r"(?:(?:'DATA|'FOR)+|'ELLIPSIS)",
        r"'RULE": r"'CONC(?:'COND|'FOR)+",
        r"'CONC": r"(?:(?:'DATA|'FOR)+|'ELLIPSIS)",
        r"'COND": r"(?:(?:'DATA|'FOR)+|'ELLIPSIS)",
        r"'RULESET": r"(?:'RULE|'VAR|'CTX|'FOR|'SIG)+",
        r"'SIG": r"(?:'DATA|'FOR)+", 
        r"'VAR": r"(?:'DATA|'FOR)+",
        (r"'CTX", r"∅"): r"(?:'CHUNK|'RULE|'RULESET|'CTX|'SIG|'VAR|'FOR)+",
        (r"'CTX", r"s"): r"(?:'RULE|'CTX|'SIG|'VAR|'FOR)+",
        (r"'FOR", r"∅"): r"(?:'VAR)+(?:'CHUNK|'RULE|'RULESET|'CTX|'SIG|'FOR)+",
        (r"'FOR", r"s"): r"(?:'VAR)+(?:'RULE|'CTX|'SIG|'FOR)+",
        (r"'FOR", r"r"): r"(?:'VAR)+(?:'COND|'FOR)+",
        (r"'FOR", r"c"): r"(?:'VAR)+(?:'DATA|'FOR)+",
        (r"'FOR", r"l"): r"(?:'VAR)+(?:'DATA|'FOR)+",
    }
    index_updates = {
        r"'CHUNK": r"c",
        r"'CONC": r"c",
        r"'COND": r"c",
        r"'RULE": r"r",
        r"'RULESET": r"s",
        r"'SIG": r"c",
        r"'VAR": r"l"
    }

    def __call__(self, stream: Iterable[Token]) -> Token:
        stack, current = [], Token(t="'ROOT", i="∅")
        for token in stream:
            current = self.build_tree(stack, current, token)
        assert current.t == "'ROOT"
        self.check_grammar(current)
        return current

    def build_tree(
        self, stack: List[Token], current: Token, token: Token
    ) -> Token:
        t = token.t
        if t == "'INDENT":
            stack.append(current)
            return current.elts[-1]
        elif t == "'DEDENT":
            self.check_grammar(current)
            assert stack
            return stack.pop()
        else:
            token.i = self.index_updates.get(current.t, current.i)
            current.elts.append(token)
            return current
    
    def check_grammar(self, tok: Token) -> None:
        seq = "".join(elt.t for elt in tok.elts)
        pat = (self.grammar[tok.t] if tok.t in self.grammar 
            else self.grammar[(tok.t, tok.i)]) # type: ignore
        if not re.fullmatch(pat, seq):
            raise CCMLError(f"Line {tok.l}: Syntax error "
                f"in {tok.t[1:]} block")
        for elt in tok.elts:
            if elt.t not in self.terminals: 
                self.check_grammar(elt)


NumDict = cl.nd.NumDict
@dataclass
class Load:
    address: str
    cs: List[cl.chunk] = field(default_factory=list)
    rs: List[cl.rule] = field(default_factory=list)
    fs: NumDict[Tuple[cl.chunk, cl.feature]] = field(default_factory=NumDict)
    ws: NumDict[Tuple[cl.chunk, cl.dimension]] = field(default_factory=NumDict)
    cr: NumDict[Tuple[cl.chunk, cl.rule]] = field(default_factory=NumDict)
    rc: NumDict[Tuple[cl.rule, cl.chunk]] = field(default_factory=NumDict)

    @property
    def wn(self) -> NumDict[cl.chunk]:
        return self.ws.abs().sum_by(kf=first)


@dataclass
class Context:
    loaded: List[Load] = field(default_factory=list)
    load: Optional[Load] = None
    lstack: List[str] = field(default_factory=list)
    fstack: List[Tuple[cl.feature, Optional[float]]] = field(default_factory=list)
    fdelims: List[int] = field(default_factory=list)
    fspace: Optional[Set[cl.feature]] = None
    frames: ChainMap = field(default_factory=ChainMap)
    vstack: List[str] = field(default_factory=list)
    for_vars: list = field(default_factory=list)
    for_index: List[str] = field(default_factory=list)
    var_id: str = ""
    lineno: List[str] = field(default_factory=list)
    _ref: str = r"{(?P<level>\*)?(?P<id>\w+)(?:#(?P<index>(?:\+|\-)?\d+))?}"

    @property
    def vdata(self):
        assert self.var_id in self.frames
        return self.frames[self.var_id]

    @property
    def l(self):
        return self.lineno[-1].lstrip("0")

    @contextmanager
    def fspace_scope(self, fspace: Optional[Set[cl.feature]]) -> Generator[None, None, None]:
        self.fspace = fspace
        yield
        self.fspace = None

    @contextmanager
    def feature_scope(self) -> Generator[None, None, None]:
        self.fdelims.append(len(self.fstack))
        yield
        self.fstack = self.fstack[:self.fdelims.pop()]

    @contextmanager
    def label_scope(self, lineno: int, lbl: str) -> Generator[None, None, None]:
        with self.at_line(lineno):
            self.lstack.append(self.deref(lineno, lbl))
            yield
            self.lstack.pop()

    @contextmanager
    def ruleset_scope(self, lineno: int, lbl: str) -> Generator[None, None, None]:
        with self.label_scope(lineno, lbl):
            with self.var_frame():
                with self.feature_scope():
                    yield

    @contextmanager
    def store_scope(self, store_id: str) -> Generator[None, None, None]:
        self.load = Load(store_id)
        with self.var_frame(): 
            with self.feature_scope():
                yield
        self.loaded.append(self.load)
        self.load = None
        
    @contextmanager
    def var_scope(self, lineno: int, var_id: str) -> Generator[None, None, None]:
        if var_id in self.frames.maps[0]:
            raise CCMLError(f"Line {lineno}: Var '{var_id}' already defined "
                f"in current scope")
        self.vstack.append(self.var_id)
        self.frames.maps[0][var_id] = []
        self.var_id = var_id
        yield
        self.var_id = self.vstack.pop()

    @contextmanager
    def var_frame(self) -> Generator[None, None, None]:
        self.frames = self.frames.new_child()
        yield
        self.frames = self.frames.parents
    
    @contextmanager
    def for_scope(self):
        self.for_index.append("")
        with self.var_frame():
            yield
        self.for_index.pop()

    @contextmanager
    def at_line(self, lineno: int) -> Generator[None, None, None]:
        self.lineno.append(str(lineno).zfill(4))
        yield
        self.lineno.pop()

    def gen_uri(self) -> str:
        assert self.load
        coords = ".".join([self.lineno[-1], *self.for_index])
        fragment = "/".join(filter(None, [coords, *self.lstack]))
        return "#".join(filter(None, [self.load.address, fragment]))

    def deref(self, lineno: int, literal: str) -> str:
        with self.at_line(lineno):
            return re.sub(self._ref, self._deref, literal)

    def _deref(self, mo: re.Match) -> str:
        try: 
            data = self.frames[mo["id"]]
        except KeyError as e: 
            raise CCMLError(f"Line {self.l}: Undefined reference") from e
        if mo["index"]:
            try: 
                data = data[int(mo["index"])]
            except IndexError as e: 
                raise CCMLError(f"Line {self.l}: Index out of bounds") from e
        if mo["level"]:
            if isinstance(data, str):
                mo2 = re.fullmatch(self._ref, f"{{{data}}}")
                if mo2: 
                    return self._deref(mo2)
                else: 
                    raise CCMLError(f"Line {self.l}: Invalid reference")
            else: 
                raise CCMLError(f"Line {self.l}: Invalid list reference")
        if isinstance(data, list): 
            data = " ".join(data)
        return data


class Interpreter:
    data_patterns = {
        r"c": (fr"(?P<d>{uri})(?: +(?P<v>{uri}))?(?: +l=(?P<l>{int_}|{ref}))?"
            fr"(?: +w=(?P<w>{float_}|{ref}))?"),
        r"l": fr"(?P<data>{literal}(?: +{literal})*)",
    }
 
    def __init__(self, structure: Optional[cl.Structure] = None):
        self.structure = structure
        self.dispatcher = {
            (r"'DATA", r"c"): self.feature,
            (r"'DATA", r"l"): self.list_,
            r"'ELLIPSIS": self.ellipsis,
            r"'CHUNK": self.chunk,
            r"'CONC": self.conc,
            r"'COND": self.cond,
            r"'RULE": self.rule,
            r"'RULESET": self.ruleset,
            r"'STORE": self.store,
            r"'CTX": self.ctx_,
            r"'SIG": self.sig,
            r"'VAR": self.var,
            r"'FOR": self.for_,
        }

    def __call__(self, ast):
        ctx = Context()
        fspace = (set(cl.inspect.fspace(self.structure)) 
            if self.structure is not None else None)
        with ctx.fspace_scope(fspace):
            self.dispatch(ast.elts, ctx)
        return ctx.loaded

    def dispatch(self, stream, ctx):
        for node in stream:
            func = (self.dispatcher[node.t] if node.t in self.dispatcher 
                else self.dispatcher[(node.t, node.i)])
            func(node, ctx)

    def feature(self, tok: Token, ctx: Context) -> None:
        args = self.parse_data(tok)
        d, v, l, w = [ctx.deref(tok.l, args[k] or "") for k in "dvlw"]
        assert d
        if v == "": 
            v = None
        else:
            try: v = int(v)
            except ValueError: pass
        l = int(l) if l else 0
        w = float(w) if w != "" else None
        f = cl.feature(d, v, l)
        if ctx.fspace is not None and f not in ctx.fspace:
            raise CCMLError(f"Line {tok.l}: {f} not a member of working "
                "feature space")
        ctx.fstack.append((f, w))

    def parse_data(self, tok: Token):
        mo = re.fullmatch(self.data_patterns[tok.i], tok.d["data"])
        if mo: 
            args = mo.groupdict()
        else:
            raise CCMLError(f"Line {tok.l}: Invalid feature expression")
        return args

    def ellipsis(self, tok: Token, ctx: Context) -> None:
        pass

    def chunk(self, tok: Token, ctx: Context) -> None:
        with ctx.label_scope(tok.l, tok.d["chunk_id"] or ""):
            with ctx.feature_scope():
                self.dispatch(tok.elts, ctx)
                self.load_chunk(tok, ctx)

    @staticmethod
    def load_chunk(tok: Token, ctx: Context, noctx=False) -> None:
        assert ctx.load
        c = cl.chunk(ctx.gen_uri())
        ctx.load.cs.append(c)
        fdata, dims, ws = ctx.fstack, [], {}
        if noctx: fdata = fdata[ctx.fdelims[-1]:]
        for f, w in fdata:
            ctx.load.fs[(c, f)] = 1.0
            dims.append(f.dim)
            if w is not None: 
                if f.dim in ws:
                    raise CCMLError(f"Line {tok.l}: Ambiguous weight "
                        f"specification in {tok.t} block")
                else:
                    ws[f.dim] = w
        for dim in dims:
            ctx.load.ws[(c, dim)] = ws.get(dim, 1.0)

    def conc(self, tok: Token, ctx: Context) -> None:
        with ctx.label_scope(tok.l, tok.d["conc_id"] or ""):
            with ctx.feature_scope():
                self.dispatch(tok.elts, ctx)
                self.load_chunk(tok, ctx, noctx=True)
        assert ctx.load
        ctx.load.rc[(ctx.load.rs[-1], ctx.load.cs[-1])] = 1.0

    def cond(self, tok: Token, ctx: Context) -> None:
        with ctx.label_scope(tok.l, tok.d["cond_id"] or ""):
            with ctx.feature_scope():
                self.dispatch(tok.elts, ctx)
                self.load_chunk(tok, ctx, noctx=False)
        assert ctx.load
        ctx.load.cr[(ctx.load.cs[-1], ctx.load.rs[-1])] = 1.0

    def rule(self, tok: Token, ctx: Context) -> None:
        assert ctx.load
        with ctx.label_scope(tok.l, tok.d["rule_id"] or ""):
            ctx.load.rs.append(cl.rule(ctx.gen_uri()))
            self.dispatch(tok.elts, ctx)
    
    def ruleset(self, tok: Token, ctx: Context) -> None:
        with ctx.ruleset_scope(tok.l, tok.d["ruleset_id"] or ""):
            self.dispatch(tok.elts, ctx)

    def store(self, tok: Token, ctx: Context) -> None:
        store_id = tok.d["store_id"]
        if self.structure is not None:
            try:
                store = self.structure[store_id]
            except KeyError as e:
                raise CCMLError(f"Line {tok.l}: No module at '{store_id}' "
                    "in working structure") from e
            else:
                if not isinstance(store, cl.Module):
                    raise CCMLError(f"Line {tok.l}: Expected Module "
                        f"instance at '{store_id}', found "
                        f"{store.__class__.__name__} instead")
                elif not isinstance(store.process, cl.Store):
                    raise CCMLError(f"Line {tok.l}: Expected process of "
                        f"type Store at '{store_id}', found "
                        f"{store.__class__.__name__} instead")
        with ctx.store_scope(store_id):
            self.dispatch(tok.elts, ctx)

    def ctx_(self, tok: Token, ctx: Context) -> None:
        with ctx.feature_scope():
            self.dispatch(tok.elts, ctx)

    def sig(self, tok: Token, ctx: Context) -> None:
        self.dispatch(tok.elts, ctx)

    def list_(self, tok: Token, ctx: Context) -> None:
        args = self.parse_data(tok)
        _ldata = ctx.deref(tok.l, args["data"])
        _ldata = re.sub(" +", " ", _ldata.strip())
        ldata = _ldata.split(" ")
        ctx.vdata.extend(ldata)

    def var(self, tok: Token, ctx: Context) -> None:
        var_id = tok.d["var_id"]
        with ctx.var_scope(tok.l, var_id):
            self.dispatch(tok.elts, ctx)
    
    def for_(self, tok: Token, ctx: Context) -> None:
        index = 0
        for elt in tok.elts: 
            if elt.t == r"'VAR": index += 1
            else: break
        with ctx.for_scope():
            self.dispatch(tok.elts[:index], ctx)
            for _ in self._iter(tok, ctx):
                if tok.i in ["∅", "s"]:
                    with ctx.feature_scope():
                        self.dispatch(tok.elts[index:], ctx)
                else:
                    assert tok.i in ["r", "c", "l"]
                    self.dispatch(tok.elts[index:], ctx)

    def _iter(self, tok: Token, ctx: Context):
        vars_, seqs = list(zip(*ctx.frames.maps[0].items()))
        lens = set(len(seq) for seq in seqs)
        if len(lens) != 1:
            raise CCMLError(f"Line {tok.l}: Iterated var lists must all "
                f"have the same length")
        else:
            n, = lens
        with ctx.var_frame():
            if tok.d["mode"] == r"each":
                for i in range(n):
                    for j, k in enumerate(vars_):
                        ctx.frames[k] = seqs[j][i]
                        ctx.for_index[-1] = str(i).zfill(2)
                    yield
            elif tok.d["mode"] == r"rotations":
                deq = deque(range(n))
                for i in range(n):
                    for j, k in enumerate(vars_):
                        ctx.frames[k] = [seqs[j][_i] for _i in deq]
                        ctx.for_index[-1] = str(i).zfill(2)
                        deq.rotate(-1)
                    yield
            else:
                assert tok.d["mode"] == r"combinations"
                k = int(tok.d["k"])
                for i, (i1, i2) in enumerate(combinations(range(n), k)):
                    for j, k in enumerate(vars_):
                        ctx.frames[k] = [seqs[j][i1], seqs[j][i2]]
                        ctx.for_index[-1] = str(i).zfill(2)
                    yield
    

def load(f: IO, structure: cl.Structure) -> None:
    t, p, i = Tokenizer(), Parser(), Interpreter(structure)
    for _load in i(p(t(f))):
        assert _load.address in structure
        module = structure[_load.address]
        store = module.process
        assert isinstance(store, cl.Store)
        p = module.inputs[0][1]() # pull parameters from parameter module
        # Populate store
        store.cf = _load.fs
        store.cw = _load.ws
        store.wn = _load.wn
        store.cr = _load.cr
        store.rc = _load.rc
        if store.cb is not None:
            store.cb.update(p, cl.nd.NumDict({c: 1 for c in _load.cs}))
        if store.rb is not None:
            store.rb.update(p, cl.nd.NumDict({r: 1 for r in _load.rs}))    
