from __future__ import annotations
import re
from typing import Dict, Tuple, List, Union, Any
from typing_extensions import TypedDict
from contextlib import contextmanager
import enum

from ..base import uris, dimension, feature, chunk, rule, Structure, Module
from ..components import Store
from ..components.utils import first
from .. import numdicts as nd
from . import inspect


__all__ = ["load"]


class ParseError(Exception):
    ...


class ChunkSpec(TypedDict):
    fs: List[Tuple[str, Union[str, None], int]]
    ws: Dict[Tuple[str, int], float]


class RuleSpecS1(TypedDict):
    conds: List[Tuple[str, ChunkSpec, float]]
    conc: Tuple[str, ChunkSpec, float]


class RuleSpecS2(TypedDict):
    conds: List[Tuple[str, float]]
    conc: Tuple[str, float]


class IterContext(enum.Enum):

    NONE = enum.auto()
    VAR = enum.auto()
    CHUNK = enum.auto()
    RULE = enum.auto()
    RULESET = enum.auto()


class Parser:

    # Tokenization regexps
    comment = re.compile("(^\#)|(\s\#)")
    empty = re.compile("\s*")
    kw_exp = re.compile("\s*\w.*:\s*[\w/+#\-.(){}\s]*")
    data_exp = re.compile("\s*[\w/+#\-.(){}\s]*")

    # Parsing regexps
    c_expr = re.compile("chunk ?(?P<id>[\w/]*)?") 
    cf_expr = re.compile(
        "(?P<d>[\w/#.]+) ?(?P<v>[\w/#+-]+)? ?(?P<l>[-+\d]*)? ?(\((?P<w>[\w.+-]+)\))?") 
    r_expr = re.compile("rule ?(?P<id>[\w/]*)?") 
    r_subexpr = re.compile(
        "(?P<type>cond|conc) ?(?P<id>[\w/]*) ?(\((?P<w>[\w.+-]+)\))?") 
    ruleset_expr = re.compile("ruleset (?P<id>\w+)") 
    ctx_expr = re.compile("ctx ?(?P<cmd>expand|contract)?")
    iter_expr = re.compile("for each (?P<vars>\w+( \w+)*)") 
    var_expr = re.compile("var (?P<id>[\w/]*)")
    var_data_expr = re.compile(".*") 
    # NOTE: Wildcard match for var_data_expr may cause problems. Makes dispatch 
    #   key order significant. Should be cleaned up at some point.

    _r_prefix = "r"
    _c_prefix = "c"

    def __init__(self, prefix, fspace=None):
        self.prefix = prefix
        self.fspace = fspace
        self._dispatcher = {
            "empty": (self.empty, self._tokenize_empty),
            "kw_exp": (self.kw_exp, self._tokenize_kw_exp),
            "data_exp": (self.data_exp, self._tokenize_data_exp),
            "cf_expr": (self.cf_expr, self._parse_cf_expr),
            "c_expr": (self.c_expr, self._parse_c_expr),
            "r_expr": (self.r_expr, self._parse_r_expr),
            "r_subexpr": (self.r_subexpr, self._parse_r_subexpr),
            "ruleset_expr": (self.ruleset_expr, self._parse_ruleset_expr),
            "ctx_expr": (self.ctx_expr, self._parse_ctx_expr),
            "iter_expr": (self.iter_expr, self._parse_iter_expr),
            "var_expr": (self.var_expr, self._parse_var_expr),
            "var_data_expr": (self.var_data_expr, self._parse_var_data_expr)
        }

    def parse(self, f):
        self._init_parse()
        self._tokenize(f)
        self._pass_1()
        self._pass_2()
        return self._prepare_output()

    def _init_parse(self):

        # Tokenizer data
        self._indent_level = 0
        self._super = []
        self._data = []
        self._current: List[Any] = self._data
        self._prev = []

        # Parser data
        self._chunks_s1: Dict[str, List[ChunkSpec]] = {}
        self._chunks_s2: Dict[str, ChunkSpec] = {}
        self._rules_s1: Dict[Tuple[str, str, str], List[RuleSpecS1]] = {}
        self._rules_s2: Dict[str, RuleSpecS2] = {}
        self._fspecs: list = []
        self._rid: str = ""
        self._rspec: Dict = {}
        self._rulesets: List[str] = []
        self._ruleset_id: str = ""
        self._rnames: List[str] = []
        self._ctx: List[str] = []
        self._ctx_delims: List[int] = []
        self._ctx_level = []
        self._ctx_ruleset_level: int = 0
        self._ctx_iter_level: List[int] = []
        self._iter_ctx: List[IterContext] = [IterContext.NONE]
        self._iter_vars: Dict[str, Any] = {}
        self._var_data: List[Tuple[int, str]] = []

    #####################
    # Tokenizer Methods #
    #####################

    def _tokenize(self, f):
        for i, line in enumerate(f):
            line = self.comment.split(line, maxsplit=1)[0] # remove comments
            for k in ["empty", "kw_exp", "data_exp"]:
                exp, handler = self._dispatcher[k]
                m = exp.fullmatch(line)
                if m is not None:
                    handler(i, line)
                    break
            else:
                raise ParseError(
                    f"Invalid syntax '{line.strip()}' on line {i + 1}")
        return self._data            

    def _tokenize_empty(self, i, line) -> None:
        pass

    def _tokenize_kw_exp(self, i: int, line: str) -> None:
        self._update_indent_level(i, line)
        kw_exp = line.strip()
        head, tail = kw_exp.split(":")
        head, tail = head.strip(), [(i, tail.strip())]
        data = {(i, head): tail}
        self._current.append(data)
        self._prev = tail
        
    def _tokenize_data_exp(self, i, line) -> None:
        self._update_indent_level(i, line)
        data = (i, line.strip())
        self._current.append(data)

    def _update_indent_level(self, i, line):
        lspaces = len(line) - len(line.lstrip())
        if lspaces % 4:
            raise ParseError(f"Indentation error on line {i + 1}")
        indent_diff = (lspaces // 4) - self._indent_level
        self._indent_level += indent_diff
        if indent_diff == 0:
            pass
        elif indent_diff == 1:
            self._super.append(self._current)
            self._current = self._prev
        elif indent_diff < 0:
            self._current = self._super[indent_diff]
            self._super = self._super[:indent_diff]
        else:
            raise ParseError(f"Indentation error on line {i + 1}")

    ##############
    # First Pass #
    ##############

    def _pass_1(self):
        for block in self._data:
            self._parser_dispatch(0, block, ["c_expr", "r_expr", 
                "ruleset_expr", "iter_expr", "ctx_expr", "var_expr"])

    def _parser_dispatch(self, i, block, keys):
        (i, cmd), data = self._unpack_block(i, block)
        for k in keys:
            template, handler = self._dispatcher[k]
            m = self._match_exp(cmd, template)
            if m is not None:
                handler(i, cmd, m, data)
                break
        else:
            raise ParseError(f"Invalid command '{cmd}' on line {i + 1}")

    def _unpack_block(self, i, block):
        if isinstance(block, dict):
            assert len(block) == 1, f"Malformed block on line {i + 1}"
            exp, data = list(block.items())[0]
            return exp, data
        elif isinstance(block, tuple):
            return block, []
        else: 
            assert False, f"Malformed block on line {i + 1}"

    def _prepare_id(self, key, index, n, prefix=True):
        id_ = key
        if n > 1:
            id_ = uris.SEP.join([id_, str(index)]).strip(uris.SEP)
        if prefix:
            id_ = uris.FSEP.join([self.prefix, id_]).strip(uris.FSEP)
        return id_

    def _extract_id(self, match, i=None) -> str:
        id_str = match.group("id")
        if id_str is None:
            raise ParseError(f"Invalid identifier '{id_str}'")
        if id_str and not uris.ispath(id_str):
            raise ParseError(f"Invalid identifier '{id_str}'")
        if i is not None:
            id_str = uris.SEP.join([id_str, str(i)])
        return id_str

    def _match_exp(self, exp, match_re):
        exp = self._clean_exp(exp)
        return match_re.fullmatch(exp)

    def _clean_exp(self, exp):
        exp = re.sub(" +", " ", exp)
        old = None
        while old != exp:
            old = exp
            for var, val in self._iter_vars.items():
                exp = re.sub(f"{{{var}}}", str(val), exp)
        return exp

    ################################
    # Methods for Parsing cf_exprs #
    ################################

    def _parse_cf_expr(self, i, cmd, m, _):
        _d = m.group("d")
        if _d is None:
            raise ParseError(f"Empty dimension field on line {i + 1}.")
        d: str = _d
        v: str | int | None = m.group("v") or None
        if v is not None:
            try: 
                v = int(v) 
            except ValueError: 
                pass
            if v == "None":
                v = None
        l: int = int(m.group("l") or 0)
        w: float = float(m.group("w") or 1)
        if self.fspace is not None and feature(d, v, l) not in self.fspace:
            raise ParseError(f"Feature spec '{(d, v, l)}' on line {i + 1} not "
                "a member of given fspace")
        self._fspecs.append(((d, v, l), ((d, l), w)))

    ###############################
    # Methods for Parsing c_exprs #
    ###############################

    @contextmanager
    def _chunk_ctx(self):
        self._fspecs = []
        self._iter_ctx.append(IterContext.CHUNK)
        yield
        self._iter_ctx.pop()
        self._fspecs = []

    def _parse_c_expr(self, i, cmd, m, data):
        cid = self._extract_id(m, i + 1)
        assert isinstance(data, list), f"Malformed chunk block on line {i + 1}"
        data = data[1:] + self._ctx
        fs, ws = self._parse_c_data(i, data)
        (self._chunks_s1
            .setdefault(cid, [])
            .append(ChunkSpec(fs=fs, ws=ws))) # type: ignore

    def _parse_c_data(self, i, data):
        with self._chunk_ctx():
            for block in data:
                self._parser_dispatch(i, block, keys=["cf_expr", "iter_expr"])
            fs, _ws = zip(*self._fspecs)
        ws = self._parse_cw_sequence(i, _ws)
        return fs, ws

    def _parse_cw_sequence(self, i, seq):
        d = {}
        for dim, weight in seq:
            if dim in d and weight != 1.0:
                raise ParseError(f"Multiple weight specs for dim '{dim}' "
                    f"on line {i + 1}")
            d[dim] = weight
        return d

    ###############################
    # Methods for Parsing r_exprs #
    ###############################

    @contextmanager
    def _rule_ctx(self, rid):
        self._rid = rid
        self._rspec = {}
        self._iter_ctx.append(IterContext.RULE)
        yield
        self._iter_ctx.pop()
        self._rspec = {}
        self._rid = ""

    def _parse_r_expr(self, i, cmd, m, data):
        rid = (self._ruleset_id, self._extract_id(m), str(i + 1))
        with self._rule_ctx(rid):
            assert isinstance(data, list), "Malformed rule body."
            for block in data[1:]:
                self._parser_dispatch(i, block, ["r_subexpr", "iter_expr"])
            if "conc" not in self._rspec:
                raise ParseError(f"No conc defined for rule '{rid}' on "
                    f"line {i + 1}")
            if "cond" not in self._rspec:
                raise ParseError(f"No conds defined for rule '{rid}' on "
                    f"line {i + 1}")
            conc, conds = self._rspec["conc"], self._rspec["cond"]
            (self._rules_s1
                .setdefault(rid, [])
                .append(RuleSpecS1(conds=conds, conc=conc)))
        
    def _parse_r_subexpr(self, i, cmd, m, data):
        t, n, w = m.group("type"), m.group("id"), float(m.group("w") or 1)
        cid = uris.SEP.join(filter(bool, [t, n, str(i + 1)])).strip(uris.SEP)
        if t == "cond": 
            data = data + self._ctx
        fs, ws = self._parse_c_data(i, data[1:])
        parse = (cid, ChunkSpec(fs=fs, ws=ws), w) #type: ignore
        if t == "conc":
            if "conc" in self._rspec:
                raise ParseError(f"Extra conc '{n}' in rule on line {i + 1}")
            self._rspec[t] = parse
        else:
            assert t == "cond"
            self._rspec.setdefault(t, []).append(parse)

    #####################################
    # Methods for Parsing ruleset_exprs #
    #####################################

    @contextmanager
    def _ruleset(self, i, ruleset_id):

        # Can this be relaxed?
        if ruleset_id in self._rulesets:
            raise ParseError(f"Framented definition for ruleset '{ruleset_id}' "
                f"on line {i} ")
        else:
            self._rulesets.append(ruleset_id)

        if self._ruleset_id != "":
            raise ParseError(f"Nested ruleset on line {i}")

        self._ruleset_id = ruleset_id
        self._iter_ctx.append(IterContext.RULESET)
        yield
        self._iter_ctx.pop()
        self._ruleset_id = ""

    def _parse_ruleset_expr(self, i, cmd, m, data):
        ruleset_id = self._extract_id(m)
        with self._ruleset(i, ruleset_id):
            for block in data[1:]:
                self._parser_dispatch(i, block, ["r_expr", "iter_expr", 
                    "ctx_expr"])

    #################################
    # Methods for Parsing ctx_exprs #
    #################################

    @contextmanager
    def _local_ctx(self, i):
        self._ctx_level.append(len(self._ctx_delims))
        yield
        lvl = self._ctx_level.pop()
        self._contract_ctx(i, len(self._ctx_delims) - lvl)

    def _parse_ctx_expr(self, i, cmd, m, data):
        with self._local_ctx(i):
            data = data[1:] # drop data after colon
            try:
                index = [type(stmt) for stmt in data].index(dict)
            except ValueError as e:
                index = len(data)
            self._ctx_delims.append(len(self._ctx))
            self._ctx.extend(data[:index])
            for block in data[index:]:
                if self._ruleset_id:
                    self._parser_dispatch(i, block, keys=["r_expr", 
                        "iter_expr", "ctx_expr"])
                else:
                    self._parser_dispatch(i, block, keys=["c_expr", "r_expr", 
                        "ruleset_expr", "iter_expr", "ctx_expr"])

    def _contract_ctx(self, i, levels):
        if levels > len(self._ctx_delims):
            raise ParseError(f"Context contraction too deep on line {i + 1}")
        if levels:
            n = self._ctx_delims[-levels]
            self._ctx_delims = self._ctx_delims[:-levels]
            self._ctx = self._ctx[:n - len(self._ctx)]

    #################################
    # Methods for Parsing for_exprs #
    #################################

    def _parse_iter_expr(self, i, cmd, m, data):
        _vars = m.group("vars")
        vars = _vars.split(" ")
        for var in vars:
            if var in self._iter_vars:
                raise ParseError(f"Reused var '{var}' in for expr on line "
                    f"{i + 1}")
        iterables, body = self._parse_iter_data(i, len(vars), data)
        for vals in zip(*iterables):
            for var, val in zip(vars, vals):
                self._iter_vars[var] = val
            for block in body:
                if self._iter_ctx[-1] == IterContext.VAR:
                    self._parser_dispatch(i, block, keys=["iter_expr", 
                        "var_data_expr"])
                elif self._iter_ctx[-1] == IterContext.CHUNK:
                    self._parser_dispatch(i, block, keys=["iter_expr", 
                        "cf_expr"])
                elif self._iter_ctx[-1] == IterContext.RULE:
                    self._parser_dispatch(i, block, keys=["iter_expr", 
                        "r_subexpr"])
                elif self._iter_ctx[-1] == IterContext.RULESET:
                    self._parser_dispatch(i, block, keys=["iter_expr", 
                        "r_expr", "ctx_expr"])
                else:
                    assert self._iter_ctx[-1] == IterContext.NONE
                    self._parser_dispatch(i, block, keys=["iter_expr", "c_expr", 
                        "r_expr", "ctx_expr"])
        for var in vars:
            del self._iter_vars[var]

    def _parse_iter_data(self, i, n_seq, data):
        if len(data) <= n_seq:
            raise ParseError(f"Malformed iter block on line {i + 1}.")
        iterables = []
        for block in data[1:n_seq + 1]:
            (j, cmd), _data = self._unpack_block(i, block)
            if cmd != "in":
                raise ParseError(f"Expected 'in' block on line {j + 1}, "
                    f"got '{cmd}' instead")
            if not len(_data):
                raise ParseError(f"Empty 'in' block on line {j + 1}")
            iterables.append(self._parse_var_data(_data).split())
        if 1 < len(set(len(d) for d in iterables)):
            raise ParseError(f"Uneven 'in' blocks on line {i + 1}")
        body = data[n_seq + 1:]
        return iterables, body

    #################################
    # Methods for Parsing var_exprs #
    #################################

    @contextmanager
    def _var(self):
        self._var_data = []
        self._iter_ctx.append(IterContext.VAR)
        yield
        self._iter_ctx.pop()
        self._var_data = []

    def _parse_var_expr(self, i, cmd, m, data):
        var = m.group("id")
        with self._var():
            for block in data:
                self._parser_dispatch(i, block, keys=["iter_expr", 
                    "var_data_expr"]) # order sig as var_data_expr matches all
            self._iter_vars[var] = self._parse_var_data(self._var_data)
 
    def _parse_var_data_expr(self, i, cmd, m, data):
        self._var_data.append((i, m.string))
 
    def _parse_var_data(self, data):
        exp = " ".join([s for _, s in data])
        return self._clean_exp(exp)

    ###############
    # Second Pass #
    ###############

    def _pass_2(self):
        self._prepare_chunks_s2()
        self._prepare_rules_s2()

    def _prepare_chunks_s2(self):
        
        for cname, cspecs in self._chunks_s1.items():
            n = len(cspecs)
            for i, cspec in enumerate(cspecs):
                cid = uris.SEP.join([self._c_prefix, cname]).strip(uris.SEP)
                cid = self._prepare_id(cid, i, n)
                self._chunks_s2[cid] = cspec

    # This might need to be broken up.
    def _prepare_rules_s2(self):
        for (ruleset, rname, lineno), rspecs in self._rules_s1.items():
            n = len(rspecs)
            rname = (uris.SEP.join(
                filter(bool, [self._r_prefix, ruleset, rname, lineno]))
                .strip(uris.SEP))
            if rname in self._rnames:
                raise ParseError("Rule stub '{rname}' already in use." 
                    "Check for clashes between rule names and ruleset names.")
            for i, rspec in enumerate(rspecs):
                rid = self._prepare_id(rname, i, n)
                conc_id, conc_spec, conc_w = rspec["conc"]
                conc_id = uris.SEP.join([rid, conc_id]).strip(uris.SEP)
                self._chunks_s2[conc_id] = conc_spec
                rspec2 = RuleSpecS2(conc=(conc_id, conc_w), conds=[])
                n_conds = len(rspec["conds"])
                for j, (cond_id, cond_spec, cond_w) in enumerate(rspec["conds"]):
                    cond_id = self._prepare_id(cond_id, j, n_conds, False)
                    cond_id = (uris.SEP
                        .join([rid, cond_id])
                        .strip(uris.SEP))
                    self._chunks_s2[cond_id] = cond_spec
                    rspec2["conds"].append((cond_id, cond_w))
                self._rules_s2[rid] = rspec2

    def _prepare_output(self):

        cs, cf, cw = set(), nd.NumDict(), nd.NumDict()
        for cid, cspec in self._chunks_s2.items():
            c = chunk(cid)
            cs.add(c)
            for fspec in cspec["fs"]:
                cf[c, feature(*fspec)] = 1
            for dspec, w in cspec["ws"].items():
                cw[c, dimension(*dspec)] = w
        wn = cw.abs().sum_by(kf=first)

        rs, cr, rc = set(), nd.NumDict(), nd.NumDict()
        for rid, rspec in self._rules_s2.items():
            conc_id, conc_w = rspec["conc"]
            r, conc = rule(rid), chunk(conc_id)
            rs.add(r)
            rc[r, conc] = conc_w
            for cond_id, cond_w in rspec["conds"]:
                cond = chunk(cond_id)
                cr[cond, r] = cond_w

        return cs, rs, cf, cw, wn, cr, rc


def load(f, structure: Structure, uri: str, p: nd.NumDict):
    """Load top level knowledge encoded in a pyClarion markup file."""

    module = structure[uri] 
    if not isinstance(module, Module):
        raise TypeError(f"Expected Module object at '{uri}', got "
            f"'{type(module)}' instead")

    store = module.process
    if not isinstance(store, Store):
        raise TypeError(f"Expected process of type Store in module '{uri}', "
            f"got '{type(store)}' instead")

    fspace = inspect.fspace(structure)
    prefix = module.path
    loader = Parser(prefix, fspace)
    cs, rs, cf, cw, wn, cr, rc = loader.parse(f)

    store.cf = cf
    store.cw = cw
    store.wn = wn
    store.cr = cr
    store.rc = rc

    if store.cb is not None:
        store.cb.update(p, nd.NumDict({c: 1 for c in cs}))
    if store.rb is not None:
        store.rb.update(p, nd.NumDict({r: 1 for r in rs}))
