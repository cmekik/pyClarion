from typing import List, Tuple, Dict, TypeVar, Type, OrderedDict

from ..base import symbols as sym

from .exc import CCMLValueError, CCMLTypeError, CCMLNameError
from .primitives import (TSpec, RSpec, CSpec, FSpec, Spec, Frame, AST, line, 
    scope)


ST = TypeVar("ST", bound=Spec)
def merge_specs(cls: Type[ST], subtrees: List[AST[ST]], frame: Frame) -> ST:
    spec = cls()
    for t in subtrees:
        ret_t = t.exec(frame)
        if isinstance(ret_t, cls): 
            spec.add(ret_t, frame)
        else: 
            with frame.line(t.lineno): 
                raise frame.new_exc(CCMLTypeError, f"Expected return "
                f"type '{cls.__name__}' but got '{ret_t.__class__.__name__}'" 
                "instead")
    return spec

def collect_kwds(prefix: str, kwds: Dict[str, str], frame: Frame) \
    -> Dict[str, str]:
    for k in kwds:
        if f"{prefix}{k}" not in frame.kwds:
            raise frame.new_exc(CCMLNameError, f"Unexpected kwarg '{k}'")
    data = {}
    for k, typ in frame.kwds.items():
        assert k in frame.vars
        if k.startswith(prefix):
            kw = k.lstrip(prefix)
            data[kw] = frame.sub(kwds.get(kw, frame.vars[k]), typ)
    return data

def name_explicit_node(s: List[str], frame: Frame) -> str:
    if s:
        i = frame.sub(s[0], "seq")
    else:
        i = str(frame.lineno).zfill(int(frame.vars["@_zfl"]))
    i = "-".join([i, *(str(idx) for idx in frame.iternos)])
    return i


class ROOT(AST):
    def exec(self, frame: Frame) -> Tuple[Dict[str, CSpec], Dict[str, RSpec]]:
        frame.kwds.update({
            "@_zfl": "itg",
            "@_fl": "itg", "@_fm": "mrk", "@_fp": "pth",
            "@_cl": "itg", "@_cm": "mrk", "@_cp": "pth", 
            "@_rl": "itg", "@_rm": "mrk", "@_rp": "pth", 
        })
        frame.vars.update({
            "@_zfl": "4",
            "@_fl": "0", "@_fm": "", "@_fp": "",
            "@_cl": "0", "@_cm": "", "@_cp": "", 
            "@_rl": "0", "@_rm": "", "@_rp": "",})
        cspecs, rspecs = {}, {}
        for c in self.children:
            ret_c = c.exec(frame)
            if isinstance(ret_c, tuple):
                try:
                    adr, spec = ret_c
                except ValueError:
                    continue
                except TypeError:
                    continue
                if isinstance(spec, CSpec):
                    cspecs[adr] = spec
                elif isinstance(spec, RSpec):
                    rspecs[adr] = spec
        return cspecs, rspecs


class F(AST):
    def check_signature(self):
        if len(self.args) != 1:
            raise CCMLTypeError("Expected exactly one arg but got "
                f"{len(self.args)} on line {self.lineno}")
        if self.children:
            raise CCMLTypeError(f"Unexpected children for expr on line "
                f"{self.lineno}")
    @line
    def exec(self, frame: Frame) -> FSpec:
        data = collect_kwds("@_f", self.kwds, frame) 
        dvs = frame.sub(self.args[0], "dvs").split(":")
        assert len(dvs) in [1, 2]
        dim = frame.sub(dvs[0], "seq")
        vals = dvs[1].strip("{}").split(",") if len(dvs) == 2 else [] 
        l, m, p = int(data.pop("l")), data.pop("m"), data.pop("p")
        d = sym.F(dim, l, m, p)
        fs = [sym.F((dim, val), l, m, p) for val in vals] if vals else [d]
        return FSpec(OrderedDict([(d, (fs, data))]))


class F_SET(AST[None]):
    def check_signature(self):
        if len(self.args) != 1:
            raise CCMLTypeError("Expected exactly one arg but got "
                f"{len(self.args)} on line {self.lineno}")
        if not sym.match(self.args[0], "var"):
            raise CCMLTypeError(f"Data '{self.args[0]}' does not match "
                f"expected type 'var' on line {self.lineno}")
        if self.args[0].startswith("@_"):
                raise CCMLValueError(f"Var arg may not start with '@_' on line"
                f"{self.lineno}")
    @line
    def exec(self, frame: Frame) -> None:
        if self.args[0] in frame.fs:
            raise frame.new_exc(CCMLNameError, f"Var '{self.args[0]}' already "
                "bound")
        frame.fs[self.args[0]] = merge_specs(FSpec, self.children, frame)


class FS(AST[FSpec]):
    def check_signature(self):
        if len(self.args) != 1:
            raise CCMLTypeError("Expected exactly one arg but got "
                f"{len(self.args)} on line {self.lineno}")
        if not sym.match(self.args[0], "var"):
            raise CCMLTypeError(f"Data '{self.args[0]}' does not match "
                "expected type 'var'")
        if self.children:
            raise CCMLTypeError(f"Unexpected children for expr on line "
                f"{self.lineno}")
    @line
    def exec(self, frame: Frame) -> FSpec:
        try:
            return frame.fs[self.args[0]]
        except KeyError:
            raise frame.new_exc(CCMLNameError, f"Var '{self.args[0]}' not "
                "bound") from None


class C(AST[CSpec]):
    def check_signature(self):
        if 1 < len(self.args):
            raise CCMLTypeError("Expected at most one arg but got "
                f"{len(self.args)} on line {self.lineno}")
    @line
    @scope
    def exec(self, frame: Frame) -> CSpec:
        if "@_fw" not in frame.kwds:
            frame.kwds["@_fw"] = "frk"
            frame.vars["@_fw"] = "1" 
        data = collect_kwds("@_c", self.kwds, frame)
        i = name_explicit_node(self.args, frame)
        l, m, p = int(data.pop("l")), data.pop("m"), data.pop("p")
        c = sym.C(i, l, m, p)
        fspec = merge_specs(FSpec, self.children, frame)
        if c in frame.cs and fspec.data:
            raise frame.new_exc(CCMLValueError, f"Chunk {c} already defined")
        frame.cs.add(c)
        return CSpec(OrderedDict([(c, (fspec, data))]))


class C_SET(AST[Tuple[str, CSpec]]):
    def check_signature(self) -> None:
        if len(self.args) != 1:
            raise CCMLTypeError("Expected exactly one arg but got "
                f"{len(self.args)} on line {self.lineno}")            
    @line
    @scope
    def exec(self, frame: Frame) -> Tuple[str, CSpec]:
        cspec = CSpec()
        adr = frame.sub(self.args[0], "adr")
        frame.vars["@_cp"] = adr
        for c in self.children:
            ret_c = c.exec(frame)
            if isinstance(ret_c, CSpec): 
                cspec.add(ret_c, frame)
            elif ret_c is None:
                continue
            else: 
                with frame.line(c.lineno): 
                    raise frame.new_exc(CCMLTypeError, f"Expected return "
                        f"type 'CSpec' but got '{ret_c.__class__.__name__}' "
                        "instead")
        return adr, cspec


class R(AST[RSpec]):
    def check_signature(self):
        if 1 < len(self.args):
            raise CCMLTypeError("Expected at most one arg but got "
                f"{len(self.args)} on line {self.lineno}")
    @line
    @scope
    def exec(self, frame: Frame) -> RSpec:
        if "@_cw" not in frame.kwds:
            frame.kwds["@_cw"] = "frk"
            frame.vars["@_cw"] = "1" 
        data = collect_kwds("@_r", self.kwds, frame)
        i = name_explicit_node(self.args, frame)
        l, m, p = int(data.pop("l")), data.pop("m"), data.pop("p")
        r = sym.R(i, l, m, p) 
        cspec = merge_specs(CSpec, self.children, frame)
        if len(cspec.data) < 2:
            raise frame.new_exc(CCMLValueError, "Expected at least 2 chunk "
                f"specs but got {len(cspec.data)}")
        if r in frame.rs:
            raise frame.new_exc(CCMLValueError, f"Rule {r} already defined")
        frame.rs.add(r)
        return RSpec(OrderedDict([(r, (cspec, data))]))


class R_SET(AST[Tuple[str, RSpec]]):
    def check_signature(self) -> None:
        if len(self.args) != 1:
            raise CCMLTypeError("Expected exactly one arg but got "
                f"{len(self.args)} on line {self.lineno}")            
    @line
    @scope
    def exec(self, frame: Frame) -> Tuple[str, RSpec]:
        rspec = RSpec()
        adr = frame.sub(self.args[0], "adr")
        frame.vars["@_rp"] = adr
        frame.vars["@_cp"] = adr
        for c in self.children:
            ret_c = c.exec(frame)
            if isinstance(ret_c, RSpec): 
                rspec.add(ret_c, frame)
            elif ret_c is None:
                continue
            else: 
                with frame.line(c.lineno): 
                    raise frame.new_exc(CCMLTypeError, f"Expected return "
                        f"type 'RSpec' but got '{ret_c.__class__.__name__}' "
                        "instead")
        return adr, rspec


class _CFG(AST[ST]):
    var_pefix: str
    spec_type: Type[ST]
    def check_signature(self):
        if len(self.args):
            raise CCMLTypeError("Expected exactly zero args but got "
                f"{len(self.args)} on line {self.lineno}")
    @line
    @scope
    def exec(self, frame: Frame) -> ST:
        for k, v in self.kwds.items():
            var = f"{self.var_pefix}{k}"
            try:
                t = frame.kwds[var]
            except KeyError:
                raise frame.new_exc(CCMLNameError, f"Unexpected kwarg '{k}'") \
                    from None
            frame.vars[var] = frame.sub(v, t)
        return merge_specs(self.spec_type, self.children, frame)


class F_CFG(_CFG[FSpec]):
    var_prefix = "@_f"
    spec_type = FSpec


class C_CFG(_CFG[CSpec]):
    var_prefix = "@_c"
    spec_type = CSpec


class R_CFG(_CFG[RSpec]):
    var_prefix = "@_r"
    spec_type = RSpec


class T(AST[TSpec]):
    def check_signature(self) -> None:
        if self.kwds:
            raise CCMLTypeError("No kwargs allowed in 't' on line "
                f"{self.lineno}")
        if self.children:
            raise CCMLTypeError(f"Unexpected children for expr on line "
                f"{self.lineno}")
    @line
    def exec(self, frame: Frame) -> TSpec:
        return TSpec(self.args)


class TABLE(AST[None]):
    def check_signature(self) -> None:
        if not len(self.args):
            raise CCMLTypeError("At least one arg must be provided to "
               f"'table' on line {self.lineno}")
        if self.kwds:
            raise CCMLTypeError("No kwargs allowed in 'table' on line "
                f"{self.lineno}")
        for arg in self.args:
            if not sym.match(arg, "var"):
                raise CCMLTypeError(f"Non-var arg '{arg}' on line "
                    f"{self.lineno}")
            if arg.startswith("@_"):
                raise CCMLValueError(f"Var arg may not start with '@_' on line "
                    f"{self.lineno}")
    @line
    def exec(self, frame: Frame) -> None:
        for c in self.children:
            ret_c = c.exec(frame)
            if isinstance(ret_c, TSpec):
                if len(ret_c) != len(self.args):
                    with frame.line(c.lineno):
                        raise frame.new_exc(CCMLValueError, "Expected "
                            f"{len(self.args)} values but got {len(ret_c)}")
                for var, val in zip(self.args, ret_c):
                    frame.lsts.setdefault(var, []).append(val)
            else:
                with frame.line(c.lineno):
                    raise frame.new_exc(CCMLTypeError, f"Expected return "
                        f"type 'TSpec' but got '{ret_c.__class__.__name__}' "
                        "instead")


class _PAT(AST[ST]):
    spec_type: Type[ST]
    def check_signature(self) -> None:
        if not len(self.args):
            raise CCMLTypeError("At least one arg must be provided to "
               f"'f_pat' on line {self.lineno}")
        if self.kwds:
            raise CCMLTypeError("No kwargs allowed in 'f_pat' on line "
                f"{self.lineno}")
        for arg in self.args:
            if not sym.match(arg, "var"):
                raise CCMLTypeError(f"Non-var arg '{arg}' on line "
                    f"{self.lineno}")
            if arg.startswith("@_"):
                raise CCMLValueError(f"Var arg may not start with '@_' on line "
                    f"{self.lineno}")
    @line
    def exec(self, frame: Frame) -> ST:
        try:
            _, = {len(frame.lsts[arg]) for arg in self.args}
        except ValueError:
            raise frame.new_exc(CCMLValueError, "Variable length value lists") \
                from None
        spec = self.spec_type()
        for i, vals in enumerate(zip(*(frame.lsts[var] for var in self.args))):
            with frame.iter(i):
                for var, val in zip(self.args, vals):
                    frame.vars[var] = val
                _spec = merge_specs(self.spec_type, self.children, frame)
                spec.add(_spec, frame)
        return spec


class F_PAT(_PAT[FSpec]):
    spec_type = FSpec


class C_PAT(_PAT[CSpec]):
    spec_type = CSpec


class R_PAT(_PAT[RSpec]):
    spec_type = RSpec


class _STATS(AST[None]):
    var_prefix: str
    def check_signature(self):
        if len(self.args):
            raise CCMLTypeError("Expected exactly zero args but got "
                f"{len(self.args)} on line {self.lineno}")
        for kwd in self.kwds:
            if kwd in ["l", "m", "p"]:
                raise CCMLValueError("Kwd may not be 'l', 'm', or 'p' on line "
                   f"{self.lineno}")
        if self.children:
            raise CCMLTypeError(f"Unexpected children for expr on line "
                f"{self.lineno}")
    @line
    def exec(self, frame: Frame) -> None:
        for kwd, val in self.kwds.items():
            var = f"{self.var_prefix}{kwd}"
            frame.kwds[var] = "frk"
            frame.vars[var] = val


class F_STATS(_STATS):
    var_prefix = "@_f"


class C_STATS(_STATS):
    var_prefix = "@_c"


class R_STATS(_STATS):
    var_prefix = "@_r"
