from typing import Sequence

from ..system import Process, Site
from ..knowledge import Family, Sort, Term, keyform
from ..numdicts import path, Index


type D = Family | Sort | Term
type V = Family | Sort
type DV = tuple[D, V]


class DualRepMixin:    
    system: Process.System

    def _init_indexes(
        self,
        *keyspaces: D | V | DV
    ) -> Sequence[Index]:
        indices = []
        for item in keyspaces:
            match item:
                case (d, v):
                    self.system.check_root(d, v)
                    idx_d = self.system.get_index(keyform(d))
                    idx_v = self.system.get_index(keyform(v))
                    indices.append(idx_d * idx_v)
                case d:
                    self.system.check_root(d)
                    idx_d = self.system.get_index(keyform(d))
                    indices.append(idx_d)
        return indices


class ParamMixin(Process):
    name: str
    system: Process.System

    def _init_params[P: Sort](self, 
        p: Family, 
        Params: type[P], 
        **params: float
    ) -> tuple[P, Site]:
        self.system.check_root(p)
        sort = Params(); p[self.name] = sort
        site = Site(
            i=self.system.get_index(keyform(sort)), 
            d={path(sort[k]): v for k, v in params.items()}, 
            c=float("nan"))
        return sort, site
