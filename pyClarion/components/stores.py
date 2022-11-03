"""Tools for instantiating top level knowledge."""


from __future__ import annotations
from typing import (OrderedDict, Tuple, Any, Dict, List, Generic, TypeVar, 
    Container, Optional, Callable)
from itertools import count

from ..base import Process, uris, dimension, feature, chunk, rule
from .. import numdicts as nd
from .. import dev as cld


__all__ = ["BLATracker", "Store", "GoalStore"]


T = TypeVar("T")


class BLATracker(Generic[T]):
    """BLA tracker for top level stores."""

    params = ("th", "bln", "amp", "dec")

    lags: nd.NumDict[Tuple[T, int]]
    uses: nd.NumDict[T]
    lifetimes: nd.NumDict[T]

    def __init__(self, depth: int = 1) -> None:
        """
        Initialize a new BLA tracker.

        :param depth: Depth of estimate.
        """
        
        if depth < 0:
            raise ValueError("Depth must be non-negative.")
        self.depth = depth

        self.lags = nd.NumDict()
        self.uses = nd.NumDict()
        self.lifetimes = nd.NumDict()

    def call(self, p: nd.NumDict[str]) -> nd.NumDict:

        bln = p.isolate(key=self.params[1])
        amp = p.isolate(key=self.params[2])
        dec = p.isolate(key=self.params[3])

        selector = self.uses > self.depth
        selector_lags = self.lags.put(selector, kf=cld.first)
        n = self.uses.keep_if(cond=selector)
        t_n = self.lifetimes.keep_if(cond=selector)
        t_k = self.lags.keep_if(cond=selector_lags).max_by(kf=self.xi2x)
        dec_rs_1 = (1 - dec)
        factor = (n - self.depth).set_c(0) / dec_rs_1
        t = (t_n ** dec_rs_1 - t_k ** dec_rs_1) / (t_n - t_k).set_c(1)
        distant_approx = factor * t

        dec_lags = self.lags.put(dec, kf=cld.first)
        sum_t = (self.lags ** -dec_lags).sum_by(kf=self.xi2x) + distant_approx

        return bln + amp * sum_t

    def update(self, p: nd.NumDict[str], d: nd.NumDict[T]) -> None:

        th = p.isolate(key=self.params[0])
        invoked = (d > th).squeeze()

        self.uses += invoked
        self.lifetimes += (
            self.lifetimes
            .add((d > th).mul(0)) # ensures new items added to lifetimes
            .mask()) 

        invoked_lags = self.lags.put(invoked, kf=cld.first)
        unshifted = self.lags.keep_if(cond=invoked_lags.rsub(1))
        self.lags = (self.lags
            .keep_if(invoked_lags) # select entries w/ new invocations
            .transform_keys(kf=self.shift) 
            .drop(sf=self.expired) # drop entries beyond depth limit
            .merge(unshifted) # merge in unshifted entries
            .add(1) # update existing lags
            .merge(invoked.transform_keys(kf=self.x2xi)))

    def drop(self, d: Container[T]) -> None:
        self.uses = self.uses.drop(sf=d.__contains__)
        self.lifetimes = self.lifetimes.drop(sf=d.__contains__)
        self.lags = self.lags.drop(sf=lambda k: k[0] in d)

    def expired(self, key: Tuple[Any, int]) -> bool:
        return key[1] >= self.depth

    @staticmethod    
    def shift(key: Tuple[T, int]) -> Tuple[T, int]:
        return (key[0], key[1] + 1)
    
    @staticmethod
    def xi2x(key: Tuple[T, int]) -> T:
        return key[0]

    @staticmethod
    def x2xi(key: T) -> Tuple[T, int]:
        return (key, 0)


class Store(Process):
    """Basic store for top-level knowledge."""

    initial = (
        nd.NumDict(), nd.NumDict(), nd.NumDict(),# chunk weights 
        nd.NumDict(), nd.NumDict(), # rule weights
        nd.NumDict(), nd.NumDict()) # blas

    cf: nd.NumDict[Tuple[chunk, feature]] # (chunk, feature): 1.0, c=0
    cw: nd.NumDict[Tuple[chunk, dimension]] # (chunk, dim): w, c=0
    wn: nd.NumDict[chunk] # chunk: sum(|w|), c=0
    cr: nd.NumDict[Tuple[chunk, rule]] # (chunk, rule): 1.0, c=0, rank=2
    rc: nd.NumDict[Tuple[rule, chunk]] # (rule, chunk): w, c=0, rank=2
    cb: Optional[BLATracker[chunk]]
    rb: Optional[BLATracker[rule]]

    # parameter prefixes for cb and rb
    c_pre = "c"
    r_pre = "r"

    def __init__(
        self, 
        g: Callable[[nd.NumDict[chunk]], nd.NumDict[chunk]] = cld.eye,
        cbt: Optional[BLATracker[chunk]] = None, 
        rbt: Optional[BLATracker[rule]] = None
    ) -> None:

        self.g = g
        self.cf = nd.NumDict() 
        self.cw = nd.NumDict()
        self.wn = nd.NumDict() 
        self.cr = nd.NumDict() 
        self.rc = nd.NumDict() 
        self.cb = cbt
        self.rb = rbt

    def call(
        self, 
        p: nd.NumDict[feature], 
        f: nd.NumDict[feature],
        c: nd.NumDict[chunk], 
        r: nd.NumDict[rule]
    ) -> Tuple[
        nd.NumDict[Tuple[chunk, feature]], 
        nd.NumDict[Tuple[chunk, dimension]],
        nd.NumDict[chunk],
        nd.NumDict[Tuple[chunk, rule]],
        nd.NumDict[Tuple[rule, chunk]],
        nd.NumDict[chunk],
        nd.NumDict[rule]]:

        cb, rb = self.update_blas(p, c, r)
        wn = self.g(self.wn).set_c(0)
        return self.cf, self.cw, wn, self.cr, self.rc, cb, rb

    def update_blas(
        self, p: nd.NumDict[feature], c: nd.NumDict[chunk], r: nd.NumDict[rule]
    ) -> Tuple[nd.NumDict[chunk], nd.NumDict[rule]]:

        if self.cb is None: 
            cb = nd.NumDict()
        else: 
            cp = self._extract_cp(p)
            self.cb.update(cp, c)
            cb = self.cb.call(cp)

        if self.rb is None: 
            rb = nd.NumDict()
        else: 
            rp = self._extract_rp(p)
            self.rb.update(rp, r)
            rb = self.rb.call(rp)

        return cb, rb

    def _extract_cp(self, p: nd.NumDict):
        return (p
            .keep(sf=self._select_cps)
            .transform_keys(kf=self._transform_cps))

    def _extract_rp(self, p: nd.NumDict):
        return (p
            .keep(sf=self._select_rps)
            .transform_keys(kf=self._transform_rps))

    def _select_cps(self, k):
        if self.cb is not None:
            return k in self.params[0:len(self.cb.params)]
        else:
            raise ValueError("Chunk BLAs not defined")

    def _transform_cps(self, k):
        if self.cb is None:
            raise ValueError("Chunk BLAs not defined")
        else:
            params = self.params[0:len(self.cb.params)]
            return self.cb.params[params.index(k)]

    def _select_rps(self, k):
        if self.rb is None:
            raise ValueError("Rule BLAs not defined")
        else:
            offset = 0 if self.cb is None else len(self.cb.params)
            return k in self.params[offset:offset + len(self.rb.params)]

    def _transform_rps(self, k):
        if self.rb is None:
            raise ValueError("Rule BLAs not defined")
        else:
            offset = 0 if self.cb is None else len(self.cb.params)
            params = self.params[offset:offset + len(self.rb.params)]
            return self.rb.params[params.index(k)]

    @property
    def params(self):
        cps = ["-".join(filter(None, [self.c_pre, p])) 
            for p in self.cb.params] if self.cb is not None else []
        rps = ["-".join(filter(None, [self.r_pre, p])) 
            for p in self.rb.params] if self.rb is not None else []
        ps = cld.prefix(cps + rps, self.prefix)
        return tuple(feature(p) for p in ps)   


class GoalStore(Store):

    _set_pre = "set"
    _eval = "eval"
    _eval_vals = ("pass", "fail", "quit", None)

    def __init__(self, 
        gspec: Dict[str, List[str]], 
        g: Callable[[nd.NumDict[chunk]], nd.NumDict[chunk]] = cld.eye,
        cbt: Optional[BLATracker[chunk]] = None,
    ) -> None:

        super().__init__(g=g, cbt=cbt)
        self.gspec = OrderedDict(gspec)
        self.count = count()

    def call(
        self, 
        p: nd.NumDict[feature],
        f: nd.NumDict[feature], 
        c: nd.NumDict[chunk], 
        r: nd.NumDict[rule]
    ) -> Tuple[
        nd.NumDict[Tuple[chunk, feature]], 
        nd.NumDict[Tuple[chunk, dimension]],
        nd.NumDict[chunk],
        nd.NumDict[Tuple[chunk, rule]],
        nd.NumDict[Tuple[rule, chunk]],
        nd.NumDict[chunk],
        nd.NumDict[rule]]:
        
        cmds = self.cmds
        f = f.drop(sf=lambda ftr: ftr.v is None)

        eval_ = f.keep(sf=cmds[-4:].__contains__)
        if len(eval_):
            self.cf = self.cf.drop(sf=lambda k: k[0] in c)
            self.cw = self.cw.drop(sf=lambda k: k[0] in c)
            if self.cb is not None: self.cb.drop(c)

        set_ = (f.keep(sf=cmds[:-4].__contains__)
            .transform_keys(kf=self._cmd2repr))
        if len(set_):
            new = chunk(uris.FSEP.join([self.prefix, str(next(self.count))])
                .strip(uris.FSEP))
            d = nd.NumDict({new: 1.0})
            self.cf = self.cf.merge(d.outer(set_))
            cw = d.outer(set_.transform_keys(kf=feature.dim.fget)) #type: ignore
            self.cw = self.cw.merge(cw)
            self.wn = self.wn.merge(cw.abs().sum_by(kf=cld.first))
            return super().call(p, nd.NumDict(), d, nd.NumDict())
        else:
            return super().call(p, nd.NumDict(), nd.NumDict(), nd.NumDict())

    def _cmd2repr(self, cmd):
        return self.reprs[self.cmds.index(cmd) - 1]

    def _goal_items(self):
        if len(self.gspec) > 0:
            return list(zip(*self.gspec.items()))
        else:
            return [], []

    @property
    def reprs(self) -> Tuple[feature, ...]:
        ds, v_lists = self._goal_items()
        ds = cld.prefix(ds, self.prefix) # type: ignore
        return tuple(feature(d, v) for d, vs in zip(ds, v_lists) for v in vs)

    @property
    def cmds(self) -> Tuple[feature, ...]:
        ds, v_lists = self._goal_items()
        ds = ["-".join([self._set_pre, d]) for d in ds] # type: ignore 
        ds = cld.prefix(ds, self.prefix) 
        v_lists = [[None] + l for l in v_lists] # type: ignore
        set_ = tuple(feature(d, v) for d, vs in zip(ds, v_lists) for v in vs)
        eval_dim = cld.prefix(self._eval, self.prefix)
        eval_ = tuple(feature(eval_dim, v) for v in self._eval_vals)
        return set_ + eval_

    @property
    def nops(self) -> Tuple[feature, ...]:
        ds = ["-".join(filter(None, [self._set_pre, d])) # type: ignore
            for d in self.gspec.keys()] 
        ds = cld.prefix(ds, self.prefix) 
        set_ = tuple(feature(d, None) for d in ds)
        eval_dim = cld.prefix(self._eval, self.prefix)
        return set_ + (feature(eval_dim, None), )
