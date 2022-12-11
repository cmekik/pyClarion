from typing import List, Optional, Callable, Deque, Tuple
from collections import deque

from ..numdicts import NumDict
from ..base.constructs import Process, subprocesses
from ..base.symbols import F, D, V, C, R
from .. import sym
from .chunks import ChunkStore, BottomUp
from .rules import RuleStore


class RuleExtractor(Process):
    rules: RuleStore
    concs: ChunkStore
    conds: ChunkStore
    bu: BottomUp
    th_match: float
    th_re: float
    th_chunk: float
    zfill: int
    _n: int
    _cc: Deque[Tuple[NumDict[C], NumDict[C]]]
    _xa: Deque[Tuple[NumDict[F], NumDict[V]]]
    _yr: Deque[Tuple[NumDict[F], NumDict[D]]]

    def __init__(
        self, 
        path: str = "", 
        inputs: Optional[List[str]] = None, 
        g: Callable[[NumDict[C]], NumDict[C]] = sym.eye,
        th_match: float = .99,
        th_re: float = 0.,
        th_chunk: float = .5,
        zfill: int = 3
    ) -> None:
        super().__init__(path, inputs)
        with subprocesses():
            self.rules = RuleStore()
            self.concs = ChunkStore(g=g)
            self.conds = ChunkStore(g=g)
            self.bu = BottomUp()
        self.th_match = th_match
        self.th_re = th_re
        self.th_chunk = th_chunk
        self.zfill = zfill
        self._n = 0
        self._cc = deque([(NumDict(),) * 2] * 2, maxlen=2)
        self._xa = deque([(NumDict(),) * 2] * 2, maxlen=2)
        self._yr = deque([(NumDict(),) * 2], maxlen=1)

    def initial(self) -> Tuple[NumDict, ...]:
        return (self.rules.cr, self.rules.rc, 
            self.conds.cf, self.conds.cd, self.conds.cn, 
            self.concs.cf, self.concs.cd, self.concs.cn)

    def call(
        self, 
        x_curr: NumDict[F],
        a_curr: NumDict[V],
        r_prev: NumDict[D]
    ) -> Tuple[NumDict, ...]:
        # Due to relative lengths of deques _xa & _yr leftmost elts satisfy:
        # x: prev state, a: prev action, y: curr state, r: immed. reinforcement
        self._xa.append((x_curr, a_curr))
        self._yr.append((x_curr, r_prev)) 
        (x, a), (_, r) = self._xa[0], self._yr[0]
        m_cr, m_rc, m_r = self._get_matches(x, a)
        m = m_r.reduce_sum() # number of matching rules
        pc = r.reduce_sum().greater(self.th_re) # pos. criterion
        if m.c == 0 and pc.c == 1:
            self._extract(x, a)
        return (self.rules.cr, self.rules.rc, 
            self.conds.cf, self.conds.cd, self.conds.cn, 
            self.concs.cf, self.concs.cd, self.concs.cn)

    def _extract(self, x, a):
        self._n = self._n + 1
        n = str(self._n).zfill(self.zfill)
        r = R(f"_{n}", p=self.path) 
        c_x, c_a = C(f"cond-{n}", p=self.path), C(f"actn-{n}", p=self.path)
        self.rules.add(NumDict({(c_x, r): 1.0}), NumDict({(r, c_a): 1.0}))
        self.conds.add(*self._make_chunk(x, c_x))
        self.concs.add(*self._make_chunk(a, c_a))
        
    def _get_matches(self, x, a):
        cond_cf, cond_cd, cond_cn = self.conds.call()
        conc_cf, conc_cd, conc_cn = self.concs.call()
        conds = self.bu.call(cond_cf, cond_cd, cond_cn, x) 
        concs = self.bu.call(conc_cf, conc_cd, conc_cn, a)
        m_cr = (self.rules.cr
            .mul_from(conds.greater(self.th_match), kf=sym.first))
        m_rc = (self.rules.rc
            .mul_from(concs.greater(self.th_match), kf=sym.second))
        m_r = (m_cr.transform_keys(kf=sym.second)
            .mul(m_rc.transform_keys(kf=sym.first)))
        return m_cr, m_rc, m_r

    def _make_chunk(self, fs, c):
        f2cf = lambda k: (c, k)
        s_dims = fs.cam_by(kf=F.dim.fget)
        dims = s_dims.abs().greater(self.th_chunk).squeeze()
        weights = dims.mul_from(s_dims.sign(), kf=sym.eye)
        vals = (fs
            .mul_from(weights, kf=F.dim.fget)
            .greater(self.th_chunk)
            .squeeze())
        return vals.transform_keys(kf=f2cf), weights.transform_keys(kf=f2cf)
