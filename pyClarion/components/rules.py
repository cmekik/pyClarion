from typing import Tuple, ClassVar, Optional, List, Callable

from ..numdicts import NumDict
from ..base.constructs import Process, subprocesses
from ..base.symbols import F, D, C, R
from .. import sym
from .chunks import ChunkStore
from .samplers import BoltzmannSampler


class RuleStore(Process):
    cr: NumDict[Tuple[C, R]]
    rc: NumDict[Tuple[R, C]]

    def __init__(self, path: str = "") -> None:
        super().__init__(path)
        self.cr = NumDict()
        self.rc = NumDict()

    def initial(self) \
        -> Tuple[NumDict[Tuple[C, R]], NumDict[Tuple[R, C]]]:
        return self.cr, self.rc

    call = initial

    def add(
        self, cr: NumDict[Tuple[C, R]], rc: NumDict[Tuple[R, C]], 
    ) -> None:
        self.cr = self.cr.merge(cr)
        self.rc = self.rc.merge(rc)


class FixedRuleSet(Process):
    rules: RuleStore
    concs: ChunkStore
    conds: ChunkStore

    def __init__(
        self, 
        path: str = "", 
        g: Callable[[NumDict[C]], NumDict[C]] = sym.eye,
    ) -> None:
        super().__init__(path)
        with subprocesses():
            self.rules = RuleStore()
            self.concs = ChunkStore(g=g)
            self.conds = ChunkStore(g=g)

    def initial(self) -> Tuple[NumDict, ...]:
        return (self.rules.cr, self.rules.rc, 
            self.conds.cf, self.conds.cd, self.conds.cn, 
            self.concs.cf, self.concs.cd, self.concs.cn)

    call = initial

    def add(
        self, 
        cr: NumDict[Tuple[C, R]], 
        rc: NumDict[Tuple[R, C]], 
        cond_cf: NumDict[Tuple[C, F]], 
        cond_cd: NumDict[Tuple[C, D]], 
        conc_cf: NumDict[Tuple[C, F]], 
        conc_cd: NumDict[Tuple[C, D]],
    ) -> None:
        self.rules.add(cr, rc)
        self.conds.add(cond_cf, cond_cd)
        self.concs.add(conc_cf, conc_cd)


class WTARules(Process):
    """Activates chunks through a winner-take-all competition among rules."""
    boltzmann: BoltzmannSampler
    _d_temp: ClassVar[str] = "temp"

    def __init__(
        self, 
        path: str = "", 
        inputs: Optional[List[str]] = None, 
        min_temp: float = 1e-8
    ) -> None:
        super().__init__(path, inputs)
        with subprocesses():
            self.boltzmann = BoltzmannSampler(path, min_temp=min_temp)

    def initial(self) \
        -> Tuple[NumDict[C], NumDict[R], NumDict[R], NumDict[F]]:
        _, _, prms = self.boltzmann.initial()
        return NumDict(), NumDict(), NumDict(), prms

    def call(
        self, 
        p: NumDict[F], 
        cr: NumDict[Tuple[C, R]], 
        rc: NumDict[Tuple[R, C]], 
        d: NumDict[C]
    ) -> Tuple[NumDict[C], NumDict[R], NumDict[R], NumDict[F]]:
        """
        Select actions chunks through action rules.
        
        :param p: Selection parameters (threshold and temperature). See 
            self.params for expected parameter keys.
        :param cr: Chunk-to-rule associations (i.e., condition weights).
        :param rc: Rule-to-chunk associations (i.e., conclusion weights; 
            typically binary).
        :param d: Condition chunk strengths.
        """
        s_r = cr.mul_from(d, kf=sym.first).transform_keys(kf=sym.second)
        r, pr, prms = self.boltzmann.call(p, s_r)
        c = rc.mul_from(r, kf=sym.first).transform_keys(kf=sym.second)
        return c, r, pr, prms


class AssociativeRules(Process):
    """Activates chunks according to associative rules."""

    def initial(self) -> Tuple[NumDict[C], NumDict[R]]:
        return NumDict(), NumDict()

    def call(
        self, 
        cr: NumDict[Tuple[C, R]], 
        rc: NumDict[Tuple[R, C]], 
        d: NumDict[C]
    ) -> Tuple[NumDict[C], NumDict[R]]:
        """
        Propagate activations through associative rules.
        
        :param cr: Chunk-to-rule associations (i.e., condition weights).
        :param rc: Rule-to-chunk associations (i.e., conclusion weights; 
            typically binary).
        :param d: Condition chunk strengths.
        """
        s_r = (cr
            .mul_from(d, kf=sym.first, strict=True)
            .sum_by(kf=sym.second))
        s_c = (rc
            .mul_from(s_r, kf=sym.first, strict=True)
            .max_by(kf=sym.second))
        return s_c, s_r
