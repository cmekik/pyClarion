__all__ = ["DiscriminalParams", "Discriminal", "DiscriminalSelection"]


from ..base.parametric import Parametric
from ...events.system import Action
from ...events.sites import State, Site
from ...events.updates import ForwardUpdate
from ...knowledge.types import Nodes 
from ...knowledge.sorts import Params
from ...knowledge.terms import Param
from ...numdicts.numdicts import numdict
from ...numdicts.keyspaces import keyform


class DiscriminalParams(Params):
    sd: Param
    th: Param
    _init_val_defaults_ = {"sd": 1.0, "th": .32905}


class Discriminal[D: Nodes, P: DiscriminalParams](Parametric[P]):
    """
    A discriminal process.
    
    Makes discrete stochastic decisions based on activation strengths.
    """
    d: D
    p: P
    main: Site[float] = Site()
    input: Site[float] = Site(lax=True)
    sample: Site[float] = Site()

    def __init__(self, name: str, p: P, d: D) -> None:
        super().__init__(name)
        self.system.check_root(d)
        self._init_parametric(p)
        i, l = self.system.get_index(keyform(d)), self.system.depth
        self.d = d
        self.main = State(i, {}, 0.0, l)
        self.input = State(i, {}, 0.0, l)
        self.sample = State(i, {}, 0.0, l)


class DiscriminalSelection[D: Discriminal](Action[D]):
    __slots__ = ()
    def __init__(self, source: D) -> None:
        super().__init__(source)
        input = source.main.new({}).sum(source.input[0])
        sd = numdict(source.main.index, {}, c=source.params[0][~source.p.sd])
        sample = input.normalvariate(sd)
        th = source.params[0][~source.p.th]
        pos = sample.isbetween(lb=th)
        neg = sample.isbetween(ub=-th)
        choices = pos.sub(neg)
        self.updates.extend([
            ForwardUpdate(source.main, choices),
            ForwardUpdate(source.sample, sample)])
