__all__ = ["DiscriminalParams", "Discriminal", "DiscriminalSelection"]


from ..base.parametric import Parametric
from ...events.system import Action
from ...events.sites import State, Site
from ...events.updates import ForwardUpdate
from ...knowledge.types import Nodes 
from ...knowledge.sorts import Params
from ...knowledge.terms import Param
from ...numdicts.numdicts import NumDict
from ...numdicts.keyspaces import keyform


class DiscriminalParams(Params):
    sd: Param
    thu: Param
    thl: Param
    _init_val_defaults_ = {"sd": 1.0, "thu": .32905, "thl": -.32905}


class Discriminal[D: Nodes, P: DiscriminalParams](Parametric[P]):
    """
    A discriminal choice process.
    
    Makes discrete stochastic decisions based on activation strengths.
    """
    input: Site[float] = Site(lax=True)
    output: Site[float] = Site()
    sample: Site[float] = Site()

    @property
    def d(self) -> D:
        return self._d

    def __init__(self, name: str, p: P, d: D) -> None:
        super().__init__(name, p)
        self.system.check_root(d)
        i, l = self.system.get_index(keyform(d)), self.system.depth
        self._d = d
        self.input = State(i, {}, 0.0, l)
        self.output = State(i, {}, 0.0, l)
        self.sample = State(i, {}, 0.0, l)


class DiscriminalSelection[D: Discriminal[Nodes, DiscriminalParams]](Action[D]):
    __slots__ = ()
    def __init__(self, source: D) -> None:
        super().__init__(source)
        input = source.output.new({}).sum(source.input[0])
        sd = source.output.new({}, c=source.params[0][~source.p.sd])
        sample = input.normalvariate(sd)
        thu = source.params[0][~source.p.thu]
        thl = source.params[0][~source.p.thl]
        pos = sample.isbetween(lb=thu)
        neg = sample.isbetween(ub=thl)
        choices = pos.sub(neg)
        self.updates.extend([
            ForwardUpdate(source.output, choices),
            ForwardUpdate(source.sample, sample)])
