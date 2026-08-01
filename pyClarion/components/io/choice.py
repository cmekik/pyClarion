__all__ = ["ChoiceParams", "Choice", "Selection"]


from ..base.parametric import Parametric
from ...events.system import Action
from ...events.sites import State, Site 
from ...events.updates import ForwardUpdate
from ...knowledge.types import Nodes
from ...knowledge.sorts import Params
from ...knowledge.terms import Param, Term, Bus
from ...numdicts.keys import KeyForm
from ...numdicts.numdicts import NumDict
from ...numdicts.keyspaces import Dyad, keyform, crawl


class ChoiceParams(Params):
    sd: Param
    _init_val_defaults_ = {"sd": 1.0}


class Choice[D: Nodes, P: ChoiceParams](Parametric[P]):
    """
    A choice process.
    
    Makes discrete stochastic decisions based on activation strengths.
    """
    d: D
    p: P
    by: KeyForm
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
        self.by = self._init_by(d)

    @staticmethod
    def _init_by(s: D) -> KeyForm:
        match s:
            case Dyad():
                (d, v) = s.paths
                return keyform(d) * keyform(v, -1)
            case s:
                return keyform(s, -1)

    def poll(self) -> list[Term] | list[Dyad[Bus, Term]]:
        """Return a symbolic representation of current decision."""
        d = self.main[0].argmax(by=self.by)
        choices = []
        for v in d.values():
            choices.append(crawl(self.system.root, v))
        return choices


class Selection[C: Choice](Action[C]):
    __slots__ = ("sample",)
    sample: NumDict
    def __init__(self, source: C) -> None:
        super().__init__(source)
        input = source.main.new({}).sum(source.input[0])
        sd = source.main.new({}, c=source.params[0][~source.p.sd])
        sample = input.normalvariate(sd)
        choices = sample.argmax(by=source.by)
        self.updates.extend([
            ForwardUpdate(source.main, {v: 1.0 for v in choices.values()}),
            ForwardUpdate(source.sample, sample)])
        self.sample = sample
