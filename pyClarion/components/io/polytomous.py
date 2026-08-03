__all__ = ["PolytomousParams", "Polytomous", "PolytomousSelection"]


import math

from typing import Any
from datetime import timedelta

from ..base.parametric import Parametric
from ...events.system import Action
from ...events.sites import State, Site 
from ...events.updates import ForwardUpdate
from ...knowledge.types import Nodes
from ...knowledge.sorts import Params
from ...knowledge.terms import Param, Term, Bus
from ...numdicts.keys import KeyForm
from ...numdicts.keyspaces import Dyad, keyform, crawl


class PolytomousParams(Params):
    sd: Param
    _init_val_defaults_ = {"sd": 1.0}


class Polytomous[D: Nodes, P: PolytomousParams](Parametric[P]):
    """
    A polytomous choice process.
    
    Makes discrete stochastic k-way choices based on activation strengths.
    """
    d: D
    p: P
    by: KeyForm
    input: Site[float] = Site(lax=True)
    output: Site[float] = Site()
    sample: Site[float] = Site()

    def __init__(self, name: str, p: P, d: D) -> None:
        super().__init__(name)
        self.system.check_root(d)
        self._init_parametric(p)
        i, l = self.system.get_index(keyform(d)), self.system.depth
        self.d = d
        self.input = State(i, {}, 0.0, l)
        self.output = State(i, {}, 0.0, l)
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
        d = self.output[0].argmax(by=self.by)
        choices = []
        for v in d.values():
            choices.append(crawl(self.system.root, v))
        return choices


class PolytomousSelection[C: Polytomous](Action[C]):
    __slots__ = ()
    def __init__(self, source: C) -> None:
        super().__init__(source)
        input = source.output.new({}).sum(source.input[0])
        sd = source.output.new({}, c=source.params[0][~source.p.sd])
        sample = input.normalvariate(sd)
        choices = sample.argmax(by=source.by)
        self.updates.extend([
            ForwardUpdate(source.output, {v: 1.0 for v in choices.values()}),
            ForwardUpdate(source.sample, sample)])


class RetrievalParams(PolytomousParams):
    _init_val_defaults_ = {"offset": 0.0, "scale": 1.0, 
        **PolytomousParams._init_val_defaults_}
    offset: Param
    scale: Param


class Retrieval[C: Polytomous[Any, RetrievalParams]](PolytomousSelection[C]):
    __slots__ = ()
    def __init__(self, source: C) -> None:
        super(PolytomousSelection, self).__init__(source)
        input = source.output.new({}).sum(source.input[0])
        sd = source.output.new({}, c=source.params[0][~source.p.sd])
        sample = input.normalvariate(sd)
        choices = sample.argmax(by=source.by)   
        offset = source.params[0][~source.p.offset]
        scale = source.params[0][~source.p.scale]     
        seconds = offset + scale * math.exp(-sample.valmax())
        self.updates.extend([
            ForwardUpdate(source.output, {v: 1.0 for v in choices.values()}),
            ForwardUpdate(source.sample, sample)])
        self.duration = timedelta(seconds=seconds)
