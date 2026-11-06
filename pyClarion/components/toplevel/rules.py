__all__ = ["RuleStore"]


from typing import cast, Final
from datetime import timedelta 

import random

from ..base.parametric import ParamUpdate
from ..io.polytomous import Polytomous, PolytomousParams, PolytomousSelection
from ...events import State, Site, Action, Event, Process, ForwardUpdate, Start, Stop
from ...knowledge.base import Root
from ...knowledge.implicit import Param
from ...knowledge.explicit import Rules 
from ...numdicts.keyspaces import keyform, find_root
from ...numdicts.keys import KeyForm


class RuleStore[R: Rules](Process[Root]):
    """
    A rule store. 

    Maintains a collection of rules and facilitates rule-based processing.
    """

    r: Final[R]
    lhs_input: Site = Site()
    lhs_output: Site = Site()
    rhs_input: Site = Site()
    rhs_output: Site = Site()
    riw: Site = Site()
    lhw: Site = Site()
    rhw: Site = Site()
    by_r: KeyForm
    by_rhs: KeyForm
    by_lhs: KeyForm

    def __init__(self, name: str, r: R) -> None:
        super().__init__(name, cast(Root | None, find_root(r)))
        l = self.system.depth
        lhs, rhs = r._lhs_, r._rhs_
        self.system.check_root(r, lhs, rhs)
        self.r = r
        self.lhs = lhs
        self.rhs = rhs
        idx_r = self.system.get_index(keyform(self.r))
        idx_lhs = self.system.get_index(keyform(lhs))
        idx_rhs = self.system.get_index(keyform(rhs))
        riw, lhw, rhw = self.initialize_rule_weights(r)
        self.lhs_input = State(idx_lhs, {}, 0.0, l)
        self.lhs_output = State(idx_r, {}, 0.0, l)
        self.rhs_input = State(idx_r, {}, 0.0, l)
        self.rhs_output = State(idx_rhs, {}, 0.0, l)
        self.riw = State(idx_r * idx_r, riw, 0.0, l)
        self.lhw = State(idx_lhs * idx_r, lhw, 0.0, l)
        self.rhw = State(idx_r * idx_rhs, rhw, 0.0, l)
        self.by_r = keyform(r)
        self.by_rhs = keyform(rhs)
        self.by_lhs = keyform(lhs)

    def initialize_rule_weights(self, rules: Rules):
        riw, lhw, rhw = {}, {}, {}
        for r in rules:
            rule = rules[r]
            _riw, _lhw, _rhw = rule.compile()
            riw.update(_riw)
            lhw.update(_lhw)
            rhw.update(_rhw)
        return riw, lhw, rhw

    def resolve(self, event: Event) -> None:
        match event.action:
            case LHSPropagation(p) | RHSPropagation(p) if p is self:
                pass
            case _:
                raise ValueError(f"Unexpected event {event}")


type AnyRuleStore = RuleStore[Rules]


class LHSPropagation[S: AnyRuleStore](Action[S]):
    __slots__ = ()
    def __init__(self, source: S) -> None:
        super().__init__(source)
        output = source.lhw[0].mul(source.lhs_input[0]).sum(by=source.by_r)
        self.updates.append(ForwardUpdate(source.lhs_output, output))


class RHSPropagation[S: AnyRuleStore](Action[S]):
    __slots__ = ()
    def __init__(self, source: S) -> None:
        super().__init__(source)
        output = source.rhw[0].mul(source.rhs_input[0]).sum(by=source.by_rhs)
        self.updates.append(ForwardUpdate(source.rhs_output, output))


class AssociativeRuleSystem[R: Rules](Process[Root]):

    store: Final[RuleStore[R]]

    @property
    def input(self) -> State:
        return self.store.lhs_input

    @input.setter
    def input(self, value: State) -> None:
        self.store.lhs_input = value

    @property
    def output(self) -> State:
        return self.store.rhs_output

    @output.setter
    def output(self, value: State) -> None:
        self.store.rhs_output = value

    def __init__(self, name: str, r: R, latency: float) -> None:
        super().__init__(name, cast(Root | None, find_root(r)))
        self.latency = latency
        with self:
            self.store = RuleStore(f"{name}.store", r)
        self.store.rhs_input = self.store.lhs_output

    def resolve(self, event: Event) -> None:
        match event.action:
            case Start(p) if p is self:
                self.system.schedule(LHSPropagation(self.store)) 
            case LHSPropagation(p) if p is self.store:
                self.system.schedule(RHSPropagation(self.store))
            case RHSPropagation(p) if p is self.store:
                self.system.schedule(Stop(self))
            case _:
                raise ValueError(f"Unexpected event {event}")


class RuleParams(PolytomousParams):
    _init_val_defaults_ = {"latency": .05}
    latency: Param
        

class ActionRuleSystem[R: Rules, P: RuleParams](Process[Root]):

    store: Final[RuleStore[R]]
    selector: Final[Polytomous[R, P]]

    signal: Site = Site()
    
    by: KeyForm

    @property
    def input(self) -> State:
        return self.store.lhs_input

    @input.setter
    def input(self, value: State) -> None:
        self.store.lhs_input = value

    @property
    def output(self) -> State:
        return self.store.rhs_output

    @output.setter
    def output(self, value: State) -> None:
        self.store.rhs_output = value

    def __init__(self, name: str, p: P, r: R) -> None:
        super().__init__(name, cast(Root | None, find_root(r)))
        idx_r = self.system.get_index(keyform(r))
        self.signal = State(idx_r, {}, 0.0, self.system.depth)
        self.by_mul = keyform(r) * keyform(r).agg
        self.by = keyform(r).agg * keyform(r)
        with self:
            self.store = RuleStore(f"{name}.store", r)
            self.selector = Polytomous(f"{name}.selector", p, r)
        self.selector.input = self.store.lhs_output
        self.store.rhs_input = self.selector.output

    def resolve(self, event: Event) -> None:
        match event.action:
            case Start(p) if p is self:
                self.system.schedule(LHSPropagation(self.store))
            case LHSPropagation(p) if p is self.store:
                self.system.schedule(RuleSelection(self.selector))
            case RuleSelection(p) if p is self.selector:
                self.system.schedule(RuleSignaling(self))
            case RuleSignaling(p) if p is self:
                self.system.schedule(RHSPropagation(self.store))
            case RHSPropagation(p) if p is self.store:
                self.system.schedule(Stop(self))
            case ParamUpdate(p) if p in self:
                pass
            case _:
                raise ValueError(f"Unexpected event {event}")


class RuleSelection[S: Polytomous[Rules, RuleParams]](PolytomousSelection[S]):
    __slots__ = ()

    def __init__(self, source: S) -> None:
        latency = source.params[0][~source.p.latency] 
        duration = timedelta(seconds=random.expovariate(1 / latency))
        super(PolytomousSelection, self).__init__(source, duration=duration)
        self.updates.extend(PolytomousSelection(source).updates)


class RuleSignaling[S: ActionRuleSystem[Rules, RuleParams]](Action[S]):
    __slots__ = ()

    def __init__(self, source: S) -> None:
        signal = (source.store.riw[0]
            .mul(source.selector.output[0], by=source.by_mul)
            .sum(by=source.by))
        super().__init__(source)
        self.updates.append(ForwardUpdate(source.signal, signal))