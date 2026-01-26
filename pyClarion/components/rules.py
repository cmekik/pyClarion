from typing import Sequence
from datetime import timedelta
import logging

from .base import Component, Priority, ChunkUpdate, RuleUpdate
from .layers import Layer
from ..events import State, Site, Event, ForwardUpdate
from ..knowledge import Rules, Rule, Family, Chunks
from ..numdicts import keyform, NumDict
from ..numdicts.ops.base import Unary


class RuleStore(Component):
    """
    A rule store. 

    Maintains a collection of rules.
    """

    r: Rules
    lhs: Chunks
    rhs: Chunks
    b: float
    bias: Site = Site()
    riw: Site = Site()
    lhw: Site = Site()
    rhw: Site = Site()

    def __init__(self, 
        name: str, 
        r: Family,
        lhs: Chunks,
        rhs: Chunks,
        *,
        b: float = float("-inf")
    ) -> None:
        super().__init__(name)
        self.system.check_root(r, lhs, rhs)
        self.r = Rules(); r[name] = self.r
        self.lhs = lhs
        self.rhs = rhs
        self.b = b
        idx_r = self.system.get_index(keyform(self.r))
        idx_lhs = self.system.get_index(keyform(lhs))
        idx_rhs = self.system.get_index(keyform(rhs))
        self.main = State(idx_r, {}, c=0.0)
        self.bias = State(idx_r, {}, c=0.0)
        self.riw = State(idx_r * idx_r, {}, c=0.0)
        self.lhw = State(idx_lhs * idx_r, {}, c=0.0)
        self.rhw = State(idx_r * idx_rhs, {}, c=0.0)

    def resolve(self, event: Event) -> None:
        updates = event.index(RuleUpdate).get(self.r, [])
        new_rules = [rule for ud in updates for rule in ud.add]
        if new_rules:
            if event.source == self.encode \
                and self.system.logger.isEnabledFor(logging.DEBUG):
                self.log_encoding(new_rules)
            self.system.schedule(self.encode_weights(*new_rules))

    def log_encoding(self, rules: Sequence[Rule]) -> None:
        data = [f"    Added the following new rule(s)"]
        for r in rules:
            data.append(str(r).replace("\n", "\n    "))
        self.system.logger.debug("\n    ".join(data))

    def encode(self, 
        *rules: Rule, 
        dt: timedelta = timedelta(),
        priority: int = Priority.LEARNING
    ) -> Event:
        """Encode a collection of new rules."""
        new_rules = []
        new_lhs_chunks = []
        new_rhs_chunks = []
        for rule in rules:
            if hasattr(rule, "_parent_"):
                if rule not in self.system.root:
                    raise ValueError(f"The following rule belongs to a "
                        f"different system:\n {rule}")
                continue
            name = next(self.r._namer_)
            if not hasattr(rule, "_name_"):
                rule._name_ = name
            for i, chunk in enumerate(rule._chunks_):
                chunk_instances = list(chunk._instantiations_())
                chunk._instances_.update(chunk_instances)
                if self.lhs is self.rhs:
                    new_lhs_chunks.append(chunk)
                    new_lhs_chunks.extend(chunk_instances)
                elif i < len(rule._chunks_) - 1:
                    new_lhs_chunks.append(chunk)
                    new_lhs_chunks.extend(chunk_instances)
                else:
                    new_rhs_chunks.append(chunk)
                    new_rhs_chunks.extend(chunk_instances)
            rule_instances = list(rule._instantiations_())
            rule._instances_.update(rule_instances)
            new_rules.append(rule)
            new_rules.extend(rule_instances)
        updates = [
            RuleUpdate(self.r, add=tuple(new_rules)),
            ChunkUpdate(self.lhs, add=tuple(new_lhs_chunks))]
        if new_rhs_chunks:
            updates.append(ChunkUpdate(self.rhs, add=tuple(new_rhs_chunks)))
        return Event(self.encode, updates, dt, priority)

    def encode_weights(self, 
        *rules: Rule, 
        dt: timedelta = timedelta(), 
        priority=Priority.LEARNING
    ) -> Event:
        bias, riw, lhw, rhw = {}, {}, {}, {}
        for rule in rules:
            data = rule._compile_()
            if 0 < len(rule._vars_):
                bias[~rule] = self.b
            riw.update(data["riw"])
            lhw.update(data["lhw"])
            rhw.update(data["rhw"])
        return Event(self.encode_weights,
            [ForwardUpdate(self.bias, bias, "write"),
             ForwardUpdate(self.riw, riw, "write"),
             ForwardUpdate(self.lhw, lhw, "write"),
             ForwardUpdate(self.rhw, rhw, "write")],
            dt, priority)
    
    def lhs_layer(self, 
        name: str, 
        *, 
        func: Unary[NumDict] | None = None, 
        l: int = 1
    ) -> Layer[Chunks, Rules]:
        layer = Layer(name, self.lhs, self.r, func=func, l=l)
        layer.weights = self.lhw
        layer.bias = self.bias
        return layer

    def rhs_layer(self, 
        name: str, 
        *, 
        func: Unary[NumDict] | None = None, 
        l: int = 1
    ) -> Layer[Rules, Chunks]:
        layer = Layer(name, self.r, self.rhs, func=func, l=l)
        layer.weights = self.rhw
        return layer
    
    def riw_layer(self, 
        name: str, 
        *, 
        func: Unary[NumDict] | None = None, 
        l: int = 1
    ) -> Layer[Rules, Rules]:
        layer = Layer(name, self.r, self.r, func=func, l=l)
        layer.weights = self.riw
        return layer
