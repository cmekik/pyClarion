import enum
import logging

from textwrap import indent

from pyClarion.components.io.pool import Pool, PoolPropagation
from pyClarion.components.toplevel.rules import ActionRuleSystem, RuleParams, RuleSelection
from pyClarion.components import (Input, Source, PolytomousParams, 
    ChunkStore, BottomUp, TopDown, Accumulator, AccumulatorUpdate, 
    AccumulatorReset, Polytomous, PolytomousSelection)
from pyClarion.events import (Process, Event, Start, Stop, Action, State, 
    Status, Fluent, Transition)
from pyClarion.knowledge import DataFamily, this, var, rule , Params
from pyClarion.knowledge import (Root, ParamFamily, BusFamily, Atoms, Buses, 
    Chunks, Rules, Atom, Bus, get_chunks, Term) 
from pyClarion.numdicts import Key, NumDict, Dyad


class Shape(Atoms):
    circle: Atom
    square: Atom
    triangle: Atom


class Color(Atoms):
    red: Atom
    green: Atom
    blue: Atom


class Match(Atoms):
    by_dim: Atom
    by_val: Atom


class Evaluation(Atoms):
    false: Atom
    true: Atom


class Main(Buses):
    wm: Bus
    gs: Bus


class Layout(BusFamily):
    main: Main


class Implicit(DataFamily):
    shape: Shape
    color: Color
    match: Match
    eval: Evaluation
    rules: Rules


class Explicit(DataFamily):
    lhs: Chunks
    rhs: Chunks


class Parameters(ParamFamily):
    rules: RuleParams
    action: PolytomousParams
    pool: Params


class KeySpace(Root):
    b: Layout
    d: Implicit
    e: Explicit
    p: Parameters

    def __init__(self) -> None:
        super().__init__()
        b, d, e = self.b, self.d, self.e
        e.lhs = Chunks(b.main * d)
        e.rhs = Chunks(b.main.wm * d.eval)
        d.rules = Rules(e.lhs, e.rhs)

        with d.rules:

            # Existential WM Query / Dimensional Test
            # (for each dimension, ensure at least one goal feature is in wm)
    
            dim_match_found = rule(
                "dim_match_found" ^
                + b.main.gs ** d.match.by_dim
                + b.main.gs ** var(d)
                + b.main.wm ** var(d)
                - b.main.wm ** this.rule_group
                >>
                + b.main.wm ** d.eval._
            )
            rule(
                "dim_match_false" ^
                + b.main.gs ** d.match.by_dim
                + b.main.gs ** var(d).dim
                - b.main.wm ** dim_match_found(var(d).dim)
                >>
                + b.main.wm ** d.eval.false
            )
            rule(
                "dim_match_true" ^
                + b.main.gs ** d.match.by_dim
                >>
                + b.main.wm ** d.eval.true
            )

            # Universal WM Query / Value Test 
            # (for each dimension, every goal feature is in wm)

            val_match_found = rule(
                "val_match_found" ^
                + b.main.gs ** d.match.by_val
                + b.main.gs ** var(d)
                + b.main.wm ** var(d)
                - b.main.wm ** this.rule_instance
                >>
                + b.main.wm ** d.eval._)
            rule(
                "val_match_false" ^
                + b.main.gs ** d.match.by_val
                + b.main.gs ** var(d)
                - b.main.wm ** val_match_found(var(d))
                >>
                + b.main.wm ** d.eval.false)
            rule(
                "val_match_true" ^
                +  b.main.gs ** d.match.by_val
                >>
                + b.main.wm ** d.eval.true)


class RuleAgent(Process[KeySpace]):

    src: Source
    ctx: Accumulator
    lhs: ChunkStore
    rhs: ChunkStore
    pool: Pool
    rules: ActionRuleSystem[Rules, RuleParams]
    action: Polytomous

    @property
    def input(self) -> State:
        return self.lhs.input_bu

    @input.setter
    def input(self, state: State) -> None:
        self.lhs.input_bu = state

    def __init__(self, name: str, root: KeySpace | None = None) -> None:
        if root is None:
            root = KeySpace()
        super().__init__(name, root)
        ks = self.system.root
        rules = ks.d.rules
        lhs, rhs = get_chunks(rules)
        dyad_ipt = ks.b.main * ks.d
        dyad_ctx = ks.b.main.wm * rules
        dyad_out = ks.b.main.wm * ks.d.eval
        with self:
            self.src = Source(f"{name}.src", dyad_ipt)
            self.ctx = Accumulator(f"{name}.ctx", dyad_ctx)
            self.lhs = ChunkStore(f"{name}.lhs", lhs)
            self.rhs = ChunkStore(f"{name}.rhs", rhs)
            self.pool = Pool(f"{name}.pool", ks.p.pool, dyad_ipt)
            self.rules = ActionRuleSystem(f"{name}.rules", ks.p.rules, rules)
            self.action = Polytomous(f"{name}.action", ks.p.action, dyad_out)
        self.pool <<= self.src
        self.pool <<= self.ctx
        self.lhs.input_bu = self.pool.output
        self.rules.input = self.lhs.output_bu
        self.rhs.input_td = self.rules.output
        self.ctx.input = self.rules.signal
        self.action.input = self.rhs.output_td

    def resolve(self, event: Event) -> None:
        match event.action:
            case Start(p) if p is self:
                self.system.schedule(PoolPropagation(self.pool, NumDict.sum))
            case PoolPropagation(p) if p is self.pool:
                self.system.schedule(BottomUp(self.lhs))
            case BottomUp(p) if p is self.lhs:
                self.system.schedule(Start(self.rules))
            case Stop(p) if p is self.rules:
                self.system.schedule(AccumulatorUpdate(self.ctx))
            case AccumulatorUpdate(p) if p is self.ctx:
                self.system.schedule(TopDown(self.rhs))
            case TopDown(p) if p is self.rhs:
                self.system.schedule(PolytomousSelection(self.action))
            case PolytomousSelection(p) if p is self.action:
                self.system.schedule(Stop(self))
            case Action(p) if p is self.rules or p in self.rules:
                self.rules.resolve(event)
            case _:
                raise ValueError(f"Unexpected event {event}")


class ModelStatus(Fluent):
    """
    Represents choice model state.
    
    The ITEM fluent indicates that an item is awaiting a response.
    """
    ITEM = enum.auto()


class RuleModel(Process[KeySpace]):
    agent: RuleAgent
    item: int
    items: list[dict[Dyad, float]]

    def __init__(self, 
        name: str, 
        root: KeySpace, 
        stimuli: list[dict[Dyad, float]]
    ) -> None:
        super().__init__(name, root)
        with self:
            self.agent = RuleAgent(f"{name}.agent", root)
        self.item = 0
        self.items = stimuli

    def resolve(self, event: Event) -> None:
        self._log(self._event_report(event))
        match self.status:
            case Status.FREE:
                self.resolve_free(event)
            case ModelStatus.ITEM:
                self.resolve_item(event)
            case _:
                raise ValueError(f"Unexpected event {event}")

    def resolve_free(self, event: Event) -> None:
        match event.action:
            case Start(p) if p is self:
                self.item = 0
            case Transition(p, Status.FREE) if p is self:
                self.item += 1
            case Stop(p) if p is self:
                return
            case _:
                raise ValueError(f"Unexpected event {event}")  
        if self.item < len(self.items):
            action = Transition(self, ModelStatus.ITEM)
        else:
            action = Stop(self)
        self.system.schedule(action) 

    def resolve_item(self, event: Event) -> None:
        match event.action:
            case Transition(p, ModelStatus.ITEM) if p is self:
                action = Input(self.agent.src, self.items[self.item])
                self.system.schedule(action)
            case Input(p) if p is self.agent.src:
                self._log(self._input_report(event))
                self.system.schedule(Start(self.agent))
            case Stop(p) if p is self.agent:
                ks = self.system.root
                choice = self.agent.action.poll()
                assert isinstance(choice[0], Dyad)
                if ks.d.eval._ is choice[0].units[1]:
                    action = Start(self.agent)
                else:
                    action = AccumulatorReset(self.agent.ctx)
                self.system.schedule(action)
            case AccumulatorReset(p) if p is self.agent.ctx:
                    self.system.schedule(Transition(self, Status.FREE))
            case RuleSelection(p) if p is self.agent.rules.selector:
                self._log(self._rule_selection_report(event))
                self.agent.resolve(event)
            case PolytomousSelection(p) if p is self.agent.action:
                self._log(self._action_report(event))
                self.agent.resolve(event)
            case Action(p) if p is self.agent or p in self.agent:
                self.agent.resolve(event)
            case _:
                raise ValueError(f"Unexpected event {event}")

    def _log(self, message: str) -> None:
        logging.info(message)
    
    def _event_report(self, event: Event) -> str:
        wall_time = self.system.clock.wall_time
        message = "\n".join([
            event.describe(),
            indent(f"Wall time is {wall_time}.", "    ")])
        return message

    def _input_report(self, event: Event[Input]) -> str:
        data = self.agent.src.output[0]
        head = "    Presenting the following input state."
        message = "\n".join([head, indent(str(data), "    ")])
        return message

    def _rule_selection_report(self, event: Event[PolytomousSelection]) -> str:
        choice = self.agent.rules.selector.output[0].top()
        sample = self.agent.rules.selector.sample[0].top(n=10)
        tail = "\n".join([
            "Agent selected an action rule.",
            "Here is the selection vector:",
            indent(choice, "    "),
            "Here is the corresponding activation vector:",
            indent(sample, "    ")])
        message = indent(tail, "    ")
        return message

    def _action_report(self, event: Event[PolytomousSelection]) -> str:
        choice = self.agent.action.output[0].top()
        sample = self.agent.action.sample[0].top(n=10)
        tail = "\n".join([
            "Agent selected an action.",
            "Here is the selection vector:",
            indent(choice, "    "),
            "Here is the corresponding activation vector:",
            indent(sample, "    ")])
        message = indent(tail, "    ")
        return message


def main() -> None:
    """Initialize a choice model instance and run it on a small item set."""
    ks = KeySpace()
    ks.p.rules.sd.init_val = 1e-3
    ks.p.action.sd.init_val = 1e-3
    stimuli = [
        {
            ks.b.main.gs * ks.d.match.by_dim: 1.0,
            ks.b.main.gs * ks.d.shape.circle: 1.0,
            ks.b.main.gs * ks.d.shape.square: 1.0,
            ks.b.main.gs * ks.d.color.green:  1.0,
            ks.b.main.wm * ks.d.shape.circle: 1.0,
            ks.b.main.wm * ks.d.color.green:  1.0
        },
        {
            ks.b.main.gs * ks.d.match.by_dim: 1.0,
            ks.b.main.gs * ks.d.shape.circle: 1.0,
            ks.b.main.gs * ks.d.shape.square: 1.0,
            ks.b.main.gs * ks.d.color.green:  1.0,
            ks.b.main.wm * ks.d.shape.circle: 1.0
        },
        {
            ks.b.main.gs * ks.d.match.by_val: 1.0,
            ks.b.main.gs * ks.d.shape.circle: 1.0,
            ks.b.main.gs * ks.d.shape.square: 1.0,
            ks.b.main.wm * ks.d.shape.circle: 1.0,
            ks.b.main.wm * ks.d.shape.square: 1.0
        },
        {
            ks.b.main.gs * ks.d.match.by_val: 1.0,
            ks.b.main.gs * ks.d.shape.circle: 1.0,
            ks.b.main.gs * ks.d.shape.square: 1.0,
            ks.b.main.wm * ks.d.shape.circle: 1.0,
        },
    ]
    logging.basicConfig(
        filename="rule_model_log.txt", 
        filemode="w", 
        format="%(message)s",
        level=logging.INFO)
    model = RuleModel("model", ks, stimuli)
    model.system.run()
    

if __name__ == "__main__":
    main()