"""
An introductory demo for the pyClarion library.

To be studied after `choice_model.py`.

Showcases the Clarion architecture's dual representational structure in a 
phoneme detection task. Phonemes are encoded as vectors of implicit features 
taken from Miller & Nicely (1955).

These encodings are represented within the model as a collection of phoneme 
chunks. Each trial involves presenting a phoneme chunk, propagating feature 
activations top-down, activating its implicit representation. Implicit feature 
activations then propagate bottom-up, generating chunk activations that reflect 
the directed similarity between the stimulus and each phoneme chunk in memory. 
The model then stochastically retrieves a phoneme chunk based on the resulting 
activations to complete the trial.
"""


import enum
import logging

from textwrap import indent

from pyClarion.numdicts import Dyad
from pyClarion.knowledge import (Root, DataFamily, ParamFamily, ChunkFamily, 
    BusFamily, Buses, Atoms, Chunks, Bus, Atom, Chunk) 
from pyClarion.components import (Source, Input, Polytomous, Retrieval, 
    RetrievalParams, ChunkStore, TopDown, BottomUp, ChunkStoreParams)
from pyClarion.events import (Event, Process, Start, Transition, Stop, Fluent, 
    Status, Action)

class Voicing(Atoms):
    """A collection of voicing values."""
    no: Atom
    yes: Atom

class Nasality(Atoms):
    """A collection of nasality values"""
    no: Atom
    yes: Atom

class Affrication(Atoms):
    """A collection of affrication values"""
    no: Atom
    yes: Atom

class Duration(Atoms):
    """A collection of duration values"""
    short: Atom
    long: Atom

class Place(Atoms):
    """A collection of place values"""
    front: Atom
    middle: Atom
    back: Atom

class Implicit(DataFamily):
    """The family of implicit (micro)features defined by the model"""
    voicing: Voicing
    nasality: Nasality
    affrication: Affrication
    duration: Duration
    place: Place


class WM(Bus):
    """A working memory bus"""
    pass

class Main(Buses):
    """The collection of main model buses"""
    wm: WM

class Layout(BusFamily):
    """The model's structural layout"""
    main: Main


class Phonemes(Chunks[Dyad[WM, Implicit]]):
    """A collection of phoneme chunks"""
    p: Chunk; t: Chunk; k: Chunk
    f: Chunk; θ: Chunk; s: Chunk; ʃ: Chunk
    b: Chunk; d: Chunk; g: Chunk
    v: Chunk; ð: Chunk; z: Chunk; ʒ: Chunk
    m: Chunk; n: Chunk        


class Explicit(ChunkFamily):
    """The family of explicit chunks defined by the model"""
    phonemes: Phonemes

class Parameters(ParamFamily):
    """The family of model parameters"""
    retrieval: RetrievalParams
    chunks: ChunkStoreParams

class ModelKeySpace(Root):
    """The complete model keyspace"""
    b: Layout
    d: Implicit
    c: Explicit
    p: Parameters

    def __init__(self) -> None:
        super().__init__()

        # Initialization of phoneme chunks

        main = self.b.main
        vo = self.d.voicing
        na = self.d.nasality
        af = self.d.affrication
        du = self.d.duration
        pl = self.d.place

        ph = Phonemes(main.wm * self.d)
        self.c["phonemes"] = ph

        ph["p"] = main.wm ** (vo.no, na.no, af.no, du.short, pl.front)
        ph["t"] = main.wm ** (vo.no, na.no, af.no, du.short, pl.middle)
        ph["k"] = main.wm ** (vo.no, na.no, af.no, du.short, pl.back)

        ph["f"] = main.wm ** (vo.no, na.no, af.yes, du.short, pl.front)
        ph["θ"] = main.wm ** (vo.no, na.no, af.yes, du.short, pl.middle)
        ph["s"] = main.wm ** (vo.no, na.no, af.yes, du.long, pl.middle)
        ph["ʃ"] = main.wm ** (vo.no, na.no, af.yes, du.long, pl.back)
        
        ph["b"] = main.wm ** (vo.yes, na.no, af.no, du.short, pl.front)
        ph["d"] = main.wm ** (vo.yes, na.no, af.no, du.short, pl.middle)
        ph["g"] = main.wm ** (vo.yes, na.no, af.no, du.short, pl.back)

        ph["v"] = main.wm ** (vo.yes, na.no, af.yes, du.short, pl.front)
        ph["ð"] = main.wm ** (vo.yes, na.no, af.yes, du.short, pl.middle)
        ph["z"] = main.wm ** (vo.yes, na.no, af.yes, du.long, pl.middle)
        ph["ʒ"] = main.wm ** (vo.yes, na.no, af.yes, du.long, pl.back)

        ph["m"] = main.wm ** (vo.yes, na.yes, af.yes, du.short, pl.front)
        ph["n"] = main.wm ** (vo.yes, na.yes, af.yes, du.short, pl.middle)

        ph._initialize_required_()


class Presentation[P: Source](Input[P]):
    """A simulated stimulus presentation."""
    pass


class Agent(Process[ModelKeySpace]):
    """A simulated participant."""

    source: Source[Phonemes, float]
    chunks: ChunkStore[Phonemes, ChunkStoreParams]
    ret: Polytomous[Phonemes, RetrievalParams]

    def __init__(self, name: str, root: ModelKeySpace | None = None) -> None:
        super().__init__(name, root)
        c = self.system.root.c
        p = self.system.root.p
        with self:
            self.source = Source(f"{name}.source", c.phonemes)
            self.chunks = ChunkStore(f"{name}.chunks", p.chunks, c.phonemes)
            self.ret = Polytomous(f"{name}.ret", p.retrieval, c.phonemes)
        self.chunks.input_td = self.source.output
        self.chunks.input_bu = self.chunks.output_td
        self.ret.input = self.chunks.output_bu

    def resolve(self, event: Event) -> None:
        match self.status:
            case Status.FREE:
                self.resolve_free(event)
            case _:
                raise ValueError(f"Unexpected event {event}")

    def resolve_free(self, event: Event) -> None:
        match event.action:
            case Presentation(p) if p is self.source:
                self.system.schedule(TopDown(self.chunks))
            case TopDown(p) if p is self.chunks:
                self.system.schedule(BottomUp(self.chunks))
                self.system.logger.info(event.describe())
            case BottomUp(p) if p is self.chunks:
                self.system.schedule(Retrieval(self.ret))
                self.system.logger.info(event.describe())
            case _:
                raise ValueError(f"Unexpected event {event}")


class ModelStatus(Fluent):
    """
    Represents model state.
    
    The ITEM fluent indicates that an item is awaiting a response.
    """
    ITEM = enum.auto()

class Model(Process[ModelKeySpace]):
    """
    A simulated task environment.

    This process houses the agent and manages stimulus presentations.
    """
    agent: Agent
    item: int
    items: list[Chunk]

    def __init__(self, 
        name: str, 
        root: ModelKeySpace, 
        stimuli: list[Chunk]
    ) -> None:
        super().__init__(name, root)
        with self:
            self.agent = Agent(f"{name}.agent")
        self.item = 0
        self.items = stimuli

    def resolve(self, event: Event) -> None:
        """Resolve the current event."""
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
                self._log(self._start_report(event))
                self.item = 0
            case Transition(p, Status.FREE) if p is self:
                self.item += 1
            case Stop(p) if p is self:
                self._log(self._stop_report(event))
                return
            case _:
                raise ValueError(f"Unexpected event {event}")
        if self.item < len(self.items):
            self.system.schedule(Transition(self, ModelStatus.ITEM))
        else:
            action = Stop(self)
            self.system.schedule(action)

    def resolve_item(self, event: Event) -> None:
        match event.action:
            case Transition(p, ModelStatus.ITEM) if p is self:
                item = self.items[self.item]
                source = self.agent.source
                self.system.schedule(Presentation(source, {item: 1.0}))
            case Presentation(p) if p is self.agent.source:
                self._log(self._presentation_report(event))
                self.agent.resolve(event)
            case Retrieval(p) if p is self.agent.ret:
                self._log(self._retrieval_report(event))
                self.system.schedule(Transition(self, Status.FREE))
            case Action(p) if p in self.agent:
                self.agent.resolve(event)
            case _:
                raise ValueError(f"Unexpected event {event}")

    def _log(self, message: str) -> None:
        self.system.logger.info(message)

    def _event_report(self, event: Event) -> str:
        wall_time = self.system.clock.wall_time
        message = "\n".join([
            event.describe(),
            indent(f"Wall time is {wall_time}.", "    ")])
        return message

    def _start_report(self, event: Event[Start]) -> str:
        head = self._event_report(event)
        tail = "\n".join([
            "Welcome to pyClarion!",
            "This demo presents some more advanced library features.",
            "It also showcases Clarion's dual-representational structure.",
            "The model simulates phoneme detection.",
            "Phonemes are encoded using features from Miller & Nicely (1955).",
            "A phoneme is detected if the corresponding chunk is retrieved.",
            "Retrieval involves implicit-explicit knowledge interaction.",
            "This interaction drives competition among phoneme chunks."])
        message = "\n".join([head, indent(tail, "    ")])
        return message

    def _presentation_report(self, event: Event[Presentation]) -> str:
        item = self.item
        head = self._event_report(event)
        tail = f"Presenting phoneme {~self.items[item]}."
        message = "\n".join([head, indent(tail, "    ")])
        return message

    def _retrieval_report(self, event: Event[Retrieval]) -> str:
        choice = str(self.agent.ret.output[0])
        sample = str(self.agent.ret.sample[0])
        head = self._event_report(event)
        tail = "\n".join([
            "Agent retrieved a phoneme chunk.",
            "Here is the selection vector:",
            indent(choice, "    "),
            "Here is the corresponding activation vector:",
            indent(sample, "    ")])
        message = "\n".join([head, indent(tail, "    ")])
        return message

    def _stop_report(self, event: Event[Stop]) -> str:
        head = self._event_report(event)
        tail = "Simulation complete."
        message = "\n".join([head, indent(tail, "    ")])
        return message

def main() -> None:
    """Initialize a choice model instance and run it on a small item set."""
    ks = ModelKeySpace()
    ks.p.retrieval.sd.init_val = 1e-1
    ks.p.retrieval.offset.init_val = .2
    ks.p.retrieval.scale.init_val = 2
    stimuli = [ks.c.phonemes.p, ks.c.phonemes.ʒ]
    logging.basicConfig(
        filename="similarity_model_log.txt", 
        filemode="w", 
        format="%(message)s",
        level=logging.INFO)
    model = Model("model", ks, stimuli)
    model.system.run(sync=1)


if __name__ == "__main__":
    main()