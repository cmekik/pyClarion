"""
An introductory demo for the pyClarion library.

Implements and runs a simple model of multiple-choice item responses. 
"""


import enum
import logging
import random
import math

from datetime import timedelta
from textwrap import indent

from pyClarion.knowledge import Root, AtomFamily, ParamFamily, Atoms, Atom 
from pyClarion.components import Source, Choice, Input, Selection, ChoiceParams
from pyClarion.events import (Event, Process, Start, Transition, Stop, Fluent, 
    Status)


########################
# Keyspace Definitions #
########################

class Options(Atoms):
    """A collection of choice options."""
    a: Atom
    b: Atom
    c: Atom
    d: Atom    

class ModelData(AtomFamily):
    """Keyspace family housing all semantic simulation data."""
    options: Options

class ModelParams(ParamFamily):
    """Keyspace family housing all simulation parameters."""
    choice: ChoiceParams

class ModelKeySpace(Root):
    """Choice model root keyspace."""
    d: ModelData
    p: ModelParams


######################
# Action Definitions #
######################

class Response[P: Choice](Selection[P]):
    """
    A simulated item response.
    
    The response is modeled as a Selection instance, which stochastically 
    selects an option via a competition with noisy activations. The duration of 
    the response selection process is determined by the following latency 
    equation, where a* is the activation of the maximally activated element, F 
    is a scaling constant, and c is non-decision time: 
        latency = c + F * exp(-a*) 
    """
    def __init__(self, source: P, c: float, f: float) -> None:
        super().__init__(source)
        seconds = c + f * math.exp(-self.sample.valmax())
        self.duration = timedelta(seconds=seconds)

class Presentation[P: Source](Input[P]):
    """
    A simulated item presentation.
    
    The presentation is modeled as an Input instance. Its duration is determined 
    by an exponential distribution representing display latencies. 
    """
    def __init__(self, source: P, data: dict[Atom, float], rate: float) -> None:
        super().__init__(source, data)
        self.duration = timedelta(seconds=random.expovariate(rate))


#####################
# Agent Definitions #
#####################

class ChoiceAgent(Process[ModelKeySpace]):
    """
    A simulated respondent.

    This agent maintains an input source and a choice process, allowing it to 
    respond to item presentations.
    """
    source: Source[Options, float]
    choice: Choice[Options, ChoiceParams]

    def __init__(self, name: str, root: ModelKeySpace | None = None) -> None:
        super().__init__(name, root)
        d, p = self.system.root.d, self.system.root.p
        with self:
            self.source = Source(f"{name}.source", d.options)
            self.choice = Choice(f"{name}.choice", p.choice, d.options)
        self.choice.input = self.source.main

    def resolve(self, event: Event) -> None:
        """
        Resolve the current event.
        
        Issues a response if current event is a presentation, otherwise raises 
        ValueError.
        """
        match event.action:
            case Presentation(p) if p is self.source:
                self.system.schedule(Response(self.choice, c=.2, f=2))
            case _:
                raise ValueError(f"Unexpected event {event}")


################################
# Task Environment Definitions #
################################
    
class ModelStatus(Fluent):
    """
    Represents choice model state.
    
    The ITEM fluent indicates that an item is awaiting a response.
    """
    ITEM = enum.auto()

class ChoiceModel(Process[ModelKeySpace]):
    """
    A simulated task environment.

    This process houses the choice agent and manages stimulus presentations.
    """
    agent: ChoiceAgent
    item: int
    items: list[dict[Atom, float]]

    def __init__(self, 
        name: str, 
        root: ModelKeySpace, 
        stimuli: list[dict[Atom, float]]
    ) -> None:
        super().__init__(name, root)
        with self:
            self.agent = ChoiceAgent(f"{name}.agent")
        self.item = 0
        self.items = stimuli

    def resolve(self, event: Event) -> None:
        """
        Resolve the current event.
        
        The model can be in one of two valid states: Status.FREE or 
        ModelStatus.ITEM. In the FREE state, the model initiates presentation 
        of the next item by moving into the ITEM state or terminates. In the 
        ITEM state, the model presents the next item, delegates presentation 
        response to the agent, and moves back into the FREE state. 
        """
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
                self.system.schedule(
                    Presentation(source, item, rate=30))
            case Presentation(p) if p is self.agent.source:
                self._log(self._presentation_report(event))
                self.agent.resolve(event)
            case Response(p) if p is self.agent.choice:
                self._log(self._response_report(event))
                self.system.schedule(Transition(self, Status.FREE))
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
            "This model demonstrates many core features of the library.",
            "The model simulates multiple-choice item responses.",
            "Each item is displayed until a response is received.",
            "Responses are generated stochastically from option activations."])
        message = "\n".join([head, indent(tail, "    ")])
        return message

    def _presentation_report(self, event: Event[Presentation]) -> str:
        item = self.item
        head = self._event_report(event)
        tail = f"Presenting item {item}."
        message = "\n".join([head, indent(tail, "    ")])
        return message

    def _response_report(self, event: Event[Response]) -> str:
        choice = str(self.agent.choice.main[0])
        sample = str(self.agent.choice.sample[0])
        head = self._event_report(event)
        tail = "\n".join([
            "Agent successfully selected an option.",
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


##############
# Simulation #
##############

def main() -> None:
    """Initialize a choice model instance and run it on a small item set."""
    ks = ModelKeySpace()
    ks.p.choice.sd.init_val = 1e-1
    stimuli = [{ks.d.options.a: -1.0}, {ks.d.options.c: .3}]
    logging.basicConfig(
        filename="choice_model_log.txt", 
        filemode="w", 
        format="%(message)s",
        level=logging.INFO)
    model = ChoiceModel("model", ks, stimuli)
    model.system.run(sync=1)


if __name__ == "__main__":
    main()
