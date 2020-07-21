"""Provides abstract classes for defining forward propagation cycles."""


__all__ = ["Propagator", "Cycle", "Assets"]


from pyClarion.base.symbols import Symbol, MatchSet, ConstructType
from types import MappingProxyType
from typing import (
    TypeVar, Generic, Mapping, Callable, Any, Mapping, Tuple, Set, Dict, 
    Sequence, Optional, no_type_check
)
from types import SimpleNamespace
from abc import abstractmethod


Dt = TypeVar('Dt') # type variable for inputs to processes
PullFuncs = Mapping[Symbol, Callable[[], Dt]]
Inputs = Mapping[Symbol, Dt]

It = TypeVar('It', contravariant=True) # type variable for emitter inputs
Ot = TypeVar('Ot', covariant=True) # type variable for emitter outputs


class Emitter(Generic[It, Ot]):
    """
    Base class for propagating strengths, decisions, etc.

    Emitters define how constructs process inputs and set outputs.
    """

    matches: MatchSet

    def __init__(self, matches: MatchSet = None):

        self.matches = matches if matches is not None else MatchSet()

    def expects(self, construct: Symbol):
        """Returns True if propagator expects input from given construct."""

        return construct in self.matches

    @abstractmethod
    def emit(self, data: Any = None) -> Ot:
        """
        Emit propagator output based on the return type of self.call().
        
        If no data is passed in, emits a default or null value of the expected
        output type. If data is passed in ensures output is of the expected 
        type and formats data as necessary before returning the result. 
        """

        raise NotImplementedError()


T = TypeVar('T', bound="Propagator")
class Propagator(Emitter[It, Ot]):
    """
    Emitters for basic constructs.

    This class contains abstract methods. 
    """

    def __copy__(self: T) -> T:
        """
        Make a copy of self.
        
        Not implemented by default.

        For cases where a propagator instance is used as a template. Should 
        ensure that copies of the template may be mutated without unwanted 
        side-effects.
        """
        raise NotImplementedError() 

    def __call__(
        self, construct: Symbol, inputs: PullFuncs[It], **kwds: Any
    ) -> Ot:
        """
        Execute construct's forward propagation cycle.

        Pulls data from inputs constructs, delegates processing to self.call(),
        and passes result to self.emit().
        """

        inputs_ = {source: pull_func() for source, pull_func in inputs.items()}
        intermediate: Any = self.call(construct, inputs_, **kwds)
        
        return self.emit(intermediate)

    @abstractmethod
    def call(self, construct: Symbol, inputs: Inputs[It], **kwds: Any) -> Any:
        """
        Execute construct's forward propagation cycle.

        Abstract method.

        :param construct: Name of the client construct. 
        :param inputs: Pairs the names of input constructs with their outputs. 
        :param kwds: Optional parameters. Propagator instances are recommended 
            to throw errors upon receipt of unexpected keywords.
        """
        raise NotImplementedError()


class Cycle(Emitter[It, Ot]):
    """Represents a container construct activation cycle."""

    # Specifies data required to construct the output packet
    output: ConstructType = ConstructType.null_construct

    def __init__(self, sequence, matches: MatchSet = None):

        super().__init__(matches=matches)
        self.sequence = sequence
    

# Decorator @no_type_check is meant to disable type_checking for the class (but 
# not sub- or superclasses). @no_type_check is not supported on mypy as of 
# 2020-06-10. Disabling type checks is required here to prevent the typechecker 
# from complaining about dynamically set attributes. 'type: ignore' is set to 
# prevent mypy from complaining until the issue is resolved.
# - Can
@no_type_check
class Assets(SimpleNamespace): # type: ignore
    """
    Dynamic namespace for construct assets.
    
    Provides handles for various datastructures such as chunk databases, rule 
    databases, bla information, etc. In general, all resources shared among 
    different components of a container construct are considered assets. 
    """
    pass
