"""Provides abstract classes for defining forward propagation cycles."""


__all__ = ["Propagator", "Cycle", "Assets"]


from pyClarion.base.symbols import ConstructSymbol, MatchSpec, ConstructType
from pyClarion.base.packets import (
    ActivationPacket, ResponsePacket, SubsystemPacket
)
from pyClarion.utils.funcs import simple_junction
from types import MappingProxyType
from typing import (
    TypeVar, Generic, Mapping, Callable, Any, Mapping, Tuple, Set, Dict, 
    Sequence, Optional, no_type_check
)
from types import SimpleNamespace
from abc import abstractmethod


Dt = TypeVar('Dt') # type variable for inputs to processes
PullFuncs = Mapping[ConstructSymbol, Callable[[], Dt]]
Inputs = Mapping[ConstructSymbol, Dt]


It = TypeVar('It', contravariant=True) # type variable for propagator inputs
Ot = TypeVar('Ot', covariant=True) # type variable for propagator outputs
T = TypeVar('T', bound="Propagator")
class Propagator(Generic[It, Ot]):
    """
    Abstract class for propagating strengths, decisions, and states.

    Propagator subclasses and instances define how constructs process inputs 
    and set outputs.
    """

    matches: MatchSpec

    def __init__(self, matches: MatchSpec = None):

        self.matches = matches if matches is not None else MatchSpec()

    def __copy__(self: T) -> T:
        """
        Make a copy of self.
        
        This method is primarily for use in factory patterns where a propagator 
        instance may be provided as a template. The copy method should ensure 
        that copies of the template instance may be mutated without unwanted 
        mutation of other instances derived from the template.
        """
        raise NotImplementedError() 

    def __call__(
        self, construct: ConstructSymbol, inputs: PullFuncs[It], **kwds: Any
    ) -> Ot:
        """
        Execute construct's forward propagation cycle.

        Wrapper for self.call().

        Pulls data from inputs constructs, delegates processing to self.call(),
        and wraps result in appropriate Packet instance.
        """

        inputs_ = {source: pull_func() for source, pull_func in inputs.items()}
        intermediate: Any = self.call(construct, inputs_, **kwds)
        
        return self.emit(intermediate)

    def expects(self, construct: ConstructSymbol):
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

    @abstractmethod
    def call(
        self, construct: ConstructSymbol, inputs: Inputs[It], **kwds: Any
    ) -> Any:
        """
        Execute construct's forward propagation cycle.

        :param construct: The construct symbol associated with the realizer 
            owning to the propagation callback. 
        :param inputs: A dictionary, mapping construct symbols to Packet 
            instances, specifying input passed to the owning construct by other 
            constructs within the simulation. 
        :param kwargs: Any additional parameters passed to the call to owning 
            construct's propagate() method through the `options` argument. These 
            arguments may be used to contextually modify propagator behavior, 
            pass in external inputs, etc. For ease of debugging and as a 
            precaution against subtle bugs (e.g., failing to set an option due 
            to misspelled keyword argument name) it is recommended that 
            Propagator instances throw errors upon receipt of unexpected 
            keyword arguments.
        """
        raise NotImplementedError()


class Cycle(Generic[It, Ot]):
    """Represents a container construct activation cycle."""

    # Specifies data required to construct the output packet
    output: ConstructType = ConstructType.null_construct

    def __init__(self, sequence, matches: MatchSpec = None):

        self.sequence = sequence
        self.matches = matches if matches is not None else MatchSpec()

    def expects(self, construct: ConstructSymbol) -> bool:
        """Returns True if propagator expects input from given construct."""

        return construct in self.matches

    def emit(self, data: Any = None) -> Ot:
        raise NotImplementedError()
    

# Decorator is meant to disable type_checking for the class (but not sub- or 
# superclasses). @no_type_check is not supported on mypy as of 2020-06-10.
# Disabling type checks is required here to prevent the typechecker from 
# complaining about dynamically set attributes. 
# 'type: ignore' is set to prevent mypy from complaining until the issue is 
# resolved.
# - Can
@no_type_check
class Assets(SimpleNamespace): # type: ignore
    """
    A namespace for ContainerConstruct assets.
    
    The main purpose of `Assets` objects is to provide handles for various
    datastructures such as chunk databases, rule databases, bla information, 
    etc. In general, all resources shared among different components of a 
    container construct are considered assets. 
    
    It is the user's responsibility to make sure shared resources are shared 
    and used as intended. 
    """
    pass
