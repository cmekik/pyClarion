"""Basic definitions for constructing components."""


__all__ = ["Emitter", "Propagator", "Cycle", "Assets", "FeatureInterface"]


from pyClarion.base.symbols import ConstructType, Symbol, feature
from pyClarion.utils.funcs import group_by_dims
from abc import abstractmethod
from types import SimpleNamespace
from typing import (
    TypeVar, Union, Tuple, Dict, Callable, Hashable, Generic, Any, Optional, 
    Text, Iterator, Iterable, Mapping, ClassVar, List, FrozenSet, cast, 
    no_type_check
)


Dt = TypeVar('Dt') # type variable for inputs to emitters
Inputs = Mapping[Symbol, Dt]

It = TypeVar('It', contravariant=True) # type variable for emitter inputs
Ot = TypeVar('Ot', covariant=True) # type variable for emitter outputs

Xt = TypeVar("Xt")
class Emitter(Generic[Xt, Ot]):
    """
    Base class for propagating strengths, decisions, etc.

    Emitters define how constructs connect, process inputs, and set outputs.
    """

    @abstractmethod
    def expects(self, construct: Symbol):
        """Return True iff self expects input from construct."""

        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def emit(data: Xt = None) -> Ot:
        """
        Emit output.

        If no data is passed in, emits a default or null value of the expected
        output type. Otherwise, ensures output is of the expected type and 
        (preferably) immutable before returning the result. 
        """

        raise NotImplementedError()


T = TypeVar('T', bound="Propagator")
class Propagator(Emitter[Xt, Ot], Generic[It, Xt, Ot]):
    """Emitter for basic constructs."""

    def __copy__(self: T) -> T:
        """
        Make a copy of self.

        Enables use of propagator instances as templates. Should ensure that 
        mutation of copies do not have unwanted side-effects.
        """
        raise NotImplementedError() 

    def __call__(self, construct: Symbol, inputs: Inputs[It]) -> Ot:
        """
        Execute construct's forward propagation cycle.

        Pulls expected data from inputs constructs, delegates processing to 
        self.call(), and passes result to self.emit().
        """

        return self.emit(self.call(construct, inputs))

    @abstractmethod
    def call(self, construct: Symbol, inputs: Inputs[It]) -> Xt:
        """
        Compute construct's output.

        :param construct: Name of the client construct. 
        :param inputs: Pairs the names of input constructs with their outputs. 
        :param kwds: Optional parameters. Propagator instances are recommended 
            to throw errors upon receipt of unexpected keywords.
        """

        raise NotImplementedError()


class Cycle(Emitter[Xt, Ot]):
    """Emitter for composite constructs."""

    # Specifies data required to construct the output packet
    output: ClassVar[ConstructType] = ConstructType.null_construct
    sequence: Iterable[ConstructType]
    

# @no_type_check disables type_checking for Assets (but not subclasses). 
# Included b/c dynamic usage of Assets causes mypy to complain.
# @no_type_check is not supported on mypy as of 2020-06-10. 'type: ignore' will 
# do for now. - Can
@no_type_check
class Assets(SimpleNamespace): # type: ignore
    """
    Dynamic namespace for construct assets.
    
    Provides handles for various datastructures such as chunk databases, rule 
    databases, bla information, etc. In general, all resources shared among 
    different components of a container construct are considered assets. 
    """
    pass

class FeatureInterface(object):
    """
    Control interface for a component.
    
    Defines control features and default values. Provides parsing utilities.
    Each defined feature dimension is interpreted as defining a specific set of 
    alternate actions. A default value must be defined for each dimension, 
    representing the 'do nothing' action.
    """

    _features: FrozenSet[feature]
    _defaults: FrozenSet[feature]
    _tags: FrozenSet[Hashable]
    _dims: FrozenSet[Tuple[Hashable, int]]

    def __post_init__(self):

        self._validate_data()
        self._set_interface_properties()
        self._validate_interface_properties()

    @property
    def features(self):
        """The set of features defined by self."""
        
        return self._features

    @property
    def defaults(self):
        """Feature, defined by self, indicating default values, if any."""
        
        return self._defaults

    @property
    def tags(self):
        """The set of dimensional labels defined by self."""
        
        return self._tags

    @property
    def dims(self):
        """The set of feature dimensions (w/ lags) defined by self."""
        
        return self._dims

    def parse_commands(self, data):
        """
        Determine the value associated with each control dimension.
        
        :param data: A set of features.
        """

        _cmds = set(f for f in data if f in self.features)

        cmds, groups = {}, group_by_dims(features=_cmds)
        for k, g in groups.items():
            if len(g) > 1:
                msg = "Received multiple commands for dim '{}'."
                raise ValueError(msg.format(k))
            cmds[k] = g[0].val
        
        for f in self.defaults:
            if f.dim not in cmds:
                cmds[f.dim] = f.val

        return cmds

    def _validate_data(self):

        raise NotImplementedError()

    def _set_interface_properties(self):

        raise NotImplementedError()

    def _validate_interface_properties(self):

        _features_dims = set(f.dim for f in self.features)
        _features_tags = set(f.tag for f in self.features)
        _defaults_dims = set(f.dim for f in self.defaults)

        # TODO: Use a more suitable exception class. - Can

        if self.tags != _features_tags:
            raise ValueError("self.tag conflicts with self.features.")
        if self.dims != _features_dims:
            raise ValueError("self.dims conflicts with self.features.")
        if not self.defaults.issubset(self.features):
            raise ValueError("self.defaults not a subset of self.features.")
        if not self.dims.issubset(_defaults_dims):
            raise ValueError("self.defaults conflicts with self.dims.")
        if len(self.dims) != len(_defaults_dims):
            raise ValueError("multiple defaults assigned to a single dim.")
