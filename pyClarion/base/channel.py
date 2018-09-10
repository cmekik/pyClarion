'''
Tools for modeling activation flows.

Provides the abstract ``Channel`` class, which defines a callable object that 
receives a single ``ActivationPacket`` as input and outputs a single 
``ActivationPacket`` in response.

Usage
=====

``Channel`` objects may be used to capture activation flows in many ways, at 
multiple levels of granularity. The role of the ``Channel`` class is to provide 
implementations of various activation flows with a uniform interface while 
leaving authors free to determine as many implementation details as possible.

Instantiation
-------------

The ``Channel`` class is an abstract class. 

>>> Channel()
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class Channel with abstract methods __call__

To define a concrete ``Channel``, one must provide a ``__call__`` method with 
an appropriate signature. 

The ``Channel`` class is implemented as a generic class, taking one type 
variable, ``Pt``, which determines the input type of the ``__call__`` method. 
This variable is bounded above by the type ``ActivationPacket``. In other words, 
it is expected that arguments to ``Channel.__call__`` are subclasses of 
``ActivationPacket``. These type annotations are useful for capturing 
assumptions about activations (for example, about default activation values).

>>> class MyPacket(ActivationPacket):
...     def default_activation(self, key):
...         return 0.0
... 
>>> class MyChannel(Channel[MyPacket]):
...     """An activation channel that outputs its input as is."""
... 
...     def __call__(self, input_map : MyPacket) -> MyPacket:
... 
...         return input_map 
... 
>>> MyChannel()
<pyClarion.base.channel.MyChannel object at ...>

Use Cases
---------

The simplest use case for a ``Channel`` object is to transform activation 
strengths.

>>> from pyClarion.base.node import Node
>>> class ScaledPacket(MyPacket):
...     pass
... 
>>> class MultiplierChannel(Channel[MyPacket]):
... 
...     def __init__(self, multiplier):
...         self.multiplier = multiplier
... 
...     def __call__(self, input_map : MyPacket) -> ScaledPacket:
...         output = ScaledPacket()
...         for n in input_map:
...             output[n] = input_map[n] * self.multiplier
...         return output
... 
>>> n1, n2 = Node(), Node()
>>> input_activations = MyPacket({n1 : 0.2, n2 : 0.4})
>>> channel = MultiplierChannel(2)
>>> output = channel(input_activations) 
>>> output == MyPacket({n1 : 0.4, n2 : 0.8})
True
>>> isinstance(output, ScaledPacket)
True

A much more interesting use case of ``Channel`` is implementing knowledge-based
processing of the input activation packet.

>>> from pyClarion.base.node import Microfeature, Chunk
>>> class MyTopLevelPacket(MyPacket):
...     pass
... 
>>> from typing import Dict
>>> class MyAssociativeNetwork(Channel[MyPacket]):
... 
...     def __init__(
...         self, assoc : Dict[Node, Dict[Node, float]]) -> None:
...         """Initialize an associative network.
... 
...         :param assoc: A dict that maps conclusion chunks to 
...         dicts mapping their condition chunks to their respective weights.
...         """
...         self.assoc = assoc
... 
...     def __call__(self, input_map : MyPacket) -> MyTopLevelPacket:
...         output = MyTopLevelPacket()
...         for conclusion_chunk in self.assoc:
...             output[conclusion_chunk] = 0.0
...             for condition_chunk in self.assoc[conclusion_chunk]:
...                 output[conclusion_chunk] += (
...                         input_map[condition_chunk] *
...                         self.assoc[conclusion_chunk][condition_chunk]
...                 )
...         return output
>>> ch1, ch2, ch3 = Chunk('BLOOD ORANGE'), Chunk('RED'), Chunk('YELLOW')
>>> # Throw in some Microfeatures to see if MyAssociativeNetwork gets confused
>>> mf1, mf2 = Microfeature('color', 'red'), Microfeature('color', 'orange')
>>> assoc = {
...     ch2 : {ch1 : 0.4},
...     ch3 : {ch1 : 0.2}    
... }
>>> channel = MyAssociativeNetwork(assoc)
>>> input_map = MyPacket({ch1 : 1.0, mf1: 0.3, mf2 : 0.2})
>>> channel(input_map) == {ch2 : 0.4, ch3 : 0.2}
True

``Channel`` instances can be used to wrap efficient implementations of various 
constructs:

>>> class AmazingDQN(object):
...     def __call__(self, input_map):
...         """This should be truly amazing; but it's just a placeholder."""
...
...         output = dict()
...         for k, v in input_map.items():
...              if isinstance(k, Microfeature):
...                  output[k] = v
...         return output
... 
>>> class MyAmazingDQNPacket(MyPacket):
...     pass
... 
...
>>> class MyAmazingDQNChannel(Channel[MyPacket]):
...     """A wrapper for an AmazingDQN implementation."""
...
...     def __init__(self, amazing_dqn):
...         self.amazing_dqn = amazing_dqn
... 
...     def __call__(self, input_map : MyPacket) -> MyAmazingDQNPacket:
...         output = self.amazing_dqn(input_map)
...         return MyAmazingDQNPacket(output)
... 
...     def call_amazing_dqn(self, input_map):
...         return self.amazing_dqn.__call__(input_map)
... 
>>> amazing_dqn = AmazingDQN()
>>> channel = MyAmazingDQNChannel(amazing_dqn)
>>> channel(input_map) == {mf1 : 0.3, mf2 : 0.2}
True

Granularity
-----------

There is no a priori restriction on the granularity of a ``Channel`` object. The 
examples above are rather coarse, often capturing and handling all of the 
knowledge of some component (e.g., an entire associative network). Such an 
architecture may be well-suited for certain use cases, but others may call for 
a more fine-grained implementation such as the one below:

>>> class FineGrainedTopDownChannel(Channel):
...     """Represents a top-down link between two individual nodes."""
... 
...     def __init__(self, chunk, microfeature, weight):
...         self.chunk = chunk
...         self.microfeature = microfeature
...         self.weight = weight
...
...     def __call__(self, input_map):
...         output = MyTopDownPacket({
...             self.microfeature : input_map[self.chunk] * self.weight
...         })
...         return output
... 

The ``FineGrainedTopDown`` class above defines a top-down link between a single 
chunk and microfeature. Such a fine-grained channel implementation may be useful 
for, e.g., constructing a massively parallel Clarion agent architecture.
'''


import abc
from typing import Generic, TypeVar
from pyClarion.base.packet import ActivationPacket


###############
# ABSTRACTION #
###############

Pt = TypeVar('Pt', bound=ActivationPacket)


class Channel(Generic[Pt], abc.ABC):
    """An abstract generic class for capturing activation flows.

    This class that provides an interface for handling basic activation flows. 
    Activation flows are implemented in the ``__call__`` method. 

    See module documentation for examples and details.
    """
    
    @abc.abstractmethod
    def __call__(self, input_map: Pt) -> ActivationPacket:
        """Compute and return activations resulting from an input to this 
        channel.

        .. note::
            Assumptions about missing expected nodes in the input map should 
            be explicitly specified/documented, along with behavior for handling 
            such cases. 

        :param input_map: An activation packet representing the input to this 
            channel.
        """

        pass
