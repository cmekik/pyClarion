'''
This module provides constructs for modeling activation flows in pyClarion.

Usage
=====

This module exports the ``Channel`` class. A pyClarion ``Channel`` is a callable 
object that receives a single ``BaseActivationPacket`` instance as input and 
outputs a single ``BaseActivationPacket`` instance in response.

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

>>> class MyPacket(BaseActivationPacket):
...     def default_activation(self, key):
...         return 0.0
... 
>>> class MyChannel(Channel):
...     """An activation channel that outputs an empty MyPacket instance."""
...     def __call__(self, input_map : BaseActivationPacket) -> MyPacket:
...         return MyPacket() 
... 
>>> MyChannel()
<__main__.MyChannel object at ...>

Channel Sub-Types
-----------------

``Channel`` subclasses are expected to extend input types and restrict output 
types, in keeping with the Liskov principle. In practice, the input type is 
generally expected to be left unchanged, and the output type is expected be 
restricted so as to reflect the type of processing that occurred.

>>> class MyTopDownPacket(MyPacket):
...     pass
... 
>>> class MyTopDownChannel(Channel):
...     """A top-down channel that outputs an empty MyTopDownPacket instance."""
...     def __call__(self, input_map : BaseActivationPacket) -> MyTopDownPacket:
...         return MyTopDownPacket() 
... 

Use Cases
---------

The simplest use case for a ``Channel`` object is to statically transform 
activation packets.

>>> from pyClarion.base.node import Node
>>> class ScaledPacket(MyPacket):
...     pass
... 
>>> class MultiplierChannel(Channel):
... 
...     def __init__(self, multiplier):
...         self.multiplier = multiplier
... 
...     def __call__(self, input_map : BaseActivationPacket) -> ScaledPacket:
...         output = ScaledPacket()
...         for n in input_map:
...             output[n] = input_map[n] * self.multiplier
...         return output
... 
>>> n1, n2 = Node(), Node()
>>> input_activations = MyPacket({n1 : 0.2, n2 : 0.4})
>>> channel = MultiplierChannel(2)
>>> output =channel(input_activations) 
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
>>> class MyAssociativeNetwork(Channel):
... 
...     def __init__(
...         self, assoc : T.Dict[Node, T.Dict[Node, float]]) -> None:
...         """Initialize an associative network.
... 
...         :param assoc: A dict that maps conclusion chunks to 
...         dicts mapping their condition chunks to their respective weights.
...         """
...         self.assoc = assoc
... 
...     def __call__(
...     self, input_map : BaseActivationPacket
... ) -> MyTopLevelPacket:
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
>>> class MyAmazingDQNChannel(Channel):
...     """A wrapper for an AmazingDQN implementation."""
...
...     def __init__(self, amazing_dqn):
...         self.amazing_dqn = amazing_dqn
... 
...     def __call__(
...     self, input_map : BaseActivationPacket
... ) -> MyAmazingDQNPacket:
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
for constructing a massively parallel Clarion agent architecture. Below is just 
a sketch of how such a system may be set up.

>>> class ActivationTracker(object):
...     """Tracks the activation associated with an individual node or edge."""
...     
...     def __init__(self, callback):
...         self.callback = callback
...         self.listeners = set()
...         self.buffer = MyPacket()
...
...     def update(self, packet):
...         self.buffer.update(packet)
...     
...     def subscribe(self, tracker):
...         tracker.register(self) 
...     
...     def register(self, listener):
...         self.listeners.add(listener)
... 
...     def notify_listeners(self):
...         output = self.callback(self.buffer)
...         for listener in self.listeners:
...             listener.update(output)
...     
...     def trigger_condition(self):
...         """This is just a place holder for an actual condition."""
...         return True
... 
...     def step(self):
...         if self.trigger_condition():
...             self.notify_listeners()
... 

This simple ``ActivationTracker`` will enable concurrent activation flows. Here 
is an example of a top-down activation flow about apples with 
``ActivationTracker`` instances driving activation flow.

>>> ch = Chunk("APPLE")
>>> mf1 = Microfeature("color", "red")
>>> mf2 = Microfeature("tasty", True)
>>> edge1 = FineGrainedTopDownChannel(ch, mf1, 1.0)
>>> edge2 = FineGrainedTopDownChannel(ch, mf2, 1.0)
>>> trackers = {
...     # Trackers associated with nodes should simply pass on their activation.
...     ch : ActivationTracker(lambda x: x),
...     mf1 : ActivationTracker(lambda x: x),
...     mf2 : ActivationTracker(lambda x: x),
...     edge1 : ActivationTracker(edge1),
...     edge2 : ActivationTracker(edge2)
... }
... 
>>> # We still need to connect everything up.
>>> trackers[mf1].subscribe(trackers[edge1])
>>> trackers[mf2].subscribe(trackers[edge2])
>>> trackers[edge1].subscribe(trackers[ch])
>>> trackers[edge2].subscribe(trackers[ch])
>>> # Set initial chunk activation.
>>> trackers[ch].update(MyPacket({ch : 1.0}))
>>> # Propagate activations
>>> for tracker in trackers.values():
...     tracker.step()
... 
>>> # Check that propagation worked
>>> trackers[mf1].buffer == MyPacket({mf1 : 1.0})
True
>>> trackers[mf2].buffer == MyPacket({mf2 : 1.0})
True

In the example above, the concept APPLE is activated, which leads to the 
activation, in the bottom level, of microfeatures corresponding to the color 
red and tastiness
'''


import abc
import typing as T
from pyClarion.base.activation.packet import BaseActivationPacket


###############
# ABSTRACTION #
###############


class Channel(abc.ABC):
    """An abstract class for capturing activation flows.

    This is a callable class that provides an interface for handling basic 
    activation flows. Outputs are allowed to be empty when such behavior is 
    sensible.
    
    It is assumed that an activation channel will pay attention only to the 
    activations relevant to the computation it implements. For instance, if an 
    activation class implementing a bottom-up connection is passed a bunch of 
    chunk activations, it should simply ignore these and look for matching
    microfeatures. 
    
    If an activation channel is handed an input that does not contain a complete 
    activation dictionary for expected nodes, it should not fail. Instead, it 
    should have a well-defined default behavior for such cases. 

    See module documentation for examples and details.
    """
    
    @abc.abstractmethod
    def __call__(
        self, input_map : BaseActivationPacket
    ) -> BaseActivationPacket:
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


if __name__=='__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)