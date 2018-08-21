'''
This module provides tools for handling activations of individual nodes.

Usage
=====

The main export of this module is the ``ActivationHandler`` class, which defines 
how an individual node reacts to incoming activations.

An ``ActivationHandler`` instance associates a node with two channels and a 
junction. One channel defines how the node reacts to inputs from the top-level, 
and the other defines how the node reacts to inputs from the bottom level.

``ActivationHandler`` instances  are callable: they compute the activation of  
assigned nodes with respect to given input.

Demo
----

>>> from pyClarion.base.node import Microfeature, Chunk
>>> from pyClarion.base.activation.packet import (
...     ActivationPacket, TopLevelPacket, BottomUpPacket
... )
... 
>>> from pyClarion.base.activation.channel import (
...     Channel, TopLevel, BottomUp    
... )
>>> from pyClarion.base.activation.junction import GenericMaxJunction
>>> class MyPacket(ActivationPacket):
...     def default_activation(self, key):
...         return 0.0
... 
>>> class MyTopLevelPacket(MyPacket, TopLevelPacket):
...     pass
... 
>>> class MyBottomUpPacket(MyPacket, BottomUpPacket):
...     pass
... 
>>> class MyChannel(Channel):
...     """An activation channel that outputs an empty MyPacket instance."""
... 
...     def __init__(self, node : Node, weights : dict) -> None:
...         self.node = node
...         self.weights = weights
... 
...     def __call__(self, input_map : ActivationPacket) -> ActivationPacket:
...         output = MyPacket()
...         for chunk in self.weights:
...             output[self.node] += self.weights[chunk] * input_map[chunk]
...         output[self.node] = min(output[self.node], 1.0)
...         return output 
... 
>>> class MyTopLevel(MyChannel, TopLevel):
...     def __call__(self, input_map : ActivationPacket) -> MyTopLevelPacket:
...         return MyTopLevelPacket(super().__call__(input_map))
...
>>> class MyBottomUp(MyChannel, BottomUp):
...     def __call__(self, input_map : ActivationPacket) -> MyBottomUpPacket:
...         return MyBottomUpPacket(super().__call__(input_map))
... 
>>> class MyMaxJunction(GenericMaxJunction):
... 
...     @property
...     def output_type(self):
...         return MyPacket
...
>>> ch_1 = Chunk('COLOR')
>>> ch_2 = Chunk('RED')
>>> ch_3 = Chunk('GREEN')
>>> mf_1 = Microfeature('R', 233)
>>> mf_2 = Microfeature('G', 118)
>>> mf_3 = Microfeature('B', 6)
>>> top_channel = MyTopLevel(ch_1, {ch_2 : 1.0, ch_3 : 1.0})
>>> bottom_channel = MyBottomUp(ch_1, {mf_1 : 1.0, mf_2 : 1.0, mf_3 : 1.0}) 
>>> junction = MyMaxJunction()
>>> ch_handler = ActivationHandler(
...     ch_1, top_channel, bottom_channel, junction
... )
...
>>> p_0 = MyPacket()
>>> p_1 = MyPacket({ch_2 : .2, mf_1 : .6 })
>>> p_2 = MyPacket({ch_2 : .2, ch_3 : .9, mf_1 : .6}) 
>>> ch_handler(p_0) == MyPacket({ch_1 : .0})
True
>>> ch_handler(p_1) == MyPacket({ch_1 : .6})
True
>>> ch_handler(p_2) == MyPacket({ch_1 : 1.})
True
'''


from pyClarion.base.node import Node, Microfeature, Chunk, NodeSet
from pyClarion.base.activation.packet import ActivationPacket
from pyClarion.base.activation.channel import (
    TopChannel, BottomChannel, TopLevel, BottomLevel, TopDown, BottomUp
)
from pyClarion.base.activation.junction import Junction


class ActivationHandler(object):
    """Handles the activation of an individual Clarion node."""
    
    def __init__(
        self, 
        node : Node, 
        top_channel : TopChannel, 
        bottom_channel : BottomChannel,
        junction : Junction
    ) -> None:
        '''Initialize a new ActivationHandler instance.

        .. warning::
           The types of ``node``, ``top_channel``, and ``bottom_channel`` must 
           be compatible. The rule is as follows:

           - If ``isinstance(node, Chunk)`` then it must be the case that 
             ``isinstance(top_channel, channel.TopLevel)`` and 
             ``isinstance(bottom_channel, channel.BottomUp)``
           - If ``isinstance(node, Chunk)`` then it must be the case that 
             ``isinstance(top_channel, channel.TopDown)`` and 
             ``isinstance(bottom_channel, channel.BottomLevel)``

           If these conditions are not satisfied, ``ActivationHandler`` will not 
           complain. But, it may do so in the future.
        
        :param node: The node assgined to this activation handler.
        :param top_channel: A channel defining how the node reacts to incoming 
          top-level activations.
        :param bottom_channel: A channel defining how the node reacts to 
          incoming botton-level activations.
        :param junction: A junction defining how top- and bottom-level 
          activations are combined to produce the node's output. 
        '''

        self._node = node
        self._top_channel = top_channel
        self._bottom_channel = bottom_channel
        self._junction = junction

    def __call__(self, input_map : ActivationPacket) -> ActivationPacket:
        '''Return current activation of node represented by `self`.

        It is expected that the return value of this function will be an 
        ``ActivationPacket`` instance with ``self.node`` as its only key. If 
        this is not the case, ``ActivationHandler`` will not complain, but may 
        do so in the future.

        :param input_map: An activation packet representing the input to 
            `self.node`.
        '''

        activation_from_top_level = self.top_channel(input_map)
        activation_from_bottom_level = self.bottom_channel(input_map)
        output = self.junction(
            activation_from_top_level, activation_from_bottom_level
        )
        
        return output

    @property
    def node(self) -> Node:
        '''The node whose activations `self` propagates.'''
        return self._node

    @property
    def top_channel(self) -> TopChannel:
        '''Defines how top-level nodes activate `self`.'''
        return self._top_channel

    @property
    def bottom_channel(self) -> BottomChannel:
        '''Defines how bottom-level nodes activate `self`.'''
        return self._bottom_channel

    @property
    def junction(self) -> Junction:
        '''Defines how `self` combines top and bottom level activations.'''
        return self._junction


if __name__ == '__main__':
    import doctest
    doctest.testmod()