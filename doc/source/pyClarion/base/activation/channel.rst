pyClarion.base.activation.channel
=================================

.. automodule:: pyClarion.base.activation.channel

Module Reference
----------------

.. currentmodule:: pyClarion.base.activation.channel

Abstraction
~~~~~~~~~~~

.. autoclass:: Channel
.. automethod:: Channel.__call__

Base Classes
~~~~~~~~~~~~

.. autoclass:: TopDown
.. autoclass:: BottomUp
.. autoclass:: TopLevel
.. autoclass:: BottomLevel

Type Aliases
~~~~~~~~~~~~

.. data:: ChannelToTop

Type alias for representing channels that output to top-level nodes (i.e., 
`TopLevel` and `BottomUp` channels).

.. data:: ChannelToBottom

Type alias for representing channels that output to bottom-level nodes (i.e., 
`BottomLevel` and `TopDown` channels).
