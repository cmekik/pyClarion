pyClarion.base.activation.handler
=================================

.. automodule:: pyClarion.base.activation.handler

Module Reference
----------------

.. currentmodule:: pyClarion.base.activation.handler

Abstraction
~~~~~~~~~~~

.. autoclass:: ActivationHandler
   :members: client, buffer, junction, listeners
.. automethod:: ActivationHandler.__init__
.. automethod:: ActivationHandler.__call__
.. automethod:: ActivationHandler.register
.. automethod:: ActivationHandler.update
.. automethod:: ActivationHandler.propagate
.. automethod:: ActivationHandler.notify_listeners

Node Handler
~~~~~~~~~~~~

.. autoclass:: NodeHandler
.. automethod:: NodeHandler.propagate

Channel Handler
~~~~~~~~~~~~~~~

.. autoclass:: ChannelHandler
.. automethod:: ChannelHandler.propagate
