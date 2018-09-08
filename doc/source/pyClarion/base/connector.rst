pyClarion.base.connector
========================

.. automodule:: pyClarion.base.connector

Module Reference
----------------

.. currentmodule:: pyClarion.base.connector

Abstractions
~~~~~~~~~~~~

.. autoclass:: Connector
   :members:
.. automethod:: Connector.__init__
.. automethod:: Connector.__call__
.. automethod:: Connector.update
.. automethod:: Connector.clear
.. automethod:: Connector.get_reporter


.. autoclass:: Propagator
    :members:
.. automethod:: Propagator.__call__
.. automethod:: Propagator.register
.. automethod:: Propagator.propagate
.. automethod:: Propagator.notify_listeners
.. automethod:: Propagator.clear_listeners

NodeConnector
~~~~~~~~~~~~~

.. autoclass:: NodeConnector
.. automethod:: NodeConnector.propagate

ChannelConnector
~~~~~~~~~~~~~~~~

.. autoclass:: ChannelConnector
.. automethod:: ChannelConnector.propagate

SelectorConnector
~~~~~~~~~~~~~~~~~

.. autoclass:: SelectorConnector
.. automethod:: SelectorConnector.propagate

EffectorConnector
~~~~~~~~~~~~~~~~~

.. autoclass:: EffectorConnector
.. automethod:: EffectorConnector.__call__