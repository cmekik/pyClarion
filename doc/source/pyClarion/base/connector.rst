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
.. automethod:: Connector.pull
.. automethod:: Connector.add_link

.. autoclass:: Propagator
    :members:
.. automethod:: Propagator.__call__
.. automethod:: Propagator.get_pull_method
.. automethod:: Propagator.propagate

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
