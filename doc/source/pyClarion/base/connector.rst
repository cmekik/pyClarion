pyClarion.base.connector
========================

.. automodule:: pyClarion.base.connector

Module Reference
----------------

.. currentmodule:: pyClarion.base.connector

Abstraction
~~~~~~~~~~~

.. autoclass:: Connector
   :members: client, buffer, junction, listeners
.. automethod:: Connector.__init__
.. automethod:: Connector.__call__
.. automethod:: Connector.register
.. automethod:: Connector.update
.. automethod:: Connector.clear
.. automethod:: Connector.propagate
.. automethod:: Connector.notify_listeners
.. automethod:: Connector.clear_listeners

NodeConnector
~~~~~~~~~~~~~

.. autoclass:: NodeConnector
.. automethod:: NodeConnector.propagate

ChannelConnector
~~~~~~~~~~~~~~~~

.. autoclass:: ChannelConnector
.. automethod:: ChannelConnector.propagate
