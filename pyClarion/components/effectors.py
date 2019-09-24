from pyClarion.base.packets import DecisionPacket

class MappingEffector(object):
    """Links actionable chunks to callbacks via a direct mapping."""

    def __init__(self, callbacks = None) -> None:
        """
        Initialize a SimpleEffector instance.

        :param chunk2callback: Mapping from actionable chunks to callbacks.
        """

        self.callbacks = callbacks if callbacks is not None else dict()

    def __call__(self, dpacket: DecisionPacket) -> None:
        """
        Execute callbacks associated with each chosen chunk.

        :param dpacket: A decision packet.
        """
        
        for chunk in dpacket.selection:
            self.callbacks[chunk].__call__()

    def set_action(self, chunk_, callback):

        self.callbacks[chunk_] = callback
