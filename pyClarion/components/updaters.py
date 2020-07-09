__all__ = ["ChunkAdder"]


from pyClarion.base.realizers import Node
from pyClarion.components import Chunks
from copy import copy


class ChunkAdder(object):
    """
    Adds new chunk nodes to client constructs.
    
    Constructs Node objects for new chunks from a given template and adds them 
    to client realizers.

    Does not allow adding updaters.

    Warning: This implementation relies on (shallow) copying. If propagators 
        have mutable attributes unexpected behavior may occur. To mitigate 
        this, propagators must define appropriate `__copy__()` methods.
    """

    def __init__(
        self, 
        propagator, 
        response, 
        subsystem=None, 
        clients=None, 
        return_added=False
    ):
        """
        Initialize a new `ChunkAdder` instance.
        
        :param template: A ChunkAdder.Template object defining the form of 
            `Node` instances representing new chunks.
        :param response: Construct symbol for a response construct emmiting new 
            chunk recommendations. 
        :param subsystem: The subsystem that should be monitored. Used only if 
            the chunk adder is located at the `Agent` level.
        :param clients: Subsystem(s) to which new chunk nodes should be added. 
            If None, it will be assumed that the sole client is the realizer 
            housing this updater.
        :param return_added: Whether to return the set of added chunks for 
            possible subsequent processing.
        """

        self.propagator = propagator
        self.response = response
        self.subsystem = subsystem
        self.clients = {subsystem} if clients is None else clients
        self.return_added = return_added

    def __call__(self, realizer):

        db: Chunks = realizer.assets.chunks # this should be a `Chunks` object.
        subsystem = (
            realizer[self.subsystem] if self.subsystem is not None 
            else realizer
        )

        state = subsystem.output.decisions[self.response]
        added = set()
        for ch, form in state.items():
            chunks = db.find_form(form)
            if len(chunks) == 0:
                db.set_chunk(ch, form)
                added.add(ch)
            elif len(chunks) == 1:
                pass
            else:
                raise ValueError("Corrupt chunk database.")

        clients = self.clients if self.clients is not None else (None,)
        for construct in clients:
            client = realizer[construct] if construct is not None else realizer
            for ch in added: 
                client.add(
                    # This is possibly problematic. May need to develop a way 
                    # to easily and correctly define construct realizer 
                    # factories. Main risk here are the updaters; since updaters 
                    # are, at this time rather unrestricted. - Can 
                    Node(
                        name=ch,
                        propagator=copy(self.propagator)
                    )
                )

        return None if not self.return_added else added
