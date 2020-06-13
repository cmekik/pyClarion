__all__ = ["ChunkAdder"]


from pyClarion.base.realizers import Node
from pyClarion.components import Chunks


class ChunkAdder(object):

    def __init__(self, config, monitor, subsystem=None):

        self.config = config
        self.monitor = monitor
        self.subsystem = subsystem

    def __call__(self, realizer):

        db: Chunks = realizer.assets.chunks # this should be a `Chunks` object.
        subsystem = (
            realizer[self.subsystem] if self.subsystem is not None 
            else realizer
        )

        state = subsystem.output.decisions[self.monitor]
        added = set()
        for ch, form in state.items():
            chunks = db.find_form(form)
            if len(chunks) == 0:
                db.set_chunk(ch, form)
                # Should be able to control target subsystems, not just add to 
                # the source subsystem - Can
                subsystem.add(
                    Node(
                        name=ch,
                        matches=self.config.matches,
                        propagator=self.config.propagator,
                        updaters=self.config.updaters
                    )
                )
                added.add(ch)
            elif len(chunks) == 1:
                pass
            else:
                raise ValueError("Corrupt chunk database.")

        return added

    class Config(object):

        # The args should really be factories. This is unsafe. Parameter 
        # changes here or to any generated element will cause updates to all 
        # generated elements. Each input should be copyable and superordinate 
        # object should work with copies. Alternatively, argument structure may 
        # be changed so that constructors are called by the superordinate 
        # object. - Can
        def __init__(self, matches=None, propagator=None, updaters=None):

            self.matches=matches
            self.propagator=propagator
            self.updaters=updaters