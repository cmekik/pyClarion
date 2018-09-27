from typing import MutableMapping, Dict, Callable, Any, Iterator, Iterable, Type, Optional
from pyClarion.base.symbols import (
    BasicConstructSymbol, Node, Flow, Appraisal, Activity, Subsystem, may_contain, connection_allowed
)
from pyClarion.base.packets import ActivationPacket, DecisionPacket
from pyClarion.base.links import (
    Propagator, NodePropagator, FlowPropagator, AppraisalPropagator, 
    ActivityDispatcher
)
from pyClarion.base.realizers.abstract import CompositeConstructRealizer
from pyClarion.base.realizers.basic import (
    NodeRealizer, FlowRealizer, AppraisalRealizer, ActivityRealizer
)


#####################################
### COMPOSITE CONSTRUCT REALIZERS ###
#####################################


def get_link_factory(construct: BasicConstructSymbol) -> Type:

    constructor: Type
    if isinstance(construct, Node):
        constructor = NodePropagator
    elif isinstance(construct, Flow):
        constructor = FlowPropagator
    elif isinstance(construct, Appraisal):
        constructor = AppraisalPropagator
    elif isinstance(construct, Activity):
        constructor = ActivityDispatcher
    else:
        raise TypeError()
    return constructor


class SubsystemRealizer(
    Propagator[ActivationPacket, DecisionPacket], 
    CompositeConstructRealizer[Subsystem]
):
    """A network of interconnected nodes and flows linked to an actuator."""

    def __init__(self, construct: Subsystem) -> None:

        CompositeConstructRealizer.__init__(self, construct)

        self.dict: Dict = dict()
        self._appraisal: Optional[Appraisal] = None
        self._activity: Optional[Activity] = None
        
        Propagator.__init__(self)

    def __len__(self) -> int:

        return len(self.dict)

    def __contains__(self, obj: Any) -> bool:

        return obj in self.dict

    def __iter__(self) -> Iterator:

        return self.dict.__iter__()

    def __getitem__(self, key: Any) -> Any:

        return self.dict[key]

    def __setitem__(self, key: Any, value: Any) -> None:

        if not key == value.construct:
            raise ValueError("Mismatch between key and realizer construct.")

        if may_contain(self.construct, key):

            if isinstance(key, Appraisal):
                if self._appraisal:
                    raise Exception("Appraisal already set")
                else:
                    self._appraisal = key

            elif isinstance(key, Activity):
                if self._activity:
                    raise Exception("Activity already set")
                else:
                    self._activity = key

            link_factory = get_link_factory(key)
            new_link = link_factory(value)
            self.dict[key] = new_link

            if isinstance(key, Node):
                for buffer, pull_method in self.input_links.items():
                    new_link.watch(buffer, pull_method)

            for construct, construct_link in self.dict.items():
                if connection_allowed(source=construct, target=key):
                    new_link.watch(construct, construct_link.get_pull_method())
                if connection_allowed(source=key, target=construct):
                    construct_link.watch(key, new_link.get_pull_method())

        else:
            raise TypeError("Unexpected type {}".format(type(key)))

    def __delitem__(self, key: Any) -> None:

        del self.dict[key]
        for construct, construct_link in self.dict.items():
            if connection_allowed(key, construct):
                construct_link.drop(key)

        if isinstance(key, Appraisal):
            self._appraisal = None
        elif isinstance(key, Activity):
            self._activity = None

    def watch(self, construct, pull_method):

        super().watch(construct, pull_method)
        for node in self.nodes:
            self[node].watch(construct, pull_method)

    def drop(self, construct):

        super().drop(construct)
        for node in self.nodes:
            self[node].drop(construct)

    def get_pull_method(self):

        return self[self.appraisal].get_pull_method()

    def get_output(self):
        
        return self[self.appraisal].get_output()

    @property
    def nodes(self) -> Iterable[Node]:
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, Node)
        }

    @property
    def flows(self) -> Iterable[Flow]:
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, Flow)
        }

    @property
    def appraisal(self) -> Appraisal:
        
        if self._appraisal:
            return self._appraisal
        else:
            raise AttributeError("Appraisal not set.")

    @property
    def activity(self) -> Activity:

        if self._activity:
            return self._activity
        else:
            raise AttributeError("Activity not set.")