from typing import MutableMapping, Dict, Callable, Any, Iterator, Iterable
from pyClarion.base.symbols import (
    BasicConstructSymbol, Node, Flow, Appraisal, Activity, Subsystem
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


class SubsystemRealizer(
    Propagator[ActivationPacket, DecisionPacket], 
    CompositeConstructRealizer[Subsystem]
):
    """A network of interconnected nodes and flows linked to an actuator."""

    def __init__(self, construct: Subsystem) -> None:

        CompositeConstructRealizer.__init__(self, construct)

        self._nodes: Dict[Node, NodePropagator] = dict()
        self._flows: Dict[Flow, FlowPropagator] = dict()
        self._appraisal: Dict[Appraisal, AppraisalPropagator] = dict()
        self._activity: Dict[Activity, ActivityDispatcher] = dict()
        
        Propagator.__init__(self)

    def __len__(self) -> int:

        return (
            len(self._nodes) +
            len(self._flows) +
            len(self._appraisal) +
            len(self._activity)
        )

    def __contains__(self, obj: Any) -> bool:
        
        if isinstance(obj, Node):
            out = obj in self._nodes
        elif isinstance(obj, Flow):
            out = obj in self._flows
        elif isinstance(obj, Appraisal):
            out = obj in self._appraisal
        elif isinstance(obj, Activity):
            out = obj in self._activity
        else:
            out = False
        return out

    def __iter__(self) -> Iterator:

        for node in self._nodes:
            yield node
        for flow in self._flows:
            yield flow
        for appraisal in self._appraisal:
            yield appraisal
        for activity in self._activity:
            yield activity

    def __getitem__(self, key: Any) -> Any:

        out : Any
        if isinstance(key, Node):
            out = self._nodes[key]
        elif isinstance(key, Flow):
            out = self._flows[key]
        elif isinstance(key, Appraisal):
            out = self._appraisal[key]
        elif isinstance(key, Activity):
            out = self._activity[key]
        else:
            raise TypeError("Unexpected type {}".format(type(key)))
        return out

    def __setitem__(self, key: Any, value: Any) -> None:

        if isinstance(key, Node):
            self._add_node(key, value)
        elif isinstance(key, Flow):
            self._add_flow(key, value)
        elif isinstance(key, Appraisal):
            self._add_appraisal(key, value)
        elif isinstance(key, Activity):
            self._add_activity(key, value)
        else:
            raise TypeError("Unexpected type {}".format(type(key)))

    def __delitem__(self, key: Any) -> None:

        if isinstance(key, Node):
            self._remove_node(key)
        elif isinstance(key, Flow):
            self._remove_flow(key)
        elif isinstance(key, Appraisal):
            self._remove_appraisal(key)
        elif isinstance(key, Activity):
            self._remove_activity(key)
        else:
            raise TypeError("Unexpected type {}".format(type(key)))

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

    def _add_node(self, node: Node, node_realizer: NodeRealizer) -> None:
        
        node_connector = NodePropagator(node_realizer)
        self._nodes[node] = node_connector
        for buffer, pull_method in self.input_links.items():
            node_connector.watch(buffer, pull_method)
        for flow, flow_connector in self._flows.items():
            flow_connector.watch(node, node_connector.get_pull_method())
            node_connector.watch(flow, flow_connector.get_pull_method())
        for appraisal, appraisal_connector in self._appraisal.items():
            appraisal_connector.watch(node, node_connector.get_pull_method())

    def _remove_node(self, node: Node) -> None:
        
        for appraisal, appraisal_connector in self._appraisal.items():
            appraisal_connector.drop(node)
        for flow_connector in self._flows.values():
            flow_connector.drop(node)
        del self._nodes[node]
            
    def _add_flow(self, flow: Flow, flow_realizer: FlowRealizer) -> None:

        flow_connector = FlowPropagator(flow_realizer)
        self._flows[flow] = flow_connector
        for node, node_connector in self._nodes.items():
            node_connector.watch(flow, flow_connector.get_pull_method())
            flow_connector.watch(node, node_connector.get_pull_method())

    def _remove_flow(self, flow: Flow) -> None:
        
        try:
            for node_connector in self._nodes.values():
                node_connector.drop(flow)
            del self._flows[flow]
        except KeyError:
            pass

    def _add_appraisal(
        self, appraisal: Appraisal, appraisal_realizer: AppraisalRealizer
    ) -> None:

        if self._appraisal:
            raise Exception("Appraisal already set")
        elif (
            isinstance(appraisal, Appraisal) and 
            isinstance(appraisal_realizer, AppraisalRealizer)
        ):
            appraisal_propagator = AppraisalPropagator(appraisal_realizer)
            self._appraisal[appraisal] = appraisal_propagator
            for node, node_propagator in self._nodes.items():
                appraisal_propagator.watch(node, node_propagator.get_pull_method())
            for activity, activity_dispatcher in self._activity.items():
                activity_dispatcher.watch(appraisal, appraisal_propagator.get_pull_method()) 
        else:
            # Print informative type error message.
            raise TypeError()

    def _remove_appraisal(self, appraisal: Appraisal) -> None:
        
        for activity, activity_dispatcher in self._activity.items():
            activity_dispatcher.drop(appraisal)
        del self._appraisal[appraisal]

    def _add_activity(
        self, activity: Activity, activity_realizer: ActivityRealizer
    ) -> None:

        if self._activity:
            raise Exception("Activity already set")
        elif (
            isinstance(activity, Activity) and 
            isinstance(activity_realizer, ActivityRealizer)
        ):
            self._activity[activity] = ActivityDispatcher(activity_realizer) 
        else:
            # Print informative type error message.
            raise TypeError()

    def _remove_activity(self, activity: Activity) -> None:

        del self._activity[activity]

    @property
    def nodes(self) -> Iterable[Node]:
        
        return self._nodes.keys()

    @property
    def flows(self) -> Iterable[Flow]:
        
        return self._flows.keys()

    @property
    def appraisal(self) -> Appraisal:
        
        return list(self._appraisal.keys())[0]

    @property
    def activity(self) -> Activity:

        return list(self._activity.keys())[0]