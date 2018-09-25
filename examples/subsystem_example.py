from pyClarion.base.symbols import Node, Microfeature, Chunk, Flow, FlowType, Appraisal, Subsystem
from pyClarion.base.packets import Level, ActivationPacket, DecisionPacket
from pyClarion.base.processors import UpdateJunction, MaxJunction, Channel, BoltzmannSelector
from pyClarion.base.realizers.basic import NodeRealizer, FlowRealizer, AppraisalRealizer
from pyClarion.base.realizers.composite import SubsystemRealizer


def default_factory(key=None):
    
    return 0.0


class AssociativeRules(Channel[float]):

    def __init__(self, assoc):

        self.assoc = assoc

    def __call__(self, input_map):
        
        output = ActivationPacket(
            default_factory=default_factory, origin=Level.TopLevel
        )
        for chunk in self.assoc:
            for cond_chunk in self.assoc[chunk]: 
                output[chunk] += (
                    self.assoc[chunk][cond_chunk] * 
                    input_map.get(cond_chunk, default_factory())
                )
            output[chunk] = min(1., output[chunk])
        return output


class TopDown(Channel[float]):

    def __init__(self, assoc):

        self.assoc = assoc

    def __call__(self, input_map):

        output = ActivationPacket(
            default_factory=default_factory, origin=Level.TopLevel
        )
        for node in input_map:
            if isinstance(node, Chunk) and node in self.assoc:
                mfs = self.assoc[node]["microfeatures"]
                weights = self.assoc[node]["weights"]
                for mf in mfs:
                    output[mf] = max(
                        output[mf],
                        weights[mf.dim] * input_map[node]
                    )
        return output


class BottomUp(Channel[float]):

    def __init__(self, assoc):

        self.assoc = assoc

    def __call__(self, input_map):

        output = ActivationPacket(
            default_factory=default_factory, origin=Level.BottomLevel
        )
        for chunk in self.assoc:
            microfeatures = self.assoc[chunk]["microfeatures"]
            weights = self.assoc[chunk]["weights"]
            dim_activations = dict()
            for mf in microfeatures:
                dim_activations[mf.dim] = max(
                    dim_activations.get(mf.dim, default_factory()),
                    input_map.get(mf, default_factory())
                )
            for dim in dim_activations:
                output[chunk] += (
                    weights[dim] * dim_activations[dim]
                )
            output[chunk] /= len(weights) ** 1.1
        return output


class NACSRealizer(SubsystemRealizer):

    def __call__(self):

        # Update Chunks
        for node, node_propagator in self._nodes.items():
            if isinstance(node, Chunk):
                node_propagator()

        # Propagate Top-down Flows
        for flow, flow_propagator in self._flows.items():
            if flow.flow_type == FlowType.TopDown:
                flow_propagator()
        
        # Update Microfeatures
        for node, node_propagator in self._nodes.items():
            if isinstance(node, Microfeature):
                node_propagator()
        
        # Simultaneously Process at Both Top and Bottom Levels
        for flow, flow_propagator in self._flows.items():
            if flow.flow_type in (FlowType.TopLevel, FlowType.BottomLevel):
                flow_propagator()
        
        # Update All Nodes
        for node, node_propagator in self._nodes.items():
            node_propagator()
        
        # Propagate Bottom-up Links
        for flow, flow_propagator in self._flows.items():
            if flow.flow_type == FlowType.BottomUp:
                flow_propagator()
        
        # Update Chunks
        for node, node_propagator in self._nodes.items():
            if isinstance(node, Chunk):
                node_propagator()
        
        # Update Appraisal
        for appraisal, appraisal_propagator in self._appraisal.items():
            appraisal_propagator()
        
        # Execute Activities
        for activity, activity_dispatcher in self._activity.items():
            activity_dispatcher()

toplevel_assoc = {
    Chunk("FRUIT"): {
        Chunk("APPLE"): 1.
    }
}

interlevel_assoc = {
    Chunk("APPLE"): {
        "weights": {
            "color": 1.,
            "tasty": 1.
        },
        "microfeatures": {
            Microfeature("color", "red"), 
            Microfeature("tasty", True)
        }
    },
    Chunk("JUICE"): {
        "weights": {
            "tasty": 1,
            "state": 1
        },
        "microfeatures": {
            Microfeature("tasty", True),
            Microfeature("state", "liquid")
        }
    },
    Chunk("FRUIT"): {
        "weights": {
            "tasty": 1,
        },
        "microfeatures": {
            Microfeature("tasty", True)
        }
    }
}

nacs_contents = [
    NodeRealizer(
        Chunk("APPLE"), 
        MaxJunction()
    ),
    NodeRealizer(
        Chunk("JUICE"), 
        MaxJunction()
    ),
    NodeRealizer(
        Chunk("FRUIT"), 
        MaxJunction()
    ),
    NodeRealizer(
        Microfeature("color", "red"), 
        MaxJunction()
    ),
    NodeRealizer(
        Microfeature("tasty", True), 
        MaxJunction()
    ),
    NodeRealizer(
        Microfeature("state", "liquid"), 
        MaxJunction()
    ),
    FlowRealizer(
        Flow("GKS", flow_type=FlowType.TopLevel),
        UpdateJunction(),
        AssociativeRules(
            assoc = toplevel_assoc
        )
    ),
    FlowRealizer(
        Flow("NACS", flow_type=FlowType.TopDown),
        UpdateJunction(),
        TopDown(
            assoc = interlevel_assoc
        )
    ),
    FlowRealizer(
        Flow("NACS", flow_type=FlowType.BottomUp),
        UpdateJunction(),
        BottomUp(
            assoc = interlevel_assoc
        )
    ),
    AppraisalRealizer(
        Appraisal("NACS"),
        UpdateJunction(),
        BoltzmannSelector(
            temperature = .1
        )
    )
]

nacs_realizer = NACSRealizer(Subsystem("NACS"))
for realizer in nacs_contents:
     nacs_realizer[realizer.construct] = realizer

nacs_realizer.watch(
    "external", lambda key: ActivationPacket({Chunk("APPLE"): 1.0})
)

nacs_realizer()

for c in nacs_realizer:
    print(c, nacs_realizer[c].get_output())
print(nacs_realizer.construct, nacs_realizer.get_output())