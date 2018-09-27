from typing import List
from pyClarion.base.symbols import Node, Microfeature, Chunk, Flow, FlowType, Appraisal, Subsystem
from pyClarion.base.packets import Level, ActivationPacket, DecisionPacket
from pyClarion.base.processors import UpdateJunction, MaxJunction, Channel, BoltzmannSelector
from pyClarion.base.realizers.abstract import BasicConstructRealizer
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
        for rule in self.assoc:
            conditions = rule["conditions"]
            conclusion = rule["conclusion"]
            strength = 0.
            for cond in conditions: 
                strength += (
                    conditions[cond] * 
                    input_map.get(cond, default_factory())
                )
            output[conclusion] = max(output[conclusion], strength)
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
        for node in self.nodes:
            if isinstance(node, Chunk):
                self[node]()

        # Propagate Top-down Flows
        for flow in self.flows:
            if flow.flow_type == FlowType.TopDown:
                self[flow]()
        
        # Update Microfeatures
        for node in self.nodes:
            if isinstance(node, Microfeature):
                self[node]()
        
        # Simultaneously Process at Both Top and Bottom Levels
        for flow in self.flows:
            if flow.flow_type in (FlowType.TopLevel, FlowType.BottomLevel):
                self[flow]()
        
        # Update All Nodes
        for node in self.nodes:
            self[node]()
        
        # Propagate Bottom-up Links
        for flow in self.flows:
            if flow.flow_type == FlowType.BottomUp:
                self[flow]()
        
        # Update Chunks
        for node in self.nodes:
            if isinstance(node, Chunk):
                self[node]()
        
        # Update Appraisal
        self[self.appraisal]()


toplevel_assoc = [
    {
        "conclusion": Chunk("FRUIT"), 
        "conditions": {
            Chunk("APPLE"): 1.
        }
    }
]


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


nacs_contents: List[BasicConstructRealizer] = [
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
    "external", lambda key: ActivationPacket(
        {Chunk("APPLE"): 1.0},
        default_factory=default_factory
    ).subpacket(key)
)


nacs_realizer()


for c in nacs_realizer:
    print(c, nacs_realizer[c].get_output())
print(nacs_realizer.construct, nacs_realizer.get_output())