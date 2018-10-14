from typing import List, Iterable
from pyClarion.base.enums import FlowType
from pyClarion.base.symbols import (
    Microfeature, Chunk, Flow, Appraisal, Subsystem, Agent, Buffer, Behavior
)
from pyClarion.base.packets import ActivationPacket
from pyClarion.base.realizers.basic import (
    NodeRealizer, FlowRealizer, AppraisalRealizer, BufferRealizer, BehaviorRealizer
)
from pyClarion.base.realizers.agent import AgentRealizer
from pyClarion.base.processors import BoltzmannSelector, MappingEffector
from pyClarion.standard.common import (
    default_activation, StandardMaxJunction, StandardUpdateJunction
)
from pyClarion.standard.nacs import (
    AssociativeRules, TopDownChannel, BottomUpChannel, NACSRealizer
)


def external_input(nodes):
    
    output = ActivationPacket(
        {Chunk("APPLE"): 1.0},
        default_factory=default_activation
    )
    
    if nodes:
        return output.subpacket(nodes)
    else:
        return output.copy()

class BehaviorRecorder(object):

    def __init__(self):

        self.recorded_actions = []


if __name__ == '__main__':

    behavior_recorder = BehaviorRecorder()

    toplevel_assoc = [
        (
            Chunk("FRUIT"),
            (
                (Chunk("APPLE"), 1.),
            )
        )
    ]

    interlevel_assoc = {
        Chunk("APPLE"): {
            "weights": {
                "color": 1.,
                "tasty": 1.
            },
            "microfeatures": {
                Microfeature("color", "#ff0000"), # "RED"
                Microfeature("color", "#008000"), # "GREEN"
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

    chunk2callback = {
        Chunk("APPLE"): lambda: behavior_recorder.recorded_actions.append(Chunk("APPLE")),
        Chunk("JUICE"): lambda: behavior_recorder.recorded_actions.append(Chunk("JUICE")),
        Chunk("FRUIT"): lambda: behavior_recorder.recorded_actions.append(Chunk("FRUIT"))
    }

    nacs_contents: List = [
        NodeRealizer(
            construct=Chunk("APPLE"), 
            junction=StandardMaxJunction()
        ),
        NodeRealizer(
            Chunk("JUICE"), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Chunk("FRUIT"), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("color", "#ff0000"), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("color", "#008000"), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("tasty", True), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("state", "liquid"), 
            StandardMaxJunction()
        ),
        FlowRealizer(
            Flow("GKS", flow_type=FlowType.Top2Top),
            StandardUpdateJunction(),
            AssociativeRules(
                assoc = toplevel_assoc
            )
        ),
        FlowRealizer(
            Flow("NACS", flow_type=FlowType.Top2Bot),
            StandardUpdateJunction(),
            TopDownChannel(
                assoc = interlevel_assoc
            )
        ),
        FlowRealizer(
            Flow("NACS", flow_type=FlowType.Bot2Top),
            StandardUpdateJunction(),
            BottomUpChannel(
                assoc = interlevel_assoc
            )
        ),
        AppraisalRealizer(
            Appraisal("NACS"),
            StandardUpdateJunction(),
            BoltzmannSelector(
                temperature = .1
            )
        ),
        BehaviorRealizer(
            Behavior("NACS"),
            MappingEffector(
                chunk2callback=chunk2callback
            )
        )
    ]

    def create_standard_agent(
        name: str,
        has_nacs: bool = False,
        nacs_contents: Iterable = None
    ):

        agent = AgentRealizer(Agent(name))

        if has_nacs:
            nacs_symb = Subsystem("NACS")
            nacs_realizer = NACSRealizer(nacs_symb)
            agent[nacs_symb] = nacs_realizer

        if nacs_contents:
            for realizer in nacs_contents:
                nacs_realizer[realizer.construct] = realizer 
        
        return agent

    alice = create_standard_agent(
        name="Alice",
        has_nacs=True,
        nacs_contents=nacs_contents
    )

    alice[Subsystem("NACS")].input.watch("external_input", external_input)

    alice.propagate()
    alice.execute()

    for c in alice[Subsystem("NACS")]:
        if not isinstance(c, Behavior):
            print(alice.construct, c, alice[Subsystem("NACS")][c].output.view())
    print(behavior_recorder.recorded_actions)