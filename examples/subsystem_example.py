from typing import List
from pyClarion.base.enums import FlowType
from pyClarion.base.symbols import (
    Microfeature, Chunk, Flow, Appraisal, Subsystem
)
from pyClarion.base.packets import ActivationPacket
from pyClarion.base.realizers.basic import (
    NodeRealizer, FlowRealizer, AppraisalRealizer
)
from pyClarion.base.processors import BoltzmannSelector
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


if __name__ == '__main__':

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
        )
    ]

    nacs_realizer = NACSRealizer(
        construct=Subsystem("NACS")
    )
    for realizer in nacs_contents:
        nacs_realizer[realizer.construct] = realizer
    nacs_realizer.input.watch("external_input", external_input)

    nacs_realizer.do()
    for c in nacs_realizer:
        print(c, nacs_realizer[c].output.view())
