"""
This example follows the same pattern as subsystem_example.py, but has as its 
theme vote choice.

A novel mechanism demonstrated here is one possibility for filtering, which is 
implemented at the junction level. 
"""

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
        {Chunk("Self"): 1.0},
        default_factory=default_activation
    )
    
    if nodes:
        return output.subpacket(nodes)
    else:
        return output.copy()

def appraisal_filter(nodes):
    
    output = ActivationPacket(
        {
            Chunk("Bob"): 1.0,
            Chunk("Smith"): 1.0
        },
        default_factory=default_activation,
        origin="Appraisal Filter"
    )
    
    if nodes:
        return output.subpacket(nodes)
    else:
        return output.copy()

class FilteringUpdateJunction(StandardUpdateJunction):

    def __call__(self, *input_maps):

        selector_filter = None
        inputs = []
        for input_map in input_maps:
            if input_map.origin == "Appraisal Filter":
                selector_filter = input_map
            else:
                inputs.append(input_map)
        combined = super().__call__(*inputs)

        output = ActivationPacket(
            {
                chunk: selector_filter[chunk]*combined[chunk] 
                for chunk in combined
            },
            default_factory=self.default_activation
        )
        return output


if __name__ == '__main__':

    interlevel_assoc = {
        Chunk("Self"): {
            "weights": {
                "party": 1.,
                "electoral reform?": .1,
                "abortion policy": 1.
            },
            "microfeatures": {
                Microfeature("party", "Conservative"),
                Microfeature("abortion policy", "pro choice"),
                Microfeature("electoral reform?", False)
            }
        },
        Chunk("Bob"): {
            "weights": {
                "party": 1.,
                "electoral reform?": 1.,
                "abortion policy": 1.,
                "gun policy": 1.
            },
            "microfeatures": {
                Microfeature("party", "Liberal"),
                Microfeature("abortion policy", "pro choice"),
                Microfeature("electoral reform?", True),
                Microfeature("gun policy", "abolish")
            }
        },
        Chunk("Smith"): {
            "weights": {
                "party": 1.,
                "electoral reform?": 1.,
                "abortion policy": 1.,
                "gun policy": 1.
            },
            "microfeatures": {
                Microfeature("party", "Conservative"),
                Microfeature("abortion policy", "pro life"),
                Microfeature("electoral reform?", False),
                Microfeature("gun policy", "free")
            }
        }
    }

    nacs_contents: List = [
        NodeRealizer(
            construct=Chunk("Self"), 
            junction=StandardMaxJunction()
        ),
        NodeRealizer(
            construct=Chunk("Bob"), 
            junction=StandardMaxJunction()
        ),
        NodeRealizer(
            construct=Chunk("Smith"), 
            junction=StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("party", "Liberal"), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("party", "Liberal"), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("party", "Conservative"), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("gun policy", "abolish"), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("gun policy", "free"), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("abortion policy", "pro choice"), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("abortion policy", "pro life"), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("electoral reform?", True), 
            StandardMaxJunction()
        ),
        NodeRealizer(
            Microfeature("electoral reform?", False), 
            StandardMaxJunction()
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
            FilteringUpdateJunction(),
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
    
    # Constant stimulus to self chunk
    nacs_realizer.input.watch("external_input", external_input)

    # Suppression of self chunk activation for decision making, lets candidate chunks through.
    nacs_realizer[Appraisal("NACS")].input.watch("Appraisal filter", appraisal_filter)

    nacs_realizer.do()
    
    for c in nacs_realizer:
        print(c, nacs_realizer[c].output.view())
    print(nacs_realizer[Appraisal("NACS")].output.view().chosen)