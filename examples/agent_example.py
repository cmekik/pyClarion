"""
A simple pyClarion demo.

Below we simulate an agent, named Alice, doing a cued association task. She will 
tell us the first thing that comes to her mind upon being presented with the 
chosen stimulus.
"""

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
from pyClarion.base.processors import (
    UpdateJunction, MaxJunction, BoltzmannSelector, MappingEffector, 
    ConstantSource
)
from pyClarion.standard.common import default_activation
from pyClarion.standard.nacs import (
    AssociativeRulesChannel, TopDownChannel, BottomUpChannel, NACSRealizer
)
from pyClarion.base.updates import UpdateManager


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


class HeavyHandedUpdateManager(UpdateManager):
    
    def update(self) -> None:

        nacs = self.get_realizer(Subsystem("NACS"))
        nacs[Chunk("ORANGE")] = NodeRealizer(
            Chunk("ORANGE"),
            MaxJunction()
        )
        nacs[Microfeature("color", "#ffa500")] = NodeRealizer(
            Microfeature("color", "#ffa500"), # "ORANGE"
            MaxJunction()
        )
        nacs[Flow("GKS", flow_type=FlowType.Top2Top)].channel.assoc.append(
            [Chunk("FRUIT"), {Chunk("ORANGE"): 1.}]
        )
        nacs[
            Flow("NACS", flow_type=FlowType.Top2Bot)
        ].channel.assoc[Chunk("ORANGE")] = {
            "weights": {
                "color": 1.,
                "tasty": 1.
            },
            "microfeatures": {
                Microfeature("color", "#ffa500"), # "RED"
                Microfeature("tasty", True)
            }
        }
        nacs[
            Flow("NACS", flow_type=FlowType.Bot2Top)
        ].channel.assoc[Chunk("ORANGE")] = {
            "weights": {
                "color": 1.,
                "tasty": 1.
            },
            "microfeatures": {
                Microfeature("color", "#ffa500"), # "RED"
                Microfeature("tasty", True)
            }
        }


class BehaviorRecorder(object):

    def __init__(self):

        self.recorded_actions = []


if __name__ == '__main__':


    #############
    ### SETUP ###
    #############

    ### Environment Setup ###
    
    # The object below allows us to record the agent's response
    behavior_recorder = BehaviorRecorder()

    ### End Environment Setup ###

    ### Agent Setup ###
        # In this section, we'll create an agent named 'Alice' with some 
        # knowledge about fruits.

    # Alices's initial top-level knowledge

    toplevel_assoc = [
        (
            Chunk("FRUIT"),
            (
                (Chunk("APPLE"), 1.),
            )
        )
    ]

    # Alice's initial inter-level associations

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

    # NACS Prep
        # Here, realizers are created for every initial piece of knowledge that 
        # alice has in addition to other important functional components of 
        # Alice's NACS.

    nacs_contents: List = [
        NodeRealizer(
            construct=Chunk("APPLE"), 
            junction=MaxJunction()
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
            Microfeature("color", "#ff0000"), 
            MaxJunction()
        ),
        NodeRealizer(
            Microfeature("color", "#008000"), 
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
            Flow("GKS", flow_type=FlowType.Top2Top),
            UpdateJunction(),
            AssociativeRulesChannel(
                assoc = toplevel_assoc
            ),
            default_activation=default_activation
        ),
        FlowRealizer(
            Flow("NACS", flow_type=FlowType.Top2Bot),
            UpdateJunction(),
            TopDownChannel(
                assoc = interlevel_assoc
            ),
            default_activation=default_activation
        ),
        FlowRealizer(
            Flow("NACS", flow_type=FlowType.Bot2Top),
            UpdateJunction(),
            BottomUpChannel(
                assoc = interlevel_assoc
            ),
            default_activation=default_activation
        ),
        AppraisalRealizer(
            Appraisal("NACS"),
            UpdateJunction(),
            BoltzmannSelector(
                temperature = .1
            )
        ),
        BehaviorRealizer(
            Behavior("NACS"),
            MappingEffector(
                chunk2callback={
                    Chunk("APPLE"): lambda: behavior_recorder.recorded_actions.append(Chunk("APPLE")),
                    Chunk("JUICE"): lambda: behavior_recorder.recorded_actions.append(Chunk("JUICE")),
                    Chunk("FRUIT"): lambda: behavior_recorder.recorded_actions.append(Chunk("FRUIT"))
                }
            )
        )
    ]

    # We create an AgentRealizer representing alice
    alice = AgentRealizer(Agent("Alice"))
    
    # We add an NACS
    alice[Subsystem("NACS")] = NACSRealizer(Subsystem("NACS"))
    
    # We insert the components defined above into Alice's NACS
    for realizer in nacs_contents:
        alice[Subsystem("NACS")][realizer.construct] = realizer 

    # We add a buffer enabling stimulation Alice's NACS
    alice[Buffer("NACS Stimulus")] = BufferRealizer(
        construct=Buffer("NACS Stimulus"),
        source=ConstantSource(),
        default_activation=default_activation
    )

    # Next, we connect the stimulus buffer to Alice's NACS
    alice[Subsystem("NACS")].input.watch(
        Buffer("NACS Stimulus"), 
        alice[Buffer("NACS Stimulus")].output.view
    )

    # Finally, we attach an update manager to handle some heavy-handed learning
    alice.attach(
        HeavyHandedUpdateManager()
    )

    ### End Agent Setup ###

    ##################
    ### SIMULATION ###
    ##################

    ### First Trial ###

    # The stimulus buffer is set to provide constant activation to the Apple 
    # chunk. This represents presentation of the concept APPLE. Note that there 
    # are simplifications. For instance, it is assumed that Alice is made to 
    # understand that the cue is APPLE, the fruit, and not e.g., APPLE, the 
    # company.
    alice[Buffer("NACS Stimulus")].source.update(
        ActivationPacket({Chunk("APPLE"): 1.})
    )

    # Alice performs one NACS cycle 
    alice.propagate()

    # Alice responds
    alice.execute()


    ### End First Trial ###

    # We can look at the exact state of Alice's NACS at the end of its 
    # activation cycle.
    print("TRIAL 1")
    for c in alice[Subsystem("NACS")]:
        if not isinstance(c, Behavior):
            print(c, alice[Subsystem("NACS")][c].output.view())

    ### Start Second Trial ###

    # To demonstrate learning, the code below repeats the same task, but only 
    # after Alice suddenly and inexplicably learns about oranges.

    alice.learn()

    # Now we run through the trial again
    alice.propagate()
    alice.execute()

    # Here is Alice's cognitive state at the end of the second trial
    print("TRIAL 2")
    for c in alice[Subsystem("NACS")]:
        if not isinstance(c, Behavior):
            print(c, alice[Subsystem("NACS")][c].output.view())

    # We can also observe her responses in the simulation environment.
    print(behavior_recorder.recorded_actions)
