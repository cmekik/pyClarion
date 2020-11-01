from pyClarion import *
import pprint
import logging

logging.basicConfig(level=logging.DEBUG)

#############
### Setup ###
#############

### Agent Setup ###

dv_pairs = [
    ("dim", "val-1"),
    ("dim", "val-2"),
    ("dim", "val-3"),
    ("dim", "val-4"),
    ("dim", "val-5"),
    ("dim", "val-6"),
]

alice = Structure(
    name=agent("alice"),
    emitter=AgentCycle()
)
with alice:

    stimulus = Construct(
        name=buffer("stimulus"), 
        emitter=Stimulus()
    )

    nacs = Structure(
        name=subsystem("nacs"),
        emitter=NACSCycle(
            sources={
                buffer("stimulus")
            }
        )
    )

    # For this example, we create a simple NACS w/ only an input flow (no 
    # rules, no associative memory networks, no top-down or bottom-up flows).

    with nacs:

        Construct(
            name=features("main"),
            emitter= MaxNodes(
                sources={
                    buffer("stimulus"), 
                    flow_in("lag")
                }
            )
        )

        Construct(
            name=flow_in("lag"), 
            emitter=Lag(
                source=features("main"), 
                max_lag=1
            ) 
        )

# Agent setup is now complete!


##################
### Simulation ###
##################

stimuli = [
    {feature("dim", "val-1"): 1.0},
    {feature("dim", "val-2"): 1.0},
    {feature("dim", "val-4"): 1.0},
    {feature("dim", "val-3"): 1.0}
]

alice.start()

for i, stim in enumerate(stimuli):
    print("Presentation {}".format(i + 1))
    stimulus.emitter.input(stim)
    alice.step()
    pprint.pprint(alice.output)


##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - Using lagged features, and
#   - Using input flows.
