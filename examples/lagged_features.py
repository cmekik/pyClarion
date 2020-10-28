from pyClarion import *
import pprint
import logging

logging.basicConfig(level=logging.DEBUG)

#############
### Setup ###
#############

### Agent Setup ###

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
            sources={buffer("stimulus")}
        )
    )

    # For this example, we create a simple NACS w/ only an input flow (no 
    # rules, no associative memory networks, no top-down or bottom-up flows).

    with nacs:

        Construct(
            name=features("main"),
            emitter= MaxNodes(
                sources={buffer("stimulus"), flow_in("lag")}
            )
        )

        Construct(
            name=flow_in("lag"), 
            emitter=Lag(source=features("main"), max_lag=1) 
        )

# Agent setup is now complete!

base_dv_pairs = [
    ("dim", "val-1", 0),
    ("dim", "val-2", 0),
    ("dim", "val-3", 0),
    ("dim", "val-4", 0),
    ("dim", "val-5", 0),
    ("dim", "val-6", 0),
]


##################
### Simulation ###
##################

stimulus_states = [
    {feature("dim", "val-1", 0): 1.0},
    {feature("dim", "val-2", 0): 1.0},
    {feature("dim", "val-4", 0): 1.0},
    {feature("dim", "val-3", 0): 1.0}
]

for i, stimulus_state in enumerate(stimulus_states):
    print("Presentation {}".format(i + 1))
    stimulus.emitter.input(stimulus_state)
    alice.propagate()
    pprint.pprint(alice.output)

# Clearing outputs during the presentation sequence will interrupt lagged 
# feature computation.    
alice.clear_outputs()

##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - Using lagged features, and
#   - Using input flows.
