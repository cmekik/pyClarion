from pyClarion import *
import pprint

#############
### Setup ###
#############

### Agent Setup ###

alice = Structure(
    name=agent("Alice"),
    emitter=AgentCycle()
)
stimulus = Construct(name=buffer("Stimulus"), emitter=Stimulus())
nacs = Structure(
    name=subsystem("NACS"),
    emitter=NACSCycle(matches=MatchSet(constructs={buffer("Stimulus")}))
)

alice.add(stimulus, nacs)

# For this example, we create a simple NACS w/ only an input flow (no rules, no 
# associative memory networks, no top-down or bottom-up flows).

nacs.add(
    Construct(
        name=flow_in("Lag"), 
        emitter=Lag(max_lag=1) 
    ),
)

fnodes = [
    Construct(
        name=feature(dim, val, lag), 
        emitter=MaxNode(
            matches=MatchSet(constructs={buffer("Stimulus"), flow_in("Lag")}) 
        )
    ) for dim, val, lag in [
        ("dim", "val-1", 0),
        ("dim", "val-2", 0),
        ("dim", "val-3", 0),
        ("dim", "val-4", 0),
        ("dim", "val-5", 0),
        ("dim", "val-6", 0),
        ("dim", "val-1", 1),
        ("dim", "val-2", 1),
        ("dim", "val-3", 1),
        ("dim", "val-4", 1),
        ("dim", "val-5", 1),
        ("dim", "val-6", 1)
    ]
]
nacs.add(*fnodes)

# Agent setup is now complete!

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
    alice.propagate(kwds={buffer("Stimulus"): {"stimulus": stimulus_state}})
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
