from pyClarion import *

#############
### Setup ###
#############

### Agent Setup ###

alice = Agent(name="Alice") # type: ignore
stimulus = Buffer(name="Stimulus", propagator=Stimulus())
nacs = Subsystem(
    name="NACS",
    matches={buffer("Stimulus")},
    cycle=NACSCycle()
)

alice.add(stimulus, nacs)

# For this example, we create a simple NACS w/ only an input flow (no rules, no 
# associative memory networks, no top-down or bottom-up flows).

nacs.add(
    Flow(
        name=flow_in("Lag"), 
        matches=ConstructType.feature, 
        propagator=Lag(max_lag=1) 
    ),
)

fnodes = [
    Node(
        name=feature(dim, val, lag), 
        matches=MatchSpec(constructs={buffer("Stimulus"), flow_in("Lag")}), 
        propagator=MaxNode()
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
    {feature("dim", "val-1"): 1.0},
    {feature("dim", "val-2"): 1.0},
    {feature("dim", "val-4"): 1.0},
    {feature("dim", "val-3"): 1.0}
]

for i, stimulus_state in enumerate(stimulus_states):
    print("Presentation {}".format(i + 1))
    alice.propagate(args={buffer("Stimulus"): {"stimulus": stimulus_state}})
    print(nacs.output.pstr())

# Clearing outputs during the presentation sequence will interrupt lagged 
# feature computation.    
alice.clear_output()

##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - Using lagged features, and
#   - Using input flows.
