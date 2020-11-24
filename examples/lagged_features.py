"""
A demo for working with lagged features.

Prerequisite: Understanding of the basics of pyClarion as discussed in the demo
`free_association.py`.
"""


from pyClarion import (
    Structure, Construct,
    agent, buffer, subsystem, feature, features, flow_in, terminus,
    AgentCycle, NACSCycle, Lag, Stimulus, MaxNodes,
    pprint
)
import logging

logging.basicConfig(level=logging.DEBUG)


#############
### Setup ###
#############

# This demo shows how lagged features can be created on the fly within a 
# simulation using the Lag component and flow_in constructs. 

### Agent Setup ###

# The feature domain for this simple demo is the same as for 
# `free_association.py`.

feature_spec = [
    feature("color", "#ff0000"), # red
    feature("color", "#008000"), # green
    feature("tasty"),
    feature("state", "liquid"),
    feature("sweet")
]

# For this example, the agent architecture is minimal. We instantiate a 
# stimulus buffer and an NACS containing only a feature pool and the Lag 
# component.

# The lag component serves the construct `flow_in("lag")` which maps, at 
# the start of the NACS cycle, activations in the feature pool held over from 
# the previous cycle to activations of corresponding lagged features up to a 
# set maximum lag value.

# In general, `flow_in()` constructs take care of any necessary activation 
# processing within a subsystem at the start of an activation cycle. Besides 
# helping compute lagged strengths, these constructs are useful for a variety 
# of purposes that involve gating or transforming inputs to the subsystem.


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
        emitter=NACSCycle()
    )

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


##################
### Simulation ###
##################

# For this simulation, we present a stream of four distinct stimuli and 
# output the state of the agent after each presentation. 

# In the output, it can be seen that the activation of a lag-0 feature 
# activated at time t is copied over, at time t+1, to its lag-1 counterpart.

stimuli = [
    {feature("color", "#008000"): 1.0},
    {feature("sweet"): 1.0},
    {feature("tasty"): 1.0},
    {feature("state", "liquid"): 1.0}
]

for i, s in enumerate(stimuli):
    print("Presentation {}".format(i + 1))
    stimulus.emitter.input(s)
    alice.step()
    pprint(alice.output)


##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - Using lagged features, and
#   - Using input flows.
