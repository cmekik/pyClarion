"""Demonstration and recipe for chunk extraction in pyClarion."""

from pyClarion import *
import pprint

# Let's simulate a subject named 'Alice' who is shopping for some fruits. 
# This simulation demonstrates a basic recipe for chunk extraction in 
# pyClarion. If you have not worked through the free association example, 
# please do so first, as you will be missing some of the prerequisite ideas.

# The basic premise of the recipe is to create a special terminus construct 
# for chunk extraction. On each cycle, this construct recommends new chunks 
# based on the state of the bottom level. These recommendations are then picked 
# up by an updater object, which adds any new chunks to its client chunk 
# database and to affected subsystem(s). At this time the recipe and associated 
# objects are highly experimental.

# Here is the scenario:
# Alice has implicit knowledge about fruits and prices, but is new to the 
# store so she doesn't have any explicit knowledge about store prices. As a 
# result, she is surveying the prices to form some explicit, crystalized 
# knowledge.

#############
### Setup ###
#############

### Agent Setup ###

# Agent setup proceeds more or less as with the free association example. 
# Except that we include a `ChunkAdder` object as an updater when we initialize 
# the agent realizer. 

# The `ChunkAdder` object assumes the chunk database it is responsible for 
# lives in the same realizer as itself. Minimally, the ChunkAdder needs to 
# be given a terminus construct to monitor and a specification for constructing 
# node realizers. Since, in this particular case, the adder lives at the agent 
# level, it is also told which subsystem it should monitor.

alice = Structure(
    name=agent("Alice"),
    emitter=AgentCycle(),
    assets=Assets(chunks=Chunks()),
    updater=ChunkAdder(
        emitter=MaxNode(
            MatchSet(
                ctype=ConstructType.flow_xt,
                constructs={buffer("Stimulus")}
            ),
        ),
        prefix="bl-state",
        terminus=terminus("bl-state"),
        subsystem=subsystem("NACS")
    )
)

stimulus = Construct(name=buffer("Stimulus"), emitter=Stimulus())
alice.add(stimulus)

# For this example, we create a simple NACS w/ horizontal flows (no rules, no 
# associative memory networks).

nacs = Structure(
    name=subsystem("NACS"),
    emitter=NACSCycle(matches=MatchSet(constructs={buffer("Stimulus")}))
)
alice.add(nacs)

nacs.add(
    Construct(
        name=flow_bt("Main"), 
        emitter=BottomUp(chunks=alice.assets.chunks) # type: ignore
    ),
    Construct(
        name=flow_tb("Main"), 
        emitter=TopDown(chunks=alice.assets.chunks) # type: ignore
    )
)

# For the purposes of this example, we continue in the domain of fruits. Fruits 
# are coded as individual values of a "fruits" dimension. Prices are also coded 
# on a five-point scale ranging from "very cheap" to "fair" to "very expensive".
# This encoding is not particularly sophisticated from a psychological point of 
# view, and probably completely unrealistic. It is for illustration purposes 
# only. 

fnodes = [
    Construct(
        name=feature(dim, val),  
        emitter=MaxNode(
            matches=MatchSet(
                ctype=ConstructType.flow_xb, 
                constructs={buffer("Stimulus")}
            ),
        )
    ) for dim, val in [
        ("fruit", "banana"),
        ("fruit", "kiwi"),
        ("fruit", "blueberry"),
        ("fruit", "dragon fruit"),
        ("fruit", "orange"),
        ("fruit", "strawberry"),
        ("price", "very cheap"),
        ("price", "cheap"),
        ("price", "fair"),
        ("price", "expensive"),
        ("price", "very expensive"),
    ]
]
nacs.add(*fnodes)

# As mentioned, we need to create a special terminus construct that produces 
# new chunk recommendations. This is achieved with a `ChunkExtractor` object,
# which assumes that chunks are stored in a `Chunks` object.

nacs.add(
    Construct(
        name=terminus("bl-state"),
        emitter=ThresholdSelector(threshold=0.9)
    )
)

# Agent setup is now complete!

##################
### Simulation ###
##################

stimulus_states = [
    {
        feature("fruit", "dragon fruit"): 1.0,
        feature("price", "expensive"): 1.0
    },
    {
        feature("fruit", "orange"): 1.0,
        feature("price", "expensive"): 1.0
    },
    {
        feature("fruit", "dragon fruit"): 1.0,
        feature("price", "expensive"): 1.0
    },
    {
        feature("fruit", "kiwi"): 1.0,
        feature("price", "very cheap"): 1.0
    },
    {
        feature("fruit", "banana"): 1.0,
        feature("price", "cheap"): 1.0
    }    
]

for i, stimulus_state in enumerate(stimulus_states):
    print("Presentation {}".format(i + 1))
    alice.propagate(kwds={buffer("Stimulus"): {"stimulus": stimulus_state}})
    alice.update()
    pprint.pprint(alice.output)
    alice.clear_outputs()

print("Learned Chunks:")
alice.assets.chunks.pprint()

##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - The basics of using updaters for learning, and
#   - A recipe for chunk extraction from the state of the bottom level. 
