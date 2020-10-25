"""
A simple pyClarion demo.

This demo simulates an agent, named Alice, doing a free association task, where 
the goal is to report the first thing that comes to mind upon presentation of 
a cue.
"""


# Import notes may be skipped on first reading. They are for clarification 
# purposes only.
from pyClarion import (
    # Below are realizer objects, implementing the behavior of simulated 
    # constructs.
    Structure, Construct,
    # Construct types are used in controlling construct behavior.
    ConstructType, MatchSet, Assets,
    # Below are functions for constructing construct symbols, which are used to 
    # name, index and reference simulated constructs
    agent, subsystem, buffer, feature, chunk, terminus, flow_tt, flow_tb, 
    flow_bt, chunks, features,
    # The objects below house datastructures handling various important 
    # concerns.
    Chunks, Rules,
    # The objects below define how realizers process activations in the forward 
    # direction.
    Stimulus, AssociativeRules, BottomUp, TopDown, BoltzmannSelector, MaxNodes, 
    FilteredT, NACSCycle, AgentCycle
)
import pprint
import logging

logging.basicConfig(level=logging.DEBUG)

#############
### Setup ###
#############

### Agent Setup ###

# Let's simulate a subject named 'Alice' with some knowledge about fruits.

# PyClarion agents are created by assembling construct realizers, which are 
# objects instantiating theoretical constructs. Much of the assembly process is 
# automated, so agent construction amounts to declaratively specifying the 
# necessary constructs. There are broadly two main types of construct: 
# structures, which may contain other constructs, and basic constructs (or 
# 'constructs' for short), which may not contain other constructs.

# We begin by creating the top-level construct: a Structure object representing 
# the agent Alice.

alice = Structure(
    name=agent("alice"),
    emitter=AgentCycle(),
    assets=Assets(chunks=Chunks())
)

# The `name` argument to the Structure constructor serves to label the 
# construct. It is mandatory to provide a name argument to construct realizers, 
# as names enable automation of important behavior, such as linking/unlinking 
# constructs. 

# Constructs are named using 'construct symbols'. Here, we see a construct 
# symbol explicitly invoked for the first time through the convenience function 
# `agent()`, which takes a hashable object and returns a construct symbol for 
# an agent construct. Typically, construct symbols are created through 
# convenience functions, as manually specifying them is rather tedious. For 
# instance, the result of `agent("Alice")` can be directly created with the 
# longer expression `Symbol("agent", "Alice")`.

# To keep track of the concepts that Alice knows about, we equip Alice with a 
# chunk database (more on chunks below). This is done by passing an `Assets` 
# object, which is given a chunk database to be stored in its `chunks` 
# attribute, as the agent's 'assets' attribute. The `assets` attribute provides 
# a namespace for convenient storage of resources shared by construct realizers 
# subordinate to Alice. All Structure objects have the `assets` attribute. The 
# `Assets` object is uncomplicated. It simply records all arguments passed to 
# it as attributes. 

# A good rule of thumb is to place shared resources in the structure directly 
# in or above the highest-level construct using the resource.

# The next step in agent construction is to populate `alice` with components 
# representing various cognitive structures postulated by the Clarion theory.
# This amounts to constructing, node by node, a complex network of networks.

# Constructs may be added to `Structure` objects using the `add()` method, like 
# `alice.add()`. A call to alice.add() on some construct, places that construct 
# within Alice's cognitive apparatus and establishes any links specified 
# between the new construct and any existing consturct within Alice. To do 
# this, the `Structure` object automatically checks all constructs it contains 
# and links up those that match.

# To facilitate the construction process, pyClarion borrows a pattern from 
# the nengo library. When a pyClarion construct is initialized in a with 
# statement where the context manager is a pyClarion `Structure`, the construct 
# is automatically added to the structure serving as the context manager. Under 
# the hood, the with syntax simply stores constructs created within it in a 
# temporary helper variable and, upon exit, adds the constructs to the parent 
# structure using its `add()` methtod (the actual process is slightly more 
# complex because it allows nested use of the with statement in this way).

with alice:

    # For this simulation, there are two main constructs at the agent-level: 
    # the stimulus and the non-action-centered subsystem (NACS). The stimulus 
    # is uncomplicated: it is simply an abstract representation of the task 
    # cue. The NACS, on the other hand, is the Clarion subsystem that is 
    # responsible for processing declarative knowledge. Knowledge is 
    # represented by chunk and feature nodes within the NACS. These nodes 
    # receive activations from external buffers and each other, and they 
    # compete to be selected as the output of NACS at each simulation step.

    # Stimulus

    # We begin by adding the stimulus component to the model. 

    stimulus = Construct(name=buffer("stimulus"), emitter=Stimulus())

    # We represent the stimulus with a buffer construct, which is a top-level 
    # construct within an agent that stores and relays activations to various 
    # subsystems. As before, we first create a construct realizer. 

    # The Stimulus object passed in as the emitter allows us to set stimulus 
    # values on the fly. 

    # Non-Action-Centered Subsystem

    # Next, we set up a realizer for the Non-Action-Centered Subsystem. The 
    # setup is similar, but we create a Structure object because subsystems may 
    # contain other constructs. The emitter is one that implements the desired 
    # activation cycle for NACS. 

    nacs = Structure(
        name=subsystem("nacs"),
        emitter=NACSCycle(
            sources={buffer("stimulus")}
        ),
        assets=Assets(rules=Rules())
    )

    # The 'matches' argument lets the subsystem know that it should receive 
    # input from the stimulus (more specifically, from the buffer construct 
    # representing the stimulus).

    # As before, we add assets. In this case, we are interested in equipping 
    # the NACS with a rule database, to be used in reasoning. In reality, this 
    # database is only used by a single construct realizer. However, it helps 
    # to keep a reference to it at the level of NACS as other objects or 
    # processes, such as learning rules, base-level activation trackers, 
    # loggers etc., may need access to the rule database.

    # Now, it is time to populate the NACS. This involves adding in chunk and 
    # feature nodes to represent any initial knowledge we think `alice` should 
    # have and adding in various other components for processing given 
    # information and selecting responses. 

    # Note that we can use nested with statements to add constructs into the 
    # NACS. The library will do this correctly and add objects constructed in 
    # the nested statement only to the parent Structure object (i.e., the 
    # NACS).

    with nacs:

        # Chunk and Feature Pools

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={flow_tb("main")}
            )
        )

        Construct(
            name=chunks("main"),
            emitter=MaxNodes(
                sources={
                    buffer("stimulus"), 
                    flow_bt("main"), 
                    flow_tt("associations")
                }
            )
        )

        # Flows

        # Flows are an abstraction native to `pyClarion`; they represent 
        # processes within subsystems that map node activations to node 
        # activations. In other words, 'flow' is pyClarion's umbrella term for 
        # the various neural networks and rule systems that may live within a 
        # Clarion subsystem. For example, a collection of associative rules in 
        # the top level of the NACS or some neural network module in the bottom 
        # level would each be represented by a corresponding flow construct.

        # For this simulation, we will create three flows. The first processes 
        # (in the top level) associative rules known to Alice, the other two 
        # links Alice's explicit (top-level) and implicit (bottom-level) 
        # declarative knowledge.

        Construct(
            name=flow_tt("associations"),
            emitter=AssociativeRules(
                source=chunks("main"),
                rules=nacs.assets.rules
            ) 
        )

        Construct(
            name=flow_bt("main"), 
            emitter=BottomUp(
                source=features("main"),
                chunks=alice.assets.chunks
            ) 
        )

        Construct(
            name=flow_tb("main"), 
            emitter=TopDown(
                source=chunks("main"),
                chunks=alice.assets.chunks
            ) 
        )

        # Note that the syntax for initializing flows is essentially the same 
        # as before, only requiring appropriate argument choices.

        # There are different kinds of flows, as indicated by the different 
        # constructors `flow_tt`, `flow_tb`, and `flow_bt`. The subdivisions 
        # are based on the source and destination of flows. For instance 'tb' 
        # stands for 'from the top level to the bottom level'. These 
        # designations serve to help accurately connect constructs and control 
        # activation cycles.

        # Responses

        # It is necessary to specify how the NACS should choose its output at 
        # the end of an activation cycle. To do this, we add a terminus 
        # construct.

        # The output terminus is identified with the construct symbol 
        # `terminus("Main")`. In more complex simulations, a single subsystem 
        # may contain several terminus nodes.
        
        Construct(
            name=terminus("main"),
            emitter=FilteredT(
                base=BoltzmannSelector(
                    source=chunks("main"),
                    temperature=.1
                ),
                filter=buffer("stimulus")
            )
        )

        # The output selection process in this example involves the 
        # construction of a Boltzmann distribution from chunk node activations. 
        # On each activation cycle, a chunk is sampled from this distribution 
        # and passed on as the selected output.

        # To prevent information in the stimulus from interfering with output 
        # selection, the `BoltzmanSelector` is wrapped in a `FilteredT` object. 
        # This object is configured to filter inputs to the selector 
        # proportionally to their strengths in the stimulus buffer. This is an 
        # example of cue-suppression.

# We are now done populating Alice with constructs, but we still need to give 
# her some knowledge. 

# Linking Up Nodes within NACS

# For Alice to produce associations between concepts, we must establish some 
# direct and indirect links among the concepts within her mind. We do this by 
# providing some initial knowledge to NACS flows in the form of some associative 
# rules and some links between top-level chunk nodes and bottom-level feature 
# nodes.

# Defining our Working Nodes

# We must first define the feature and chunk nodes we will use in the 
# simulation. These definitions are for ease of use. In practice, nodes are 
# defined by the constructs that use them. 

# Feature nodes represent Alice's implicit knowledge about the world. Each 
# feature node is associated with a unique dimension-value pair (dv pair) 
# indicating its dimension (e.g., color) and value (e.g., red). For this 
# simulation, we include (somewhat arbitrarily) feature nodes for the colors 
# red and green and a feature for each of tastiness, sweetness and the liquid 
# state. These dv pairs are listed below.

dim_val_pairs = [
    ("color", "#ff0000"), # red
    ("color", "#008000"), # green
    ("tasty", True),
    ("state", "liquid"),
    ("sweet", True)
]

# Feature values for red and green are given in hex code to emphasize 
# the idea that features in Clarion theory represent implicit knowledge 
# (since most people don't know the meaning of hex color codes off the 
# top of their head). In practice, it is better to label features in a 
# way that is intelligible to readers.

# Chunk nodes correspond roughly to concepts known to Alice. Chunk nodes are 
# simpler to identify than feature nodes in that they are differentiated only 
# by their names, which are taken to be purely formal labels. We will represent 
# chunk nodes for the concepts FRUIT, APPLE, and JUICE.

chunk_names = ["FRUIT", "APPLE", "JUICE"]

# Now that we've defined the symbols we will be working with, we populate Alice 
# with some knowledge.

# We can add rules to the `link()` method of the rule database (i.e., the 
# `Rules()` object stored in NACS). 

# The argument signature for `link()` is the conclusion chunk followed by one 
# or more condition chunks. Thus, below, `chunk("FRUIT")` is the conclusion and 
# `chunk("APPLE")` is the only condition. In other words, this rule establishes 
# an association from the concept APPLE to the concept FRUIT. This association 
# is meant to capture the fact that apples are fruits. 

nacs.assets.rules.link(chunk("FRUIT"), chunk("APPLE")) # type: ignore

# We proceed in much the same way to link chunk and feature nodes. 

# The chunk database also has a `link()` method, which can be used to link a 
# chunk node to its microfeature nodes. The call signature expects the chunk 
# node first, followed by the feature nodes. By default, feature notes have a 
# dimensional weight of 1, dimensional weights may be set explicitly through a 
# keyword argument to `links()`.

# The first call to `link()` connects the 'APPLE' chunk node to the red and 
# green color feature nodes and the tasty feature node.

alice.assets.chunks.link( # type: ignore
    chunk("APPLE"), 
    feature("color", "#ff0000"), 
    feature("color", "#008000"),
    feature("tasty", True)
)

# The second call to `link()` connects the 'JUICE' chunk node to the tasty 
# feature node and the liquid state feature node.

alice.assets.chunks.link( # type: ignore
    chunk("JUICE"),
    feature("tasty", True),
    feature("state", "liquid")
)

# The third and last call to `link()` connects the 'FRUIT' chunk node to the 
# sweet and tasty feature nodes.

alice.assets.chunks.link( # type: ignore
    chunk("FRUIT"),
    feature("tasty", True),
    feature("sweet", True)
)

# Agent setup is now complete!


#########################
### Simulation Basics ###
#########################

# To start, the stimulus buffer is set to activate the APPLE chunk. This 
# represents presentation of the concept APPLE. Remember that we assume Alice 
# understands that the cue is APPLE, the fruit, and not e.g., APPLE, the 
# company. 

# Alice performs one NACS cycle. 

alice.propagate(
    kwds={buffer("stimulus"): {"stimulus": {chunk("APPLE"): 1.}}}
)

# To see what came to Alice's mind, we can simply inspect the output state of 
# the NACS at the end of the cycle. 

# To do this, we first retrieve the SubsystemPacket object emitted by the 
# nacs object. This object stores all relevant information about the 
# state of the NACS at the end of the propagation cycle.

# We can then simply print out a nicely formatted representation of the output 
# using the subsystem_packet.pstr() method.
 
print("Alice's cognitive state upon presentation of 'APPLE':") 
pprint.pprint(alice.output)

# Finally, we clear the output so as not to contaminate any subsequent trials 
# with persistent activations. This is an optional step, taken here for 
# demonstration purposes. Its use in practice will depend on the simulation 
# requirements.

alice.clear_outputs()


##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - The mechanics of pyClarion agent construction, and
#   - The basics of running simulations using pyClarion 

# Some functionalities that are supported but not demonstrated include:
#   - Dynamic modification of agent components,
#   - Learning and state/parameter updates,
#   - Deep customization (e.g., custom emitters, construct symbols etc.).
