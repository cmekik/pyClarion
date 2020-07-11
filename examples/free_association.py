"""
A simple pyClarion demo.

This demo simulates an agent, named Alice, doing a free association task, where 
the goal is to report the first thing that comes to mind upon presentation of 
a cue.
"""

# Import notes may be skipped on first reading. They are for clarification 
# purposes only.
from pyClarion import (
    # These are realizer objects, implementing behavior of simulated constructs.
    Structure, Construct,
    # Construct types are used in controlling construct behavior
    ConstructType, MatchSpec, Assets,
    # These functions are constructors for construct symbols, which are used to 
    # name, index and reference simulated constructs
    agent, subsystem, buffer, feature, chunk, response, flow_tt, flow_tb, flow_bt,
    # These objects house datastructures handling various important concerns.
    Chunks, Rules,
    # These objects define how realizers process activations in the forward 
    # direction.
    Stimulus, AssociativeRules, BottomUp, TopDown, BoltzmannSelector, MaxNode, 
    FilteredR, NACSCycle, AgentCycle
)


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
    name=agent("Alice"),
    cycle=AgentCycle(),
    assets=Assets(chunks=Chunks())
)

# The `name` argument to the Structure constructor serves to label the 
# construct. It is mandatory to provide a name argument to construct realizers, 
# as names enable automation of important behavior, such as linking/unlinking 
# constructs.  

# To keep track of concepts that Alice knows about, we equip Alice with a chunk 
# database (more on chunks below). This is done by passing an `Assets` object, 
# which is given a chunk database to be stored in its `chunks` attribute, as 
# the agent's 'assets' attribute. The `assets` attribute provides a namespace 
# for convenient storage of resources shared by construct realizers subordinate 
# to Alice. All Structure objects have the `assets` attribute. The `Assets` 
# object is uncomplicated. It simply records all arguments passed to 
# it as an attribute. 

# A good rule of thumb is to place shared resources in the structure directly 
# in or above the highest-level construct realizer using the resource.

# For this simulation, there are two main constructs at the agent-level: the 
# stimulus and the non-action-centered subsystem (NACS). The stimulus is 
# uncomplicated: it is simply an abstract representation of the task cue. The 
# NACS, on the other hand, is the Clarion subsystem that is responsible for 
# processing declarative knowledge. Knowledge is represented by chunk and 
# feature nodes within the NACS. These nodes receive activations from external 
# buffers and each other, and they compete to be selected as the output of NACS 
# at each simulation step.

# Stimulus

# We begin by adding the stimulus component to the model. 

stimulus = Construct(name=buffer("Stimulus"), propagator=Stimulus())
alice.add(stimulus)

# We represent the stimulus with a buffer construct, which is a top-level 
# construct within an agent that stores and relays activations to various 
# subsystems. As before, we first create a construct realizer. We then pass it 
# to `alice` using `alice.add()`. This automatically stores the buffer within 
# the `alice` object.
 
# We provide the Buffer constructor with a `propagator` object. The propagator 
# object is a callable that defines how the buffer processes information. The 
# Stimulus object passed in as proc allows us to set stimulus values on the fly, 
# as demonstrated below. 

# Non-Action Centered Subsystem

# Next, we set up a realizer for the Non-Action-Centered Subsystem. The setup is 
# syntactically similar, but differs a bit in its semantics. The propagator is,
# this time, a callable that implements the desired activation cycle for NACS. 

nacs = Structure(
    name=subsystem("NACS"),
    cycle=NACSCycle(matches={buffer("Stimulus")}),
    assets=Assets(rules=Rules())
)

# The 'matches' argument lets the subsystem know that it should receive input 
# from the stimulus (more specifically, from the buffer construct representing 
# the stimulus).

# In this example, we see a construct symbol explicitly invoked for the first 
# time through the convenience function `buffer()`, which takes a hashable 
# object and returns a construct symbol for a buffer construct. Typically, 
# construct symbols are created through convenience functions, as manually 
# specifying them is rather tedious. For instance, the result of 
# `buffer("stimulus")` can be directly created with the longer expression 
# `ConstructSymbol("buffer", "Stimulus")`, or the abbreviated but still verbose 
# ConSymb("buffer", "Stimulus").

# As before, we add assets. In this case, we are interested in equipping the 
# NACS with a rule database, to be used in reasoning. In reality, this database 
# is only used by a single construct realizer. However, it helps to keep a 
# reference to it at the level of NACS as other objects/processes, such as 
# learning rules, base-level activation trackers, loggers etc., may need access 
# to the rule database.

alice.add(nacs)

# A call to alice.add() places the NACS within Alice's cognitive apparatus and 
# establishes all specified links (i.e., the link from stimulus buffer to NACS). 
# To do this, the agent object automatically checks all constructs it contains 
# and links up those that match.

# Now, it is time to populate the NACS. This involves adding in chunk and 
# feature nodes to represent any initial knowledge we think `alice` should have 
# and adding in various other components for processing given information and 
# selecting responses. 

# Flows

# Flows are an abstraction native to `pyClarion`; they represent processes 
# within subsystems that map node activations to node activations. In other 
# words, 'flow' is pyClarion's umbrella term for the various neural networks and 
# rule systems that may live within a Clarion subsystem. For example, a 
# collection of associative rules in the top level of the NACS or some neural 
# network module in the bottom level would each be represented by a 
# corresponding `Flow` object.

# For this simulation, we will create three flows. The first processes (in the 
# top level) associative rules known to Alice, the other two links Alice's 
# explicit (top-level) and implicit (bottom-level) declarative knowledge.

# We instantiate flows by constructing `Flow` objects.

nacs.add(
    Construct(
        name=flow_tt("Associations"),
        propagator=AssociativeRules(rules=nacs.assets.rules) 
    ),
    Construct(
        name=flow_bt("Main"), 
        propagator=BottomUp(chunks=alice.assets.chunks) 
    ),
    Construct(
        name=flow_tb("Main"), 
        propagator=TopDown(chunks=alice.assets.chunks) 
    )
)

# Note that the syntax for initializing Flows is essentially the same as before, 
# only requiring appropriate argument choices.

# Responses

# It is necessary to specify how the NACS should choose its output at the end of 
# an activation cycle. The construct realizer for this job is given by the 
# `Response` class. The interface here is essentially the same, but an 
# additional `effector` argument may be used to directly map responses to 
# callbacks, enabling `pyClarion` agents to automatically execute actions when 
# necessary. This feature is not used in the present simulation.

nacs.add(
    Construct(
        name=response("Main"),
        propagator=FilteredR(
            base=BoltzmannSelector(
                temperature=.1,
                matches=MatchSpec(ctype=ConstructType.chunk)
            ),
            input_filter=buffer("Stimulus")
        )
    )
)

# The response selection procedure in this example involves the construction of 
# a Boltzmann distribution from chunk node activations. A chunk is then 
# sampled from this distribution and passed on as the selected response.

# Furthermore, to prevent information in the stimulus from interfering with 
# response selection, the `BoltzmanSelector` is wrapped in a `FilteredD` object. 
# This object is set to filter inputs to the selector proportionally to their 
# strengths in the stimulus buffer. This amounts to cue-suppression.

# In this case, the response construct is identified with the construct symbol 
# `response("Main")`, which may seem a little redundant. However, in some 
# cases (e.g., a complex ACS), a single subsystem may contain several response 
# constructs. In such cases, we add several Response realizers, ideally each 
# with a helpful name.

# Nodes

# We must add feature nodes for all features that we assume Alice may perceive 
# for the purposes of the simulation. Feature nodes represent Alice's implicit 
# knowledge about the world.

# Each feature node is associated with a unique dimension-value pair indicating 
# its dimension (e.g., color) and value (e.g., red). For this simulation, we 
# include (somewhat arbitrarily) feature nodes for the colors red and green and 
# a feature for each of tastiness, sweetness and the liquid state. 

# In the lines below, a list comprehension is used to construct the five 
# feature nodes in the simulation. Once the feature list is constructed, we 
# simply pass it to alice.add() to complete feature addition.

fnodes = [
    Construct(
        name=feature(dim, val), 
        propagator=MaxNode(
            matches=MatchSpec(ctype=ConstructType.flow_xb)
        )
    ) for dim, val in [
        ("color", "#ff0000"), # red
        ("color", "#008000"), # green
        ("tasty", True),
        ("state", "liquid"),
        ("sweet", True)
    ]
]

nacs.add(*fnodes)

# The `matches` argument specifies that the features should take inputs from 
# bottom-level flows and flows linking the top level to the bottom level (i.e., 
# flows ending at the bottom level, or flow_xb). The propagator `MaxNode()` 
# simply outputs the maximum activation value recommended for the client 
# construct by linked flows. 

# Feature values for red and green are given in hex code to emphasize the idea 
# that features in Clarion theory represent implicit knowledge (since most 
# people don't know the meaning of hex color codes off the top of their head). 
# In practice, it is better to label features in a way that is intelligible to 
# readers.

# Next, we initialize the chunk nodes. These correspond roughly to concepts 
# known to Alice.

# Chunk nodes are simpler to identify than feature nodes in that they are 
# differentiated only by name. We will represent chunk nodes for the concepts 
# FRUIT, APPLE, and JUICE.

# As with feature nodes, we construct the chunk nodes with a list comprehension 
# and pass them to `alice.add()`.

cnodes = [
    Construct(
        name=chunk(name),
        propagator=MaxNode(
            matches=MatchSpec(
                ctype=ConstructType.flow_xt | ConstructType.buffer
            )
        )
    ) for name in ["FRUIT", "APPLE", "JUICE"]
]

nacs.add(*cnodes)

# This time, the `matches` argument takes a compound ConstructType value. This 
# is because in the NACS, chunk nodes receive input from incoming flows 
# (i.e., bottom-up and top-level flows) as well as the stimulus, which we 
# represent as a buffer construct. In other words, information coming into the 
# NACS from the stimulus buffer activates the top level first.

# Linking Up Nodes within NACS

# For Alice to produce associations between concepts, we must establish some 
# direct and indirect links among the concepts within her mind. We do this by 
# providing some initial knowledge to NACS flows in the form of some associative 
# rules and some links between top-level chunk nodes and bottom-level feature 
# nodes.

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
    args={
        buffer("Stimulus"): {"stimulus": {chunk("APPLE"): 1.}}
    }
)

# To see what came to Alice's mind, we can simply inspect the output state of 
# the NACS at the end of the cycle. 

# To do this, we first retrieve the SubsystemPacket object emitted by the 
# nacs object. This object stores all relevant information about the 
# state of the NACS at the end of the propagation cycle.

subsystem_packet = nacs.output

# We can then simply print out a nicely formatted representation of the output 
# using the subsystem_packet.pstr() method.
 
print("Information returned by Alice's NACS upon presentation of 'APPLE':") 
print(subsystem_packet.pstr())

# Finally, we clear the output so as not to contaminate any subsequent trials 
# with persistent activations. This is an optional step, taken here for 
# demonstration purposes. Its use in practice will depend on the simulation 
# requirements.

alice.clear_output()

##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - The mechanics of pyClarion agent construction, and
#   - The basics of running simulations using pyClarion 

# Some functionalities that are supported but not demonstrated include:
#   - Streamlined initialization of standard constructs (like NACS),  
#   - Executing action callbacks through subsystems responses,
#   - Dynamic modification of agent components,
#   - Learning and state/parameter updates,
#   - Deep customization (e.g., custom propagators, construct symbols etc.).
