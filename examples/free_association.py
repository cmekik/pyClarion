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
    # Below are functions for constructing construct symbols, which are used to 
    # name, index and reference simulated constructs
    agent, subsystem, buffer, feature, chunk, rule, terminus, flow_tt, flow_tb, 
    flow_bt, chunks, features,
    # The objects below house datastructures handling various important 
    # concerns such as chunk and rule definitions.
    Chunks, Rules,
    # Below is a simple container for shared datastructures like chunk and 
    # rule containers.
    Assets, 
    # The objects below define how realizers process activations in the forward 
    # and backward directions.
    Stimulus, AssociativeRules, BottomUp, TopDown, BoltzmannSelector, MaxNodes, 
    Filtered, NACSCycle, AgentCycle,
    # Finally, pyClarion augments the stdlib pprint functionality to support 
    # its own datatypes.
    pprint
)


#############
### Setup ###
#############

# Let's simulate a subject named 'Alice' with some knowledge about fruits 
# performing a free association task.

### Agent Setup ###

# PyClarion agents are created by assembling construct realizers, which are 
# objects instantiating theoretical constructs. Much of the assembly process is 
# automated, so agent construction amounts to declaratively specifying the 
# necessary constructs. There are broadly two main types of construct: 
# structures, which may contain other constructs, and basic constructs (or 
# 'constructs' for short), which may not contain other constructs. Structures 
# and constructs may be viewed as nodes in a hyper-graphical structure 
# describing the input/output relations among architectural modules.

# Defining Initial Features and Chunks

# An initial step in constructing a pyClarion simulation is to define the 
# primitive representations that will appear in the simulation, as well as any 
# initial knowledge available to agent(s). 

# The primitive representational constructs of Clarion are chunk and feature 
# nodes. At this stage, we must minimally specify what features will appear in 
# the simulation, as features define the representational domain over which 
# chunks and any other knowledge may be constructed.

# Feature nodes represent implicit knowledge about the world. In Clarion theory, 
# each feature node is associated with a unique dimension-value pair (dv pair) 
# indicating its dimension (e.g., color) and value (e.g., red). In pyClarion, we 
# further analyze feature dimensions as consisting of a (tag, lag) pair. The 
# tag simply represents the name of the dimension. The lag value is handy for 
# tracking the activation of a particular feature over small time windows, as 
# may be required in, e.g., temporal difference learning. 

# In pyClarion, constructs are named using 'construct symbols'. As the name 
# suggests, construct symbols are intended to behave like formal tokens, and 
# their primary function is to help associate data with the constructs they 
# name. As a result, they are required to be immutable and hashable (so that 
# they may be used with dict-like structures). It may be helpful to think of 
# construct symbols as fancy python tuples.

# We can invoke the construct symbol for a particular feature node by calling 
# the `feature()` constructor as shown below. 

f = feature(tag="my-tag", val="val-1", lag=0)

# The lag value is optional and defaults to 0.

assert f == feature(tag="my-tag", val="val-1") # does not fail

# For this simulation, we include (somewhat arbitrarily) feature nodes for the 
# colors red and green and a feature for each of tastiness, sweetness and the 
# liquid state. These dv pairs are specified below. We omit lag values from the 
# specification. (We will not make use of lagged features in this simulation, 
# and lagged features may be constructed dynamically as needed.) 

# Note that in some cases, we do not provide feature values. This is sometimes 
# desirable, when we have singleton dimensions. In such cases, the feature 
# constructor automatically sets the value to the empty string.

feature_spec = [
    feature("color", "#ff0000"), # red
    feature("color", "#008000"), # green
    feature("tasty"),
    feature("state", "liquid"),
    feature("sweet")
]

# Feature values for red and green are given in hex code to emphasize the idea 
# that features in Clarion theory represent implicit knowledge. (Of course, it 
# is better, in practice, to label features in a way that is intelligible to 
# readers.)

# Moving on, let us consider chunk nodes. Chunk nodes correspond roughly to 
# the concepts known to Alice. Chunk nodes are simpler to identify than feature 
# nodes in that they are differentiated only by their names, which are taken to 
# be purely formal labels. 

# We can invoke chunk symbols using the `chunk()` constructor as follows.

chunk("Chunk-1")

# We will represent chunk nodes for the concepts FRUIT, APPLE, and JUICE.

chunk_names = ["FRUIT", "APPLE", "JUICE"] 

# In this simulation, we specify the initial chunks and features explicitly 
# only for the sake of clarity. Strictly speaking, these specificaions are not 
# required in this particular case. But, in more complex simulations, where 
# constructs can pass around commands for example, explicit specification of 
# at least parts of the feature domain becomes a necessity.

# Now that we've defined the symbols we will be working with, we populate Alice 
# with some knowledge.

# Setting up initial knowledge

# For Alice to produce associations between concepts, we must establish some 
# direct and indirect links among the concepts within her mind. We do this by 
# providing some initial knowledge in the form of some associative rules and 
# some links between top-level chunk nodes and bottom-level feature nodes.

# To house chunk and rule definitions, we initialize a chunk database and a 
# rule database.

cdb = Chunks()
rdb = Rules()

# We can add rules to the rule database using the `link()` method of the rule 
# database. The argument signature for `link()` is a rule symbol, followed by 
# its conclusion chunk and then by one or more condition chunks. Thus, below, 
# `chunk("FRUIT")` is the conclusion and `chunk("APPLE")` is the only condition. 
# In other words, this rule establishes an association from the concept APPLE 
# to the concept FRUIT. This association is meant to capture the fact that 
# apples are fruits. In truth, we may also designate condition weights, but 
# this feature is not explored here.

rdb.link(rule(1), chunk("FRUIT"), chunk("APPLE"))

# We proceed in much the same way to link chunk and feature nodes in order to 
# define chunks. 

# The chunk database has a `link()` method, which can be used to link a chunk 
# node to feature nodes, creating a fully-formed chunk. The call signature 
# expects the chunk node first, followed by the feature nodes. By default, 
# feature notes have a dimensional weight of 1, dimensional weights may be set 
# explicitly through a keyword argument to `links()`.

# The first call to `link()` connects the 'APPLE' chunk node to the red and 
# green color feature nodes and the tasty feature node. 

cdb.link( 
    chunk("APPLE"), 
    feature("color", "#ff0000"), 
    feature("color", "#008000"),
    feature("tasty")
)

# The second call to `link()` connects the 'JUICE' chunk node to the tasty 
# feature node and the liquid state feature node.

cdb.link(
    chunk("JUICE"),
    feature("tasty"),
    feature("state", "liquid")
)

# The third and last call to `link()` connects the 'FRUIT' chunk node to the 
# sweet and tasty feature nodes.

cdb.link(
    chunk("FRUIT"),
    feature("tasty"),
    feature("sweet")
)

# In models with lots of pre-built knowledge, it may be helpful to express 
# chunk and rule definitions more compactly. This can easily be done, as 
# rdb.link() and cdb.link() both return the initial symbol that they are passed.
# So rule(1), which was defined earlier, can equivalently and compactly be 
# defined as follows. 

rdb.link(
    rule(1), 
    cdb.link(
        chunk("JUICE"),
        feature("tasty"),
        feature("state", "liquid")
    ),
    cdb.link( 
        chunk("APPLE"), 
        feature("color", "#ff0000"), 
        feature("color", "#008000"),
        feature("tasty")
    )
)


# Agent assembly

# We begin by creating the top-level construct: a Structure object representing 
# the agent Alice.

alice = Structure(
    name=agent("alice"),
    emitter=AgentCycle()
)

# The `name` argument to the Structure constructor is a construct symbol that 
# serves to label the construct. It is mandatory to provide a name argument to 
# construct realizers, as names enable automation of important behavior, such 
# as linking/unlinking constructs. 

# The `emitter` argument defines how the structure computes its outputs. In 
# Structure objects, this involves specifying directives for controlling 
# the firing of subordinate constructs. The `AgentCycle()` object defines such 
# directives for agents. 

# The next step in agent construction is to populate `alice` with components 
# representing various cognitive structures postulated by the Clarion theory.

# Constructs may be added to `Structure` objects using the `add()` method, like 
# `alice.add()`. A call to alice.add() on some construct, places that construct 
# within Alice's cognitive apparatus and establishes any links specified 
# between the new construct and any existing consturct within Alice. To do 
# this, the `Structure` object automatically checks all constructs it contains 
# and links up those that match.

# To facilitate the construction process, pyClarion borrows a pattern from 
# the nengo library. When a pyClarion construct is initialized in a `with` 
# statement where the context manager is a pyClarion `Structure`, the construct 
# is automatically added to the structure serving as the context manager. 
# Nested use of the with statement is supported.

with alice:

    # For this simulation, there are two main constructs at the agent-level: 
    # the stimulus and the non-action-centered subsystem (NACS). The stimulus 
    # is uncomplicated: it is simply an abstract representation of the task 
    # cue. The NACS, on the other hand, is the Clarion subsystem that is 
    # responsible for processing non-procedural knowledge. Knowledge is 
    # represented by chunk and feature nodes within the NACS. These nodes 
    # receive activations from external buffers and each other, and they 
    # compete to be selected as the output of NACS at each simulation step.

    # Stimulus

    # We begin by adding the stimulus component to the model. 

    # We represent the stimulus with a buffer construct, which is a top-level 
    # construct within an agent that stores and relays activations to various 
    # subsystems. Buffers count as constructs, so we invoke the `Construct` 
    # class (as opposed to the `Structure` class as above). Aside from that, 
    # the initialization is essentially identical to the way we created the 
    # `alice` object.

    stimulus = Construct(
        name=buffer("stimulus"), 
        emitter=Stimulus()
    )

    # We pass a `buffer()` symbol as name, to indicate that we are defining a 
    # buffer, and we pass a `Stimulus` object as emitter to provide the 
    # necessary input/output methods. That completes initialization of the 
    # stimulus construct.

    # Non-Action-Centered Subsystem

    # Next, we set up a realizer for the Non-Action-Centered Subsystem. The 
    # setup is similar, but we create a Structure object because subsystems may 
    # contain other constructs. The emitter is one that implements the desired 
    # activation cycle for NACS. 

    nacs = Structure(
        name=subsystem("nacs"),
        emitter=NACSCycle(),
        assets=Assets(
            cdb=cdb,
            rdb=rdb
        )
    )

    # The 'sources' argument to the emitter lets the subsystem know that it 
    # should receive input from the stimulus (more specifically, from the 
    # buffer construct representing the stimulus).

    # To keep track of the chunks and rules that Alice knows about, we equip 
    # the NACS with the chunk and rule databases we defined earlier. This is 
    # done by passing an `Assets` object containing the two databases as the 
    # structure's `assets` attribute. The `assets` attribute provides a 
    # namespace for convenient storage of resources shared by construct 
    # realizers subordinate to a `Structure`. The `Assets` object is 
    # uncomplicated: It simply records all arguments passed to it as attributes.

    # In reality, the rule database will only be used by a single construct 
    # realizer. However, it helps to keep a reference to it at the level of 
    # NACS as other objects or processes in more advanced models, such as 
    # learning rules, base-level activation trackers, loggers etc., may need 
    # access to the rule database.

    # There is no hard and fast rule about where in the `Structure` hierarchy 
    # a shared resource should be placed.

    # Now, it is time to populate the NACS. 

    with nacs:

        # Chunk and Feature Pools

        # In the standard implementation of pyClarion, individual chunk and 
        # feature nodes are not explicitly represented in the system. Instead, 
        # chunk and feature pools are used. These pools handle computing the 
        # strengths of chunk and feature nodes in bulk.
         
        # This design offers a number of advantages: it is flexible (we do not 
        # need to register new chunk or feature nodes), efficient (all chunk 
        # and feature node activations are computed by one single object in one 
        # pass), simple (it reduces bookeeping requirements when adding and 
        # removing nodes) and more intelligible (nodes do not cause clutter in 
        # large models). 

        # One downside to this approach is that we have to be careful about 
        # tracking the feature domain. This is why it is good to define the 
        # (initial) feature domain explicitly prior to agent assembly. 

        # Below, two constructs are initialized, one for the feature pool 
        # dubbed 'features("main")' and one for the chunk pool dubbed 
        # 'chunks("main")'. Both of these constructs make use of the `MaxNodes` 
        # emitter, which outputs, for each node, the maximum strength 
        # associated with the respective feature or chunk.

        # In general, emitters are aware of the constructs they serve. Thus, 
        # the MaxNodes emitter will only output activations for chunk nodes 
        # when paired with a `chunks()` construct, and it will likewise only 
        # output activations for feature nodes for a `features()` construct.

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={
                    flow_tb("main")
                }
            )
        )

        # Note that buffer("stimulus") will automatically be linked with 
        # chunks("main") even though it is on a different level of the 
        # hierarchy. This is guaranteed when the nested `with` syntax is used, 
        # otherwise one must take care to construct agents in a bottom-up 
        # fashion.

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

        # Flows are an abstraction representing processes within subsystems 
        # that map node activations to node activations. In other words, 'flow' 
        # is pyClarion's umbrella term for the various neural networks and rule 
        # systems that may live within a Clarion subsystem. For example, a 
        # collection of associative rules in the top level of the NACS or some 
        # neural network module in the bottom level would each be represented 
        # by a corresponding flow construct.

        # For this simulation, we will create three flows. The first processes 
        # (in the top level) associative rules known to Alice, the other two 
        # links Alice's explicit (top-level) and implicit (bottom-level) 
        # declarative knowledge.

        Construct(
            name=flow_tt("associations"),
            emitter=AssociativeRules(
                source=chunks("main"),
                rules=nacs.assets.rdb
            ) 
        )

        Construct(
            name=flow_bt("main"), 
            emitter=BottomUp(
                source=features("main"),
                chunks=nacs.assets.cdb
            ) 
        )

        Construct(
            name=flow_tb("main"), 
            emitter=TopDown(
                source=chunks("main"),
                chunks=nacs.assets.cdb
            ) 
        )

        # Note that the syntax for initializing flows is essentially the same 
        # as for any other Construct, only requiring appropriate argument 
        # choices.

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
        # `terminus("main")`. In more complex simulations, a single subsystem 
        # may contain several terminus nodes.
        
        Construct(
            name=terminus("main"),
            emitter=Filtered(
                base=BoltzmannSelector(
                    source=chunks("main"),
                    temperature=.1
                ),
                sieve=buffer("stimulus")
            )
        )

        # The output selection process in this example involves the 
        # construction of a Boltzmann distribution from chunk node activations. 
        # On each activation cycle, a chunk is sampled from this distribution 
        # and emitted as the selected output.

        # To prevent information in the stimulus from interfering with output 
        # selection, the `BoltzmanSelector` is wrapped in a `FilteredT` object. 
        # This object is configured to filter inputs to the selector 
        # proportionally to their strengths in the stimulus buffer. This is one 
        # way to achieve cue-suppression.

# We are now done populating Alice with constructs. On exit from the 
# highest-level Structure context, pyClarion will automatically finalize the 
# agent assembly. This process involves setting structure outputs to reflect 
# their contents and checking for any missing expected links. If missing 
# expected links are encountered a RealizerAssemblyError will be thrown.

# Agent setup is now complete!


#########################
### Simulation Basics ###
#########################

# To start off the simulation, the stimulus buffer is set to activate the APPLE 
# chunk. This represents presentation of the concept APPLE. Remember that we 
# assume Alice understands that the cue is APPLE, the fruit, and not e.g., 
# APPLE, the company. 

# Alice performs one simulation step. 

stimulus.emitter.input({chunk("APPLE"): 1.})
alice.step()

# To see what came to Alice's mind, we can simply inspect the output state of 
# the NACS at the end of the cycle. 

# To do this, we first retrieve the SubsystemPacket object emitted by the 
# nacs object. This object stores all relevant information about the 
# state of the NACS at the end of the propagation cycle.

# We can then simply print out a nicely formatted representation of the output 
# using the subsystem_packet.pstr() method.
 
print("Alice's cognitive state upon presentation of 'APPLE':") 
pprint(alice.output)


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
