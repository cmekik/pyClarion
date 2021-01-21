"""
A simple pyClarion demo.

This demo simulates an agent, named Alice, doing a free association task, where 
the goal is to report the first thing that comes to mind upon presentation of 
a cue.
"""


# Import notes may be skipped on first reading. They are for clarification 
# purposes only.
from pyClarion import (
    # Realizer objects, implementing the behavior of simulated constructs.
    Structure, Construct,
    # Constructors for construct symbols, which are used to name, index and 
    # reference simulated constructs
    agent, subsystem, buffer, feature, chunk, rule, terminus, flow_tt, flow_tb, 
    flow_bt, chunks, features,
    # Chunk and rule databases
    Chunks, Rules,
    # Container for shared datastructures like chunk and rule containers.
    Assets, 
    # Realizer processes
    Stimulus, AssociativeRules, BottomUp, TopDown, BoltzmannSelector, MaxNodes, 
    Filtered,
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
# objects instantiating theoretical constructs. The assembly process is 
# declarative and mostly automated. There are broadly two main types of 
# construct: structures, which may contain other constructs, and basic 
# constructs (or 'constructs' for short), which may not contain other 
# constructs. Structures and constructs may be viewed as nodes in a 
# hyper-graphical structure describing the input/output relations among 
# architectural modules.

# Defining Initial Features and Chunks

# An initial step in constructing a pyClarion simulation is to define the 
# primitive representations, as well as any initial knowledge available to 
# agent(s). 

# The primitive representational constructs of Clarion are chunk and feature 
# nodes. We must minimally specify what features will appear in the simulation, 
# as features define the representational domain over which chunks and any 
# other knowledge may be constructed.

# Feature nodes represent implicit knowledge about the world. In Clarion theory, 
# each feature node is associated with a unique dimension-value pair (dv pair) 
# indicating its dimension (e.g., color) and value (e.g., red). In pyClarion, 
# feature dimensions are further analyzed as consisting of a (tag, lag) pair. 
# The tag simply represents the name of the dimension. The lag value is handy 
# for tracking the activation of a particular feature over small time windows, 
# as may be required in, e.g., temporal difference learning. 

# In pyClarion, constructs are named using 'construct symbols'. As the name 
# suggests, construct symbols are intended to behave like formal tokens, and 
# their primary function is to help associate data with the constructs they 
# name. As a result, they are required to be immutable and hashable. It may be 
# helpful to think of construct symbols as fancy python tuples.

# We can invoke the construct symbol for a particular feature node by calling 
# the `feature()` constructor as shown below. 

f = feature(tag="my-tag", val="val-1", lag=0)

# The lag value is optional and defaults to 0.

assert f == feature(tag="my-tag", val="val-1") # does not fail

# For this simulation, we include (somewhat arbitrarily) feature nodes for the 
# colors red and green and a feature for each of tastiness, sweetness and the 
# liquid state. These dv pairs are specified below. We omit lag values from the 
# specification.

# Note that, in some cases, we do not provide feature values. This is sometimes 
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
# that features in Clarion theory represent implicit knowledge. (It is better, 
# in practice, to label features in a way that is intelligible to readers.)

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
# required. Only in more complex simulations, where constructs can pass around 
# commands for example, explicit specification of at least parts of the feature 
# domain becomes a necessity.

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

# We can add rules to the rule database using the `define()` method of the rule 
# database. The argument signature for `define()` is a rule symbol, followed by 
# its conclusion chunk and then by one or more condition chunks. Thus, below, 
# `chunk("FRUIT")` is the conclusion and `chunk("APPLE")` is the only condition. 
# In other words, this rule establishes an association from the concept APPLE 
# to the concept FRUIT. This association is meant to capture the knowledge that 
# "apples are fruits". In truth, we may also designate condition weights, but 
# this feature is not explored here.

rdb.define(rule(1), chunk("FRUIT"), chunk("APPLE"))

# We proceed in much the same way to link chunk and feature nodes in order to 
# define chunks. 

# The chunk database has a `define()` method, which can be used to link a chunk 
# node to feature nodes, creating a fully-formed chunk. The call signature 
# expects the chunk node first, followed by the feature nodes. By default, 
# feature notes have a dimensional weight of 1, dimensional weights may be set 
# explicitly through a keyword argument to `define()`.

# The first call to `define()` connects the 'APPLE' chunk node to the red and 
# green color feature nodes and the tasty feature node. 

cdb.define( 
    chunk("APPLE"), 
    feature("color", "#ff0000"), 
    feature("color", "#008000"),
    feature("tasty")
)

# The second call to `define()` connects the 'JUICE' chunk node to the tasty 
# feature node and the liquid state feature node.

cdb.define(
    chunk("JUICE"),
    feature("tasty"),
    feature("state", "liquid")
)

# The third and last call to `define()` connects the 'FRUIT' chunk node to the 
# sweet and tasty feature nodes.

cdb.define(
    chunk("FRUIT"),
    feature("tasty"),
    feature("sweet")
)

# In models with lots of pre-built knowledge, it may be helpful to express 
# chunk and rule definitions more compactly. This can easily be done, as 
# rdb.define() and cdb.define() both return the initial symbol that they are 
# passed. So rule(1), which was defined earlier, can equivalently and compactly 
# be defined as follows. 

rdb.define(
    rule(1), 
    cdb.define(
        chunk("JUICE"),
        feature("tasty"),
        feature("state", "liquid")
    ),
    cdb.define( 
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
    name=agent("alice")
)

# The `name` argument to the Structure constructor is a construct symbol that 
# serves to label the construct. It is mandatory to provide a name argument to 
# construct realizers, as names enable automation of important behavior, such 
# as linking/unlinking constructs.  

# In this particular model, this is all that is needed to initially set up the 
# agent. The next step is to populate `alice` with components representing 
# various cognitive structures and processes.

# To facilitate the construction process, pyClarion borrows a pattern from 
# the nengo library. When a pyClarion construct is initialized in a `with` 
# statement where the context manager is a pyClarion `Structure`, the construct 
# is automatically added to the structure serving as the context manager. 
# Nested use of the with statement is supported. 

# The order in which constructs are defined within the scope of a with 
# statement roughly determines the order in which the constructs are called 
# when the simulation is stepped. The order is rough because constructs of 
# similar type may be stepped in parallel (though the current implementation is 
# fully sequential).

with alice:

    # For this simulation, there are two main constructs at the agent-level: 
    # the stimulus and the non-action-centered subsystem (NACS). The stimulus 
    # is uncomplicated: it is simply an abstract representation of the task 
    # cue. The NACS, on the other hand, is the Clarion subsystem that is 
    # responsible for processing non-procedural knowledge.

    # Stimulus

    # We begin by adding the stimulus component to the model. 

    # We represent the stimulus with a buffer construct, which is a top-level 
    # construct within an agent that may temporarily store data and relays 
    # activations to various subsystems. Buffers count as constructs, so we 
    # invoke the `Construct` class (as opposed to the `Structure` class as 
    # above). Aside from that, the initialization is similar to the way we 
    # created the `alice` object.

    stimulus = Construct(
        name=buffer("stimulus"), 
        process=Stimulus()
    )

    # The `process` argument defines how the structure computes its outputs. It 
    # is only available in Construct objects. 

    # Non-Action-Centered Subsystem

    # Next, we set up a realizer for the Non-Action-Centered Subsystem. The 
    # setup is similar, but we create a Structure object because subsystems may 
    # contain other constructs.  

    nacs = Structure(
        name=subsystem("nacs"),
        assets=Assets(
            cdb=cdb,
            rdb=rdb
        )
    )

    # To keep track Alice's non-action-centered explicit knowledge, we use the 
    # chunk and rule databases we defined earlier. We store these databases in 
    # an `Assets` object. The `assets` attribute simply provides a namespace for
    # convenient storage of resources shared by construct realizers subordinate 
    # to a `Structure`. The `Assets` object itself is uncomplicated: It simply 
    # records all arguments passed to it as attributes.

    # In reality, the rule database will only be used by a single construct 
    # realizer. However, it helps to keep a reference to it at the level of 
    # NACS as other objects or processes in more advanced models, such as 
    # learning rules, base-level activation trackers, loggers etc., may need 
    # access to the rule database.

    # There is no hard and fast rule about where in the `Structure` hierarchy 
    # a shared resource should be placed.

    # Now, it is time to populate the NACS. 

    with nacs:

        # The entry point for activations in the NACS are chunks, so we begin 
        # by setting up an input chunk pool.

        # In the standard implementation of pyClarion, individual chunk and 
        # feature nodes are not explicitly represented in the system. Instead, 
        # chunk and feature pools are used. These pools handle computing the 
        # strengths of chunk and feature nodes in bulk.

        # This design offers a number of advantages: it is flexible (we do not 
        # need to explicitly declare new chunk or feature nodes to the system), 
        # efficient (all chunk and feature node activations are computed by one 
        # single object in one pass), simple (it reduces bookeeping requirements
        # when adding and removing nodes) and more intelligible (nodes do not 
        # cause clutter in large models). 

        # One downside to this approach is that we have to be careful about 
        # tracking the feature domain. This is why it is good to define the 
        # (initial) feature domain explicitly prior to agent assembly. 

        Construct(
            name=chunks("in"),
            process=MaxNodes(sources=[buffer("stimulus")])
        )

        # Next up is a top-down activation flow, where activations flow from 
        # chunk nodes to linked feature nodes.

        Construct(
            name=flow_tb("main"), 
            process=TopDown(
                source=chunks("in"),
                chunks=nacs.assets.cdb
            ) 
        )

        # In this simulation, because there are no bottom-level flows (i.e., 
        # associative neural networks), we can get away with a single feature 
        # pool. 

        Construct(
            name=features("main"),
            process=MaxNodes(sources=[flow_tb("main")])
        )

        # At the top level, activations are propagated among chunks through an 
        # associative rule flow. Furthermore, a bottom up flow also produces 
        # activation recommendations for chunks based on feature strengths in 
        # the bottom level.

        Construct(
            name=flow_tt("associations"),
            process=AssociativeRules(
                source=chunks("in"),
                rules=nacs.assets.rdb
            ) 
        )

        Construct(
            name=flow_bt("main"), 
            process=BottomUp(
                source=features("main"),
                chunks=nacs.assets.cdb
            ) 
        )

        # The main motivation for having chunk and feature pools is to combine 
        # activation recommendations from multiple flows. In this case, the 
        # output chunk pool takes a straight maximum, but it is common for 
        # inputs from various sources to also be weighted.

        Construct(
            name=chunks("out"),
            process=MaxNodes(
                sources=[
                    chunks("in"),
                    flow_bt("main"), 
                    flow_tt("associations")
                ]
            )
        )
        
        # Finally, a chunk is retrived by an activation-driven competitive 
        # selection process in a terminus construct.

        Construct(
            name=terminus("main"),
            process=Filtered(
                base=BoltzmannSelector(
                    source=chunks("out"),
                    temperature=.1
                ),
                controller=buffer("stimulus")
            )
        )

        # The output selection process in this example involves the 
        # construction of a Boltzmann distribution from chunk node activations. 
        # On each activation cycle, a chunk is sampled from this distribution 
        # and emitted as the selected output.

        # To prevent information in the stimulus from interfering with output 
        # selection, the `BoltzmanSelector` process is wrapped by a `Filtered` 
        # object. This object is configured to filter inputs to the selector 
        # proportionally to their strengths in the stimulus buffer. This is one 
        # way to achieve cue-suppression.

# We are now done populating Alice with constructs. On exit from the 
# highest-level Structure context, pyClarion will automatically populate the 
# agent, link the various constructs, and finalize the agent assembly. 
# Finalization involves setting structure outputs to reflect their contents and 
# checking for any missing expected links. If missing expected links are 
# encountered an error will be thrown.


#########################
### Simulation Basics ###
#########################

# To start off the simulation, the stimulus buffer is set to activate the APPLE 
# chunk. This represents presentation of the concept APPLE. Remember that we 
# assume Alice understands that the cue is APPLE, the fruit, and not e.g., 
# APPLE, the company. 

# Alice performs one simulation step. 

stimulus.process.input({chunk("APPLE"): 1.})
alice.step()

# To see what came to Alice's mind, we can simply inspect the output state of 
# the agent at the end of the cycle. 

# To do this, we simply access the agents `output` attribute. We can then simply
# print out a nicely formatted representation of the output using the pprint() 
# function, which is a version of the stdlib pprint function augmented to handle 
# various pyClarion objects.
 
print("Alice's cognitive state upon presentation of 'APPLE':") 
pprint(alice.output)


##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - The mechanics of pyClarion agent construction, and
#   - The basics of running simulations using pyClarion 
