"""
A simple pyClarion demo.

This demo simulates an agent, named Alice, doing a free association task, where 
the goal is to report the first thing that comes to mind upon presentation of 
a cue.
"""


from pyClarion import *


#############
### Setup ###
#############

### Simulation Environment ###

# Let's keep the environment lean. Just a recorder:

class BehaviorRecorder(object):

    def __init__(self):

        self.actions = []

recorder = BehaviorRecorder()

### Agent Setup ###

# In this section, we'll create an agent named 'Alice' with some knowledge 
# about fruits.

# Default node strength

# In general, default strengths are not required. But they are useful.

def default_strength(node = None):
    """Default node strength."""
    
    return 0.0


# Agent assembly

# Agent Architecture

# The initial structure of an agent is specified using construct symbols, which 
# are basic data structures used to identify and reference different components 
# of a pyClarion agent.

# The function make_agent() constructs an agent realizer from a structured 
# collection of construct symbols. This involves initializing an agent realizer 
# object to represent alice and then initializing and placing a set of construct 
# realizer objects within this agent realizer according to the given 
# specifications.

alice = make_agent(
    csym = agent('Alice'),
    subsystems = {
        subsystem("NACS"): {
            chunk("FRUIT"), 
            chunk("APPLE"), 
            chunk("JUICE"),
            feature("color", "#ff0000"), # red
            feature("color", "#008000"), # green
            feature("tasty", True),
            feature("state", "liquid"),
            feature("sweet", True),
            flow_tt("Assoc"),
            flow_tb(1),
            flow_bt(1),
            response("Output"),
            behavior("Report")
        }
    },
    buffers = {
        buffer("Stimulus")
    }
)

# The output of make_agent() is not ready to be run since its components' 
# behaviors are not yet defined. Only the general structure of the agent has 
# been defined:
#   - The agent is called 'Alice'
#   - Alice has one subsystem called 'NACS' (Non-action-centered subsystem)
#   - NACS is initialized with some chunks and features representing knowledge 
#     about fruits
#   - NACS has three different activation flows implementing associative, 
#     top-down and bottom-up links
#   - NACS outputs one response which then gets reported (to the simulation 
#     environment)
#   - Alice also has one stimulus buffer which feeds into her NACS 

# Behavioral Specifications

# To specify the behavior of alice, we populate her construct realizers with 
# appropriate components. These components may be swapped or modified during the 
# course of a simulation. A call to alice.ready() will let us know if we have 
# forgotten to define any components.

# We begin by setting up the stimulus buffer. ConstantSource simply stores an 
# activation pattern and outputs it every propagation cycle. The stored pattern 
# may be modified over the course of a simulation.

alice[buffer("Stimulus")].source = ConstantSource()

# Next, we work on setting up alice's NACS.

nacs = alice[subsystem("NACS")]

# We first define the propagation cycle for alice's NACS. The function 
# nacs_propagation_cycle() will make sure that all necessary propagation steps 
# are executed.

nacs.pull_rule = ConstructType.buffer
nacs.propagation_rule = nacs_propagation_cycle

# Now we define how activations may flow within alice's NACS.

# Associative rules govern how activations may propagate among chunks in 
# NACS. 

# For this example, the only association that Alice will initially have is one
# taking the concept APPLE to the concept FRUIT. This is implemented as a 
# weighted link between the APPLE and FRUIT chunks. In this particular case, a 
# weight of 1.0 is assigned to the link. 

nacs[flow_tt("Assoc")].pull_rule = ConstructType.chunk
nacs[flow_tt("Assoc")].junction = SimpleJunction()
nacs[flow_tt("Assoc")].channel = AssociativeRuleCollection(
    assoc={chunk("FRUIT"): [{chunk("APPLE"): 1.}]},
    default_strength=default_strength
)


# assoc is a dict whose keys represent conclusion chunks, its 
# values are lists of condition-weight mappings, There is one mapping for each 
# individual rule about a particular conclusion chunk.

# Next, we define interlevel flows. These control how chunks may activate 
# features and features may activate chunks. 

nacs[flow_tb(1)].pull_rule = ConstructType.chunk
nacs[flow_tb(1)].junction = SimpleJunction()
nacs[flow_tb(1)].channel = TopDownLinks(
    assoc={
        chunk("APPLE"): {
            "color": (
                1., {feature("color", "#ff0000"), feature("color", "#008000")}
            ),
            "tasty": (1., {feature("tasty", True)})
        },
        chunk("JUICE"): {
            "tasty": (1., {feature("tasty", True)}),
            "state": (1., {feature("state", "liquid")})
        },
        chunk("FRUIT"): {
            "tasty": (1., {feature("tasty", True)}),
            "sweet": (1., {feature("sweet", True)})
        }
    },
    default_strength=default_strength
)

# Here, assoc keys are chunks, which are each linked to zero or more 
# features. Linked features are grouped by dimension as inter-level 
# activation propagation applies a dimension-dependent weight to strengths.

nacs[flow_bt(1)].pull_rule = ConstructType.feature
nacs[flow_bt(1)].junction = SimpleJunction()
nacs[flow_bt(1)].channel = BottomUpLinks(
    assoc=nacs[flow_tb(1)].channel.assoc,
    default_strength=default_strength
)

# Note that top down and bottom up links in this simulation share the same assoc 
# dict.

# Now we define the behavior of individual nodes (chunks and features). 

for node, realizer in nacs.items_ctype(ConstructType.node):
    if node.ctype == ConstructType.feature:
        ftype = ConstructType.flow_bb | ConstructType.flow_tb
    elif node.ctype == ConstructType.chunk:
        ftype = ConstructType.flow_tt | ConstructType.flow_bt
    realizer.pull_rule = ftype | ConstructType.buffer
    realizer.junction = SimpleNodeJunction(node, default_strength)

# Node junctions determine the final output of a node given recommendations 
# from various afferent flows.

# To determine how responses are selected, we specify a selector for the 
# response construct. 

nacs[response("Output")].pull_rule = ConstructType.chunk
nacs[response("Output")].junction = SimpleJunction()
nacs[response("Output")].selector = BoltzmannSelector(temperature=.1)

# The SimpleJunction attached to the response realizer serves to aggregate 
# activations from afferent nodes.

# Effectors link selected chunks to necessary action callbacks. In addition to 
# behavioral responses, action callbacks may implement non-external actions, 
# such as goal-setting, attention allocation, working-memory allocation etc., 
# depending on simulation requirements.

alice[subsystem("NACS"), behavior("Report")].pull_rule = {response("Output")}
alice[subsystem("NACS"), behavior("Report")].effector = MappingEffector(
    chunk2callback={
        chunk("APPLE"): lambda: recorder.actions.append(chunk("APPLE")),
        chunk("JUICE"): lambda: recorder.actions.append(chunk("JUICE")),
        chunk("FRUIT"): lambda: recorder.actions.append(chunk("FRUIT"))
    }
)

# Make sure everything is linked:

alice.make_links()

# Finally, we check if we've missed anything.

assert alice.ready()

# Done!


#########################
### Simulation Basics ###
#########################

# We are almost ready to start simulating. Let us first give ourselves a tool 
# to summarize the results of an NACS activation cycle.

def summarize_nacs_cycle(nacs, recorder, title, digits=3):

    print(title)

    print(" ", "Strengths:")
    for c, r in nacs.items_ctype(ConstructType.node):
        strength = round(r.output.view().strengths[c], digits)
        print("   ", c, strength)

    print(" ", "Response:")
    for _, realizer in nacs.items_ctype(ConstructType.response):
        for c, s in realizer.output.view().strengths.items():
            print("   ", c, round(s, 3))
    # We could just get selected from the response decision packet, but why 
    # not demonstrate that the agent has affected its envirnoment?
    print("   ", "Selected:", recorder.actions.pop())


# Okay. Let's begin simulation.

# To start, the stimulus buffer is set to provide constant activation to the 
# APPLE chunk. This represents presentation of the concept APPLE. Note that 
# there are simplifications. For instance, it is assumed that Alice is made to 
# understand that the cue is APPLE, the fruit, and not e.g., APPLE, the 
# company. 

alice[buffer("Stimulus")].source.update({chunk("APPLE"): 1.})

# Alice performs one NACS cycle. 

alice.propagate()

# Alice responds.

alice.execute()

# Here is Alice's cognitive state at the end of the trial.

summarize_nacs_cycle(alice[subsystem("NACS")], recorder, 'Initial Trial')

# To finish, we clear Alice's NACS in preparation for the next example.

alice.clear_activations()


#######################
### Cue Suppression ###
#######################

# In free association, subjects do not return the cue as their response. This is 
# called cue suppression. But, Alice may return the cue because cue suppression 
# requires input/output filtering (i.e., selective attention), which is a 
# capability not included in the current model. Let's add it. 

# First, we'll define a component capable of carrying out the necessary 
# filtering.

# We need to replace the existing junction for nacs appraisals with a new one.
# This is very easy: just assign to buffer.junction!

nacs[response("Output")].junction = FilteredSimpleJunction()

# We assume that alice filters out the cue. The decision to do this and its 
# execution are not the responsibility of NACS, so they are not explicitly 
# simulated.

nacs[response("Output")].junction.fdict[chunk("APPLE")] = .0

# Now we run through the trial again.

alice.propagate()
alice.execute()

# Here is Alice's cognitive state at the end of the trial.

summarize_nacs_cycle(alice[subsystem("NACS")], recorder, 'With Cue Suppression')

# Once again, we clear Alice's NACS.

alice.clear_activations()


################
### Learning ###
################

# Learning is an essential part of Clarion, so a simple example below 
# demonstrates learning in pyClarion. Note that we will again modify Alice on 
# the fly to enable learning. 

# First, we must define a learning routine. This will do:

class HeavyHandedLearningRoutine(object):
    """Injects some preset knowledge about oranges into the NACS."""

    def __init__(
        self, recorder, nacs, toplevel_assoc, interlevel_assoc, behavior
    ):

        self.recorder = recorder
        self.nacs = nacs
        self.toplevel_assoc = toplevel_assoc
        self.interlevel_assoc = interlevel_assoc
        self.behavior = behavior

    def __call__(self) -> None:

        # Add orange chunk node and orange color feature to NACS
        self.nacs.insert_realizers(
            NodeRealizer(
                chunk("ORANGE"),
                ConstructType.flow_tt | ConstructType.flow_bt, 
                SimpleNodeJunction(chunk("ORANGE"), default_strength)
            ),
            NodeRealizer(
                feature("color", "#ffa500"), 
                ConstructType.flow_bb | ConstructType.flow_tb,
                SimpleNodeJunction(
                    feature("color", "#ffa500"), default_strength
                )
            )    
        )

        # Add rule associating ORANGE to FRUIT
        self.toplevel_assoc[chunk("FRUIT")].append({chunk("ORANGE"): 1.})
        
        # Link ORANGE with tasty and orange color features
        self.interlevel_assoc[chunk("ORANGE")] = {
                "color": (1., {feature("color", "#ffa500")}),
                "tasty": (1., {feature("tasty", True)})
            }

        # Add ORANGE as possible response
        self.behavior.effector.chunk2callback[chunk("ORANGE")] = (
            lambda: self.recorder.actions.append(chunk("ORANGE"))
        )

# Once we have a routine, we simply attach it to Alice. That's it!

alice.attach(
    HeavyHandedLearningRoutine(
        recorder, 
        alice[subsystem("NACS")], 
        alice[subsystem("NACS"), flow_tt("Assoc")].channel.assoc, 
        alice[subsystem("NACS"), flow_tb(1)].channel.assoc,
        alice[subsystem("NACS"), behavior("Report")]
    )
)

# Also, note that multiindexing construct realizers is a possibility as shown 
# above.

# The code below repeats the same task, but only after Alice suddenly and 
# inexplicably learns about oranges!

# Here is the learning part.

alice.learn()

# Now we run through the trial again.

alice.propagate()
alice.execute()

# Here is Alice's cognitive state at the end of the trial.

summarize_nacs_cycle(alice[subsystem("NACS")], recorder, 'With Learning')

# Finally, we clear Alice's NACS for the sake of completeness.

alice.clear_activations()


##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#
#   - mechanics of pyClarion agent construction
#   - basics of running simulations using pyClarion 
#   - flexibility and customizability of pyClarion agents
#   - basics of learning in pyClarion
