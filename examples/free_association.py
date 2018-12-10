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

# Constructs

# Construct symbols are used to name, identify, and refer to various Clarion 
# constructs used within a simulation.

# We begin by defining variables for construct symbols that will be used in this 
# simulation.

# Strictly speaking, this step is not mandatory. Construct symbols really behave 
# like symbols, so there is no real need to store them in variables; it may only 
# save us some typing.

fruit_ck = Chunk("FRUIT")
apple_ck = Chunk("APPLE")
juice_ck = Chunk("JUICE")

# Color microfeature values (below) are hex-codes to emphasize implicitness.
# In practice, it would be better to give constructs easily intelligible names. 

red_mf = Microfeature(dim="color", val="#ff0000")
green_mf = Microfeature("color", "#008000")
tasty_mf = Microfeature("tasty", True)
liquid_mf = Microfeature("state", "liquid")
sweet_mf = Microfeature("sweet", True)

associative_rules_flow = Flow("Associative Rules", ftype=FlowType.TT) 
top_down_flow = Flow("NACS", ftype=FlowType.TB)
bottom_up_flow = Flow("NACS", ftype=FlowType.BT)

response_nacs = Response("NACS", itype=ConstructType.Chunk)
behavior_nacs = Behavior("NACS", response=response_nacs)

nacs = Subsystem("NACS")
stim_buf = Buffer("Stimulus", outputs=(Subsystem("NACS"),))


# Default node strength

# In general, default strengths are not required. But they are useful.

def default_strength(node = None):
    """Default node strength."""
    
    return 0.0


# Agent assembly

# Agent Architecture

# make_agent constructs an agent realizer from an agent specification.
# the output is not ready to be run, since its components' behaviors are not yet 
# defined

alice = make_agent(
    csym = Agent('Alice'),
    subsystems = {
        nacs: {
            fruit_ck, apple_ck, juice_ck,
            red_mf, green_mf, tasty_mf, liquid_mf, sweet_mf,
            associative_rules_flow, top_down_flow, bottom_up_flow,
            response_nacs, behavior_nacs
        }
    },
    buffers = {
        stim_buf
    }
)

# Behavioral Specifications

# To specify the behavior of alice, we populate her construct realizers with 
# appropriate components. These components may be swapped or modified during the 
# course of a simulation. alice.ready() will let us know if we have forgotten 
# to define any components.

# We begin by setting up the stimulus buffer. ConstantSource simply stores an 
# activation pattern and outputs it every propagation cycle. The stored pattern 
# may be modified over the course of a simulation.

alice[stim_buf].source = ConstantSource()

# Next, we define the propagation cycle for alice's NACS. The function 
# nacs_propagation_cycle() will make sure that all necessary propagation steps 
# are executed.

alice[nacs].propagation_rule = nacs_propagation_cycle

# Now we define how activations may flow within alice's NACS.

# NACS associative rules govern how activations may propagate among chunks in 
# NACS. assoc is a dict whose keys represent conclusion chunks, its 
# values are lists of condition-weight mappings, There is one mapping for each 
# individual rule about a particular conclusion chunk.

alice[nacs, associative_rules_flow].junction = SimpleJunction()
alice[nacs, associative_rules_flow].channel = AssociativeRuleCollection(
    assoc={fruit_ck: [{apple_ck: 1.}]},
    default_strength=default_strength
)

# Next, we define interlevel flows. These control how chunks may activate 
# microfeatures and microfeatures may activate chunks. 
# Here, assoc keys are chunks, which are each linked to zero or more 
# microfeatures. Linked microfeatures are grouped by dimension, as inter-level 
# activation propagation applies a dimension-dependent weight to strengths.

alice[nacs, top_down_flow].junction = SimpleJunction()
alice[nacs, top_down_flow].channel = TopDownLinks(
    assoc={
        apple_ck: {
            "color": (1., set([red_mf, green_mf])),
            "tasty": (1., set([tasty_mf]))
        },
        juice_ck: {
            "tasty": (1., set([tasty_mf])),
            "state": (1., set([liquid_mf]))
        },
        fruit_ck: {
            "tasty": (1., set([tasty_mf])),
            "sweet": (1., set([sweet_mf]))
        }
    },
    default_strength=default_strength
)

alice[nacs, bottom_up_flow].junction = SimpleJunction()
alice[nacs, bottom_up_flow].channel = BottomUpLinks(
    assoc=alice[nacs][top_down_flow].channel.assoc,
    default_strength=default_strength
)

# Now we define the behavior of individual nodes. Node junctions determine the 
# final output of a node given recommendations from various afferent flows.

for node, realizer in alice[nacs].items_ctype(ConstructType.Node):
    realizer.junction = SimpleNodeJunction(node, default_strength)

# To determine how responses are selected, we specify a selector for the 
# response construct. The junction serves to aggregate activations from afferent 
# nodes.

alice[nacs, response_nacs].junction = SimpleJunction()
alice[nacs, response_nacs].selector = SimpleBoltzmannSelector(temperature=.1)

# Effectors link selected chunks to necessary action callbacks. In addition to 
# behavioral responses, action callbacks may implement non-external actions, 
# such as goal-setting, attention allocation, working-memory allocation etc., 
# depending on simulation requirements.

alice[nacs, behavior_nacs].effector = MappingEffector(
    chunk2callback={
        apple_ck: lambda: recorder.actions.append(apple_ck),
        juice_ck: lambda: recorder.actions.append(juice_ck),
        fruit_ck: lambda: recorder.actions.append(fruit_ck)
    }
)

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
    for c, r in nacs.items_ctype(ConstructType.Node):
        strength = round(r.output.view().strengths[c], digits)
        print("   ", c, strength)

    print(" ", "Response:")
    for _, realizer in nacs.items_ctype(ConstructType.Response):
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
alice[stim_buf].source.update({apple_ck: 1.})

# Alice performs one NACS cycle. 
alice.propagate()

# Alice responds.
alice.execute()

# Here is Alice's cognitive state at the end of the trial.
summarize_nacs_cycle(alice[nacs], recorder, 'Initial Trial')


#######################
### Cue Suppression ###
#######################

# A limitation of this model: in free association, subjects do not 
# return the cue as their response. But, Alice does return the cue because cue 
# suppression requires input/output filtering (i.e., selective attention), 
# which is a capability not included in the current model. Let's add it. 

# First, we'll define a component capable of carrying out the necessary 
# filtering.

class ResponseFilterJunction(SimpleJunction):

    def __init__(self, filter_dict = None):

        self.fdict = filter_dict or {}

    def __call__(self, packets):

        d = super().__call__(packets)
        for node, factor in self.fdict.items():
            if node in d: d[node] *= factor
        return d

# We need to replace the existing junction for nacs appraisals with a new one.
# This is very easy: just assign to buffer.junction!
alice[nacs][response_nacs].junction = ResponseFilterJunction()

# We assume that alice filters out the cue. The decision to do this and its 
# execution are not the responsibility of NACS, so they are not explicitly 
# simulated.
alice[nacs][response_nacs].junction.fdict[apple_ck] = .0

# Now we run through the trial again (previous activations persist).
alice.propagate()
alice.execute()

# Here is Alice's cognitive state at the end of the trial.
summarize_nacs_cycle(alice[nacs], recorder, 'With Cue Suppression')


################
### Learning ###
################

# Learning is an essential part of Clarion, so a simple example below 
# demonstrates learning in pyClarion. Note that we will again modify Alice on 
# the fly to enable learning. 

# First, we must define a learning routine. This will do:

class HeavyHandedLearningRoutine(object):
    # This routine simply injects some preset knowledge into the NACS.

    def __init__(
        self, recorder, nacs, toplevel_assoc, interlevel_assoc
    ):

        self.recorder = recorder
        self.nacs = nacs
        self.toplevel_assoc = toplevel_assoc
        self.interlevel_assoc = interlevel_assoc
    
    def __call__(self) -> None:

        fruit_ck = Chunk("FRUIT")
        orange_ck = Chunk("ORANGE")
        orange_color_mf = Microfeature("color", "#ffa500")
        tasty_mf = Microfeature("tasty", True)
        behavior = Behavior("NACS", Response("NACS", ConstructType.Chunk))

        self.nacs[orange_ck] = NodeRealizer(
            orange_ck, SimpleNodeJunction(orange_ck, default_strength)
        )
        self.nacs[orange_color_mf] = NodeRealizer(
            orange_color_mf, 
            SimpleNodeJunction(orange_color_mf, default_strength)
        )
        self.toplevel_assoc[fruit_ck].append({orange_ck: 1.})
        self.interlevel_assoc[orange_ck] = {
                "color": (1., set([orange_color_mf])),
                "tasty": (1., set([tasty_mf]))
            }
        self.nacs[behavior].effector.chunk2callback[orange_ck] = (
            lambda: self.recorder.actions.append(orange_ck)
        )

# Once we have a routine, we simply attach it to Alice. That's it!
alice.attach(
    HeavyHandedLearningRoutine(
        recorder, 
        alice[nacs], 
        alice[nacs, associative_rules_flow].channel.assoc, 
        alice[nacs, top_down_flow].channel.assoc
    )
)

# The code below repeats the same task, but only after Alice suddenly and 
# inexplicably learns about oranges!

# Here is the learning part.
alice.learn()

# Now we run through the trial again (previous activations persist).
alice.propagate()
alice.execute()

# Here is Alice's cognitive state at the end of the trial.
summarize_nacs_cycle(alice[nacs], recorder, 'With Learning')
