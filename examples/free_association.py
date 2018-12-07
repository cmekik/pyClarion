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

fruit_ck = Chunk("FRUIT")
apple_ck = Chunk("APPLE")
juice_ck = Chunk("JUICE")

red_mf = Microfeature(dim="color", val="#ff0000")
green_mf = Microfeature("color", "#008000")
tasty_mf = Microfeature("tasty", True)
liquid_mf = Microfeature("state", "liquid")
sweet_mf = Microfeature("sweet", True)

associative_rules_flow = Flow("Associative Rules", ftype=FlowType.TT) 
top_down_flow = Flow("NACS", ftype=FlowType.TB)
bottom_up_flow = Flow("NACS", ftype=FlowType.BT)

appraisal_nacs = Appraisal("NACS", itype=ConstructType.Chunk)
behavior_nacs = Behavior("Respond", appraisal=appraisal_nacs)

nacs = Subsystem("NACS")
stim_buf = Buffer("Stimulus", outputs=(Subsystem("NACS"),))


# Alices's initial top-level knowledge

toplevel_assoc = {fruit_ck: [{apple_ck: 1.}]}

# NACS associative rules govern how activations may propagate among chunks in 
# NACS. toplevel_assoc is a dict whose keys represent conclusion chunks, its 
# values are lists of condition-weight mappings, There is one mapping for each 
# individual rule about a particular conclusion chunk.

# Alice's initial inter-level associations

interlevel_assoc = {
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
}

# interlevel_assoc defines how activations may propagate from chunks to 
# microfeatures and vice-versa. interlevel_assoc keys are chunks, which are 
# each linked to zero or more microfeatures. Linked microfeatures are grouped 
# by dimension, as inter-level activation propagation applies a 
# dimension-dependent weight to strengths.

# Below, actions represents the set of responses that the agent may give during 
# the task. Action callbacks may also represent non-external actions, such as 
# goal-setting, attention allocation, working-memory allocation etc., depending 
# on simulation requirements.

actions = {
    apple_ck: lambda: recorder.actions.append(apple_ck),
    juice_ck: lambda: recorder.actions.append(juice_ck),
    fruit_ck: lambda: recorder.actions.append(fruit_ck)
}

def default_strength(node = None):
    """Default node strength."""
    
    return 0.0

# NACS Prep

# Here, realizers are created for every initial piece of knowledge that Alice 
# has in addition to other important functional components of Alice's NACS.

nacs_contents = [
    NodeRealizer(
        csym=apple_ck, 
        junction=SimpleNodeJunction(apple_ck, default_strength)
    ),
    NodeRealizer(juice_ck, SimpleNodeJunction(juice_ck, default_strength)),
    NodeRealizer(fruit_ck, SimpleNodeJunction(fruit_ck, default_strength)),
    NodeRealizer(red_mf, SimpleNodeJunction(red_mf, default_strength)),
    NodeRealizer(green_mf, SimpleNodeJunction(green_mf, default_strength)),
    NodeRealizer(tasty_mf, SimpleNodeJunction(tasty_mf, default_strength)),
    NodeRealizer(sweet_mf, SimpleNodeJunction(sweet_mf, default_strength)),
    NodeRealizer(liquid_mf, SimpleNodeJunction(liquid_mf, default_strength)),
    FlowRealizer(
        csym=associative_rules_flow, 
        junction=SimpleJunction(),
        channel=AssociativeRuleCollection(toplevel_assoc, default_strength)
    ),
    FlowRealizer(
        top_down_flow,
        SimpleJunction(),
        TopDownLinks(interlevel_assoc, default_strength)
    ),
    FlowRealizer(
        bottom_up_flow,
        SimpleJunction(),
        BottomUpLinks(interlevel_assoc, default_strength)
    ),
    AppraisalRealizer(
        appraisal_nacs,
        SimpleJunction(),
        SimpleBoltzmannSelector(temperature = .1)
    ),
    BehaviorRealizer(behavior_nacs, MappingEffector(actions))
]

# Agent Assembly

# We create an AgentRealizer representing alice.
alice = AgentRealizer(Agent("Alice"))

# We add an NACS.
alice[nacs] = SubsystemRealizer(
    csym=nacs, 
    propagation_rule=nacs_propagation_cycle
)

# We insert the components defined above into Alice's NACS. The constructs are 
# automatically linked! 
alice[nacs].insert_realizers(*nacs_contents)

# We add a buffer enabling stimulation of Alice's NACS.
alice[stim_buf] = BufferRealizer(stim_buf, ConstantSource())

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

    print(" ", "Appraisal:")
    for _, realizer in nacs.items_ctype(ConstructType.Appraisal):
        for c, s in realizer.output.view().strengths.items():
            print("   ", c, round(s, 3))
    # We could just get responses from the appraisal decision packet, but why 
    # not demonstrate that the agent has affected its envirnoment?
    print("   ", "Response:", recorder.actions.pop())


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

class AppraisalFilterJunction(SimpleJunction):

    def __init__(self, filter_dict = None):

        self.fdict = filter_dict or {}

    def __call__(self, packets):

        d = super().__call__(packets)
        for node, factor in self.fdict.items():
            if node in d: d[node] *= factor
        return d

# We need to replace the existing junction for nacs appraisals with a new one.
# This is very easy: just assign to buffer.junction!
alice[nacs][appraisal_nacs].junction = AppraisalFilterJunction()

# We assume that alice filters out the cue. The decision to do this and its 
# execution are not the responsibility of NACS, so they are not explicitly 
# simulated.
alice[nacs][appraisal_nacs].junction.fdict[apple_ck] = .0

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
        behavior = Behavior("Respond", Appraisal("NACS", ConstructType.Chunk))

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
        recorder, alice[nacs], toplevel_assoc, interlevel_assoc
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
