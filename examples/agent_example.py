"""
A simple pyClarion demo.

Below we simulate an agent, named Alice, doing a cued association task. She will 
tell us the first thing that comes to her mind upon being presented with the 
chosen stimulus.
"""

from pyClarion.base import *
from pyClarion.components.processors import (
    UpdateJunction, MaxJunction, NodeMaxJunction, BoltzmannSelector, 
    MappingEffector, ConstantSource
)
from pyClarion.components.nacs import (
    AssociativeRuleCollection, TopDownLinks, BottomUpLinks, 
    nacs_propagation_cycle, may_connect
)


#############
### Setup ###
#############

### Simulation Environment ###

class BehaviorRecorder(object):

    def __init__(self):

        self.actions = []

# The object below allows us to record the agent's response
recorder = BehaviorRecorder()

### Agent Setup ###

# In this section, we'll create an agent named 'Alice' with some knowledge 
# about fruits.

# Constructs

fruit_ck = Chunk("FRUIT")
apple_ck = Chunk("APPLE")
juice_ck = Chunk("JUICE")

red_mf = Microfeature("color", "#ff0000")
green_mf = Microfeature("color", "#008000")
tasty_mf = Microfeature("tasty", True)
liquid_mf = Microfeature("state", "liquid")
sweet_mf = Microfeature("sweet", True)

appraisal_nacs = Appraisal("NACS")

nacs = Subsystem("NACS")
stim_buf = Buffer("NACS Stimulus")

# Alices's initial top-level knowledge

toplevel_assoc = {fruit_ck: [{apple_ck: 1.}]}

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

actions = {
    apple_ck: lambda: recorder.actions.append(apple_ck),
    juice_ck: lambda: recorder.actions.append(juice_ck),
    fruit_ck: lambda: recorder.actions.append(fruit_ck)
}

def default_strength(node = None):
    
    return 0.0

# NACS Prep

# Here, realizers are created for every initial piece of knowledge that alice 
# has in addition to other important functional components of Alice's NACS.

nacs_contents = [
    NodeRealizer(apple_ck, NodeMaxJunction(apple_ck, default_strength)),
    NodeRealizer(juice_ck, NodeMaxJunction(juice_ck, default_strength)),
    NodeRealizer(fruit_ck, NodeMaxJunction(fruit_ck, default_strength)),
    NodeRealizer(red_mf, NodeMaxJunction(red_mf, default_strength)),
    NodeRealizer(green_mf, NodeMaxJunction(green_mf, default_strength)),
    NodeRealizer(tasty_mf, NodeMaxJunction(tasty_mf, default_strength)),
    NodeRealizer(sweet_mf, NodeMaxJunction(sweet_mf, default_strength)),
    NodeRealizer(liquid_mf, NodeMaxJunction(liquid_mf, default_strength)),
    FlowRealizer(
        Flow("Associative Rules", FlowType.TT), 
        UpdateJunction(),
        AssociativeRuleCollection(toplevel_assoc, default_strength)
    ),
    FlowRealizer(
        Flow("NACS", FlowType.TB),
        UpdateJunction(),
        TopDownLinks(interlevel_assoc, default_strength)
    ),
    FlowRealizer(
        Flow("NACS", FlowType.BT),
        UpdateJunction(),
        BottomUpLinks(interlevel_assoc, default_strength)
    ),
    AppraisalRealizer(
        appraisal_nacs,
        UpdateJunction(),
        BoltzmannSelector(temperature = .1)
    ),
    BehaviorRealizer(Behavior("NACS"), MappingEffector(actions))
]

# We create an AgentRealizer representing alice
alice = AgentRealizer(Agent("Alice"))

# We add an NACS
alice[nacs] = SubsystemRealizer(nacs, nacs_propagation_cycle, may_connect)

# We insert the components defined above into Alice's NACS
alice[nacs].insert_realizers(*nacs_contents)

# We add a buffer enabling stimulation of Alice's NACS
alice[stim_buf] = BufferRealizer(stim_buf, ConstantSource())

# Next, we connect the stimulus buffer to Alice's NACS nodes
for csym, realizer in alice[nacs].items():
    if csym.ctype in ConstructType.Node:
        realizer.input.watch(stim_buf, alice[stim_buf].output.view)


##################
### Simulation ###
##################

# The stimulus buffer is set to provide constant activation to the Apple 
# chunk. This represents presentation of the concept APPLE. Note that there 
# are simplifications. For instance, it is assumed that Alice is made to 
# understand that the cue is APPLE, the fruit, and not e.g., APPLE, the 
# company. 
alice[stim_buf].source.update({apple_ck: 1.})

# Another limitation of this model: in cued association, subjects tend not 
# to return the cue as their response but this is not the case for Alice. 
# Cue suppression requires input/output filtering, which is a function not 
# included in the current simulation. 

# Alice performs one NACS cycle. 
alice.propagate()

# Alice responds.
alice.execute()


# We can look at the exact state of Alice's NACS at the end of its 
# activation cycle.

print("Initial Trial")
for c in alice[nacs]:
    if c.ctype in ConstructType.Node:
        print(" ", c, round(alice[nacs][c].output.view().strengths[c], 3))
print(" ", "Appraisal:")
for c, s in alice[nacs][appraisal_nacs].output.view().strengths.items():
    print("   ", c, round(s, 3))
print(" ", "Response:", recorder.actions.pop())

################
### Learning ###
################

# Learning is an essential part of Clarion, so a simple example below 
# demonstrates learning in pyClarion. Note that we will simply be modifying 
# Alice on the fly to enable learning. 

# First, we must define a learning routine. This will do:

class HeavyHandedLearningRoutine(object):
    # This routine simply injects some preset knowledge into the NACS.

    def __init__(self, recorder, nacs, toplevel_assoc, interlevel_assoc):

        self.recorder = recorder
        self.nacs = nacs
        self.toplevel_assoc = toplevel_assoc
        self.interlevel_assoc = interlevel_assoc
    
    def __call__(self) -> None:

        fruit_ck = Chunk("FRUIT")
        orange_ck = Chunk("ORANGE")
        orange_color_mf = Microfeature("color", "#ffa500")
        tasty_mf = Microfeature("tasty", True)
        top_down_flow = Flow("NACS", ftype=FlowType.TB)
        bottom_up_flow = Flow("NACS", ftype=FlowType.BT)
        behavior = Behavior("NACS")

        self.nacs[orange_ck] = NodeRealizer(
            orange_ck, NodeMaxJunction(orange_ck, default_strength)
        )
        self.nacs[orange_color_mf] = NodeRealizer(
            orange_color_mf, NodeMaxJunction(orange_color_mf, default_strength)
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
print("With Learning")
for c in alice[nacs]:
    if c.ctype in ConstructType.Node:
        print(" ", c, round(alice[nacs][c].output.view().strengths[c], 3))
print(" ", "Appraisal:")
for c, s in alice[nacs][appraisal_nacs].output.view().strengths.items():
    print("   ", c, round(s, 3))
print(" ", "Response:", recorder.actions.pop())
