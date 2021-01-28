"""Demonstrates a more complex simulation, where NACS guides ACS."""


from pyClarion import (
    feature, chunk, terminus, features, chunks, buffer, subsystem, agent, 
    flow_in, flow_tb, flow_bt,
    Domain, Interface,
    Construct, Structure,
    RegisterArray, Stimulus, Constants, TopDown, BottomUp, MaxNodes,
    Filtered, ActionSelector, BoltzmannSelector,
    Assets, Chunks,
    nd, pprint
)

from itertools import count

 
#############
### Setup ###
#############

# This simulation demonstrates a basic recipe for creating and working with 
# working memory structures in pyClarion to guide action selection. 

# Here is the scenario:
# Alice has learned about fruits, as demonstrated in the chunk extraction 
# example. She is now presented some strange new fruits and attempts to 
# identify them.

### Knowledge Setup ###

# For this simulation, we continue using the fruit-related visual feature 
# domain from `chunk_extraction.py`. 

visual_domain = Domain(
    features=(
        feature("color", "red"),
        feature("color", "green"),
        feature("color", "yellow"),
        feature("color", "orange"),
        feature("color", "purple"),
        feature("shape", "round"),
        feature("shape", "oblong"),
        feature("size", "small"),
        feature("size", "medium"),
        feature("texture", "smooth"),
        feature("texture", "grainy"),
        feature("texture", "spotty")
    )
)

# We will have word features do double duty as perceptual representations and 
# action features. The idea here is that when a word feature is selected as an 
# action, the agent is assumed to have uttered that word (e.g., in response to 
# the question "What is the fruit that you see?").

speech_interface = Interface(
    cmds=(
        feature("word", "//"), # Silence
        feature("word", "/banana/"),
        feature("word", "/apple/"),
        feature("word", "/orange/"),
        feature("word", "/plum/"),
    )
)

# The working memory interface serves several purposes. It sets the number of 
# available memory slots, defines commands for writing to and reading from each 
# individual slot, and defines commands for globally resetting the working 
# memory state.

wm_interface = RegisterArray.Interface(
    name="wm",
    slots=3,
    vops=("retrieve",)
)

# We set up default action activations, as in `flow_control.py`.

default_strengths = nd.MutableNumDict(default=0)
default_strengths.extend(
    wm_interface.defaults,
    speech_interface.defaults,
    value=0.5
)

# As in `chunk_extraction.py`, we construct a chunk database to store chunks. 

nacs_cdb = Chunks()

# We then manually populate the database with chunks representing the fruits 
# that alice discovered in `chunk_extraction.py`.

nacs_cdb.define(
    chunk("APPLE"),
    feature("word", "/apple/"),
    feature("color", "red"),
    feature("shape", "round"),
    feature("size", "medium"),
    feature("texture", "smooth")
)

nacs_cdb.define(
    chunk("ORANGE"),
    feature("word", "/orange/"),
    feature("color", "orange"),
    feature("shape", "round"),
    feature("size", "medium"),
    feature("texture", "grainy")
)

nacs_cdb.define(
    chunk("BANANA"),
    feature("word", "/banana/"),
    feature("color", "yellow"),
    feature("shape", "oblong"),
    feature("size", "medium"),
    feature("texture", "spotty")
)

nacs_cdb.define(
    chunk("PLUM"),
    feature("word", "/plum/"),
    feature("color", "purple"),
    feature("shape", "round"),
    feature("size", "small"),
    feature("texture", "smooth")
) 


### Agent Assembly ###

# Agent assembly follows a pattern similar to that shown in `flow_control.py`.

alice = Structure(
    name=agent("alice"),
    assets=Assets(
        visual_domain=visual_domain,
        wm_interface=wm_interface,
        speech_interface=speech_interface
    )
)

with alice:

    stimulus = Construct(
        name=buffer("stimulus"), 
        process=Stimulus()
    )

    acs_ctrl = Construct(
        name=buffer("acs_ctrl"), 
        process=Stimulus()
    )

    # We define the working memory, by entrusting the working memory buffer 
    # construct to the RegisterArray process. 

    wm = Construct(
        name=buffer("wm"),
        process=RegisterArray(
            controller=(subsystem("acs"), terminus("wm")),
            sources=((subsystem("nacs"), terminus("retrieval")),),
            interface=wm_interface
        )
    )

    defaults = Construct(
        name=buffer("defaults"),
        process=Constants(strengths=default_strengths)
    )

    acs = Structure(
        name=subsystem("acs")
    )

    with acs:

        # We include a flow_in construct to handle converting chunk activations 
        # from working memory into features that the ACS can understand.

        Construct(
            name=flow_in("wm"), 
            process=TopDown(
                source=buffer("wm"),
                chunks=nacs_cdb
            ) 
        )

        Construct(
            name=features("main"),
            process=MaxNodes(
                sources=[
                    buffer("acs_ctrl"), 
                    flow_in("wm"),
                    buffer("defaults")
                ]
            )
        )

        # This terminus controls the working memory.

        Construct(
            name=terminus("wm"),
            process=ActionSelector(
                source=features("main"),
                temperature=.01,
                interface=alice.assets.wm_interface
            )
        )

        # This terminus controls the agent's speech actions.

        Construct(
            name=terminus("speech"),
            process=ActionSelector(
                source=features("main"),
                temperature=.01,
                interface=alice.assets.speech_interface
            )
        )

    nacs = Structure(
        name=subsystem("nacs"),
        assets=Assets(
            chunks=nacs_cdb
        )
    )

    with nacs:

        Construct(
            name=chunks("in"),
            process=MaxNodes(
                sources=[
                    buffer("stimulus"),
                    buffer("wm")
                ]
            )
        )

        Construct(
            name=flow_tb("main"), 
            process=TopDown(
                source=chunks("in"),
                chunks=nacs.assets.chunks
            )
        )

        Construct(
            name=features("main"),
            process=MaxNodes(
                sources=[
                    buffer("stimulus"), 
                    buffer("wm"),
                    flow_tb("main")
                ]
            )
        )

        Construct(
            name=flow_bt("main"), 
            process=BottomUp(
                source=features("main"),
                chunks=nacs.assets.chunks
            ) 
        )

        Construct(
            name=chunks("out"),
            process=MaxNodes(
                sources=[
                    chunks("in"), 
                    flow_bt("main")
                ]
            )
        )

        Construct(
            name=terminus("retrieval"),
            process=Filtered(
                base=BoltzmannSelector(
                    source=chunks("out"),
                    temperature=.1
                ),
                controller=buffer("wm")
            )
        )

# Agent setup is now complete!


##################
### Simulation ###
##################

# We write a convenience function to record simulation data. 

def record_step(agent, step):

    print("Step {}:".format(step))
    print()
    print("Activations")
    output = dict(agent.output)
    del output[buffer("defaults")]
    pprint(output)
    print()
    print("WM Contents")
    pprint([cell.store for cell in agent[buffer("wm")].process.cells])
    print()

# We simulate one proper trial. In this trial, alice is shown what should be an 
# ambiguous object for her, not quite fitting any fruit category she already 
# has. We want to know what alice thinks this object is. Because the stimulus 
# is purely visual, what alice has to do is effectively to retrieve an 
# association between what she sees and a word. This is achieved through the 
# fruit chunks that we have defined. Once a fruit chunk is forwarded to the 
# ACS, it can drive the utterance of a word corresponding to that chunk.

# This simulation mocks alice performing such a response process through manual 
# control of the ACS. The whole response procedure takes two steps.

print("Prompt: \"What do you see?\"\n")
print(
    "The stimulus is quite ambiguous, not fitting any fruit category that "
    "alice has seen before...\n"
)
print(
    "In response to the stimulus, alice retrieves a chunk from NACS based on "
    "the stimulus and forwards it to ACS.\nThis is achieved by simultaneously "
    "issuing read and write commands to WM slot 0 is Step 1.\nIn the next "
    "step, a non-default speech action is chosen thanks to the way alice's "
    "knowledge is structured.\n"
)

step = count(1)

# Step 1

stimulus.process.input({
    feature("color", "green"): 1.0,
    feature("shape", "round"): 1.0,
    feature("size", "medium"): 1.0,
    feature("texture", "smooth"): 1.0
})
acs_ctrl.process.input({
    feature(("wm", (".w", 0)), "retrieve"): 1.0,
    feature(("wm", (".r", 0)), ".read"): 1.0
})
alice.step()
record_step(alice, next(step))

# Step 2

stimulus.process.input({
    feature("color", "green"): 1.0,
    feature("shape", "round"): 1.0,
    feature("size", "medium"): 1.0,
    feature("texture", "smooth"): 1.0
})
acs_ctrl.process.input({})
alice.step()
record_step(alice, next(step))

print("Response: {}".format(alice[subsystem("acs")][terminus("speech")].output))


##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - A more complex simulation, where NACS drives action selection in ACS 
#     through working memory, and
#   - A recipe for setting up and using a working memory buffer.
