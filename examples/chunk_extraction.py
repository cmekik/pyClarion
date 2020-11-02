"""
Demo for chunk extraction in pyClarion.

Prerequisite: Understanding of the basics of pyClarion as discussed in the demo
`free_association.py`.
"""

from pyClarion import (
    Structure, Construct,
    agent, subsystem, buffer, flow_bt, flow_tb, features, chunks, terminus, 
    feature,
    Chunks, Assets, ChunkAdder,
    AgentCycle, NACSCycle, Stimulus, MaxNodes, TopDown, BottomUp, 
    BoltzmannSelector, ThresholdSelector
)
import pprint


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
# We are teaching Alice about fruits by showing her pictures of fruits and 
# simultaneously speaking out their names. Afterwards, we quiz alice by either 
# showing her more pictures or naming fruits.


#############
### Setup ###
#############

### Agent Setup ###

# For this simulation, we develop a simple feature domain containing visual and 
# auditory features. The visual features include color, shape, size, and 
# texture. The only auditory dimension is that of words. 

fspecs = [
    ("word", "/banana/"),
    ("word", "/apple/"),
    ("word", "/orange/"),
    ("word", "/plum/"),
    ("color", "red"),
    ("color", "green"),
    ("color", "yellow"),
    ("color", "orange"),
    ("color", "purple"),
    ("shape", "round"),
    ("shape", "oblong"),
    ("size", "small"),
    ("size", "medium"),
    ("texture", "smooth"),
    ("texture", "grainy"),
    ("texture", "spotty")
]

# As in `free_association.py`, we construct a chunk database to store chunks. 
# However, instead of populating the database manually, we will have the agent 
# create chunks based on its interactions with audio-visual stimuli. 

chunk_db = Chunks()

# Agent setup proceeds more or less as with the free association example. 
# Except that we include a `ChunkAdder` object as an updater.

chunk_adder = ChunkAdder(
    chunks=chunk_db,
    terminus=terminus("bl"),
    prefix="bl"
)

# Needless to say, the ChunkAdder is the star of this particular demo. Its job 
# is to add chunks to the chunk database with which it is entrusted. To do 
# this, the chunk adder monitors a terminus that selects active features, and, 
# at update time on each step, it creates and adds a new chunk to the database 
# containing the selected features if no such chunk exists.

# The `ChunkAdder` minimally needs to be given a terminus construct to monitor 
# and a prefix to use in constructing chunk names (the names are formatted as 
# "prefix-number") in addition to a reference to the client chunk database.

# Agent Assembly

# The assembly process should be familiar from the free association example, 
# with only a couple of mild novelties.

alice = Structure(
    name=agent("alice"),
    emitter=AgentCycle()
)

with alice:

    stimulus = Construct(
        name=buffer("stimulus"), 
        emitter=Stimulus()
    )

    # Note that we add the chunk database and the chunk adder are placed at the 
    # level of the NACS. The two constructs need not be placed in the same 
    # construct. Notably, in more complex models, it may be necessary to place 
    # the chunk adder higher up in the hierarchy depending on the desired order 
    # of update applications.

    nacs = Structure(
        name=subsystem("nacs"),
        emitter=NACSCycle(
            sources={
                buffer("stimulus")
            }
        ),
        updater=chunk_adder,
        assets=Assets(
            chunk_db=chunk_db
        )
    )

    with nacs:

        # Chunk and Feature Pools

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={
                    buffer("stimulus"), 
                    flow_tb("main")
                }
            )
        )

        Construct(
            name=chunks("main"),
            emitter=MaxNodes(
                sources={
                    flow_bt("main")
                }
            )
        )

        # Flows

        # For this example, we create a simple NACS without horizontal flows 
        # (i.e., no rules and no associative memory networks). So, we only 
        # include top-down and bottom-up flows.

        Construct(
            name=flow_bt("main"), 
            emitter=BottomUp( 
                source=features("main"),
                chunks=nacs.assets.chunk_db
            ) 
        )

        Construct(
            name=flow_tb("main"), 
            emitter=TopDown( 
                source=chunks("main"),
                chunks=nacs.assets.chunk_db
            ) 
        )

        # Termini

        # In addition to introducting chunk extraction, this example 
        # demonstrates the use of two temrmini in one single subsytem. We 
        # include one terminus for the output of the top level and one for the 
        # bottom level. 

        # The top level terminus is basically the same as the one used in the 
        # free association example. It randomly selects a chunk through a 
        # competitive process which involves sampling chunks from a boltzmann 
        # distribution constructed from their respective strength values. This 
        # terminus is relevant for the quizzing/querying section of the 
        # simulation.

        Construct(
            name=terminus("tl"),
            emitter=BoltzmannSelector(
                source=chunks("main"),
                temperature=0.01,
                threshold=0.0
            )
        )

        # The bottom level terminus introduces a new emitter, the 
        # `ThresholdSelector`. As suggested by the name, this emitter selects 
        # features based on a threshold applied to feature strengths in the 
        # bottom level. Features that are above threshold according to the 
        # selector will be picked up by the chunk adder and included in any 
        # newly constructed chunks.

        Construct(
            name=terminus("bl"),
            emitter=ThresholdSelector(
                source=features("main"),
                threshold=0.9
            )
        )


# We start the agent to complete preparation for simulations.

alice.start()

# Agent setup is now complete!


##################
### Simulation ###
##################

# In the learning stage of the simulation, let us imagine that we present the 
# agent with a sequence of fruit pictures paired with the words that name the 
# respective fruits.

# In this demo, we will present four instances of such stimuli, as defined 
# below.

stimuli = [
    {
        feature("lexeme", "/apple/"): 1.0,
        feature("color", "red"): 1.0,
        feature("shape", "round"): 1.0,
        feature("size", "medium"): 1.0,
        feature("texture", "smooth"): 1.0
    },
    {
        feature("lexeme", "/orange/"): 1.0,
        feature("color", "orange"): 1.0,
        feature("shape", "round"): 1.0,
        feature("size", "medium"): 1.0,
        feature("texture", "grainy"): 1.0
    },
    {
        feature("lexeme", "/banana/"): 1.0,
        feature("color", "yellow"): 1.0,
        feature("shape", "oblong"): 1.0,
        feature("size", "medium"): 1.0,
        feature("texture", "spotty"): 1.0
    },
    {
        feature("lexeme", "/plum/"): 1.0,
        feature("color", "purple"): 1.0,
        feature("shape", "round"): 1.0,
        feature("size", "small"): 1.0,
        feature("texture", "smooth"): 1.0
    }
]

# In the loop below, we present each stimulus in turn and print the state of 
# the agent at each step. The agent will automatically extract chunks as 
# necessary. The final chunk database is printed on loop termination.

# To prevent interference between presentations, we clear the output at the 
# end of each presentation. Admittedly, this is somewhat arbitrary, but helps 
# keep the demo simple. (A better way of doing this would be to block top-down 
# flows through flow control mechanisms as demonstrated in `flow_control.py`.)

for i, s in enumerate(stimuli):
    print("Presentation {}".format(i + 1))

    stimulus.emitter.input(s)
    alice.step()

    pprint.pprint(alice.output)

    alice.clear_outputs()

print("Learned Chunks:")
nacs.assets.chunk_db.pprint()

# Visual and Auditory Queries

# Having formed some chunks, we can perform visual and auditory queries. In 
# this simple setting, we represent queries as feature activations and take 
# query responses to be the chunks selected by the top level terminus called 
# 'terminus("tl")'.

# There are two queries defined below, one visual and one auditory. 

queries = [
    {
        feature("color", "green"): .85,
        feature("shape", "round"): .85,
        feature("size", "medium"): .85,
        feature("texture", "smooth"): .85
    },
    {
        feature("lexeme", "/orange/"): .85,
    }
]

# To prevent formation of new chunks, we set the query strengths to be below 
# the chunk inclusion threshold. This is a simple, but hacky, solution. For a 
# more elegant approach, once again refer to `flow_control.py`.

# Once again we run a loop, presenting the stimuli and printing the agent 
# state at each step.

for i, s in enumerate(queries):
    print("Presentation {}".format(i + 1))

    stimulus.emitter.input(s)
    alice.step()

    pprint.pprint(alice.output)
    
    alice.clear_outputs()


##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - The basics of using updaters for learning, and
#   - A recipe for chunk extraction from the state of the bottom level. 
