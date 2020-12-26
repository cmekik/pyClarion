"""
Demo for chunk extraction in pyClarion.

Prerequisite: Understanding of the basics of pyClarion as discussed in the demo
`free_association.py`.
"""


from pyClarion import (
    Structure, Construct,
    agent, subsystem, buffer, flow_bt, flow_tb, features, chunks, terminus, 
    feature, updater,
    Chunks, Assets,
    Stimulus, MaxNodes, TopDown, BottomUp, 
    BoltzmannSelector, ChunkExtractor, ChunkDBUpdater,
    pprint
)


#############
### Setup ###
#############

# This simulation demonstrates a basic recipe for chunk extraction in 
# pyClarion. If you have not worked through the free association example, 
# please do so first, as you will be missing some of the prerequisite ideas.

# The basic premise of the recipe is to create a special terminus construct 
# for chunk extraction. In pyClarion, each simulation step consists of a 
# propagation stage, called 'propagation time', and an updating/learning stage, 
# called 'update time'. On each cycle, the chunk extractor recommends chunks 
# based on the state of the bottom level at propagation time. If the bottom 
# level state matches an existing chunk, that chunk is recommended. Otherwise, 
# a new chunk is recommended. These recommendations are then placed in the 
# corresponding chunk database at update time by an appropriate updater object.

# Here is the scenario:
# We are teaching Alice about fruits by showing her pictures of fruits and 
# simultaneously speaking out their names. Afterwards, we quiz alice by either 
# showing her more pictures or naming fruits.

### Knowledge Setup ###

# For this simulation, we develop a simple feature domain containing visual and 
# auditory features. The visual features include color, shape, size, and 
# texture. The only auditory dimension is that of words. 

fspecs = [
    feature("word", "/banana/"),
    feature("word", "/apple/"),
    feature("word", "/orange/"),
    feature("word", "/plum/"),
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
]

# As in `free_association.py`, we construct a chunk database to store chunks. 
# However, instead of populating the database manually, we will have the agent 
# create chunks based on its interactions with audio-visual stimuli. 

chunk_db = Chunks()


### Agent Assembly ###

# The assembly process should be familiar from the free association example, 
# with only a couple of mild novelties.

alice = Structure(
    name=agent("alice")
)

with alice:

    stimulus = Construct(
        name=buffer("stimulus"), 
        process=Stimulus()
    )

    # When we create the NACS, we store the chunk database as an asset, as 
    # before. However, we also pass the chunk database's updater object as an 
    # updater for NACS. This is because when the extractor recommends a new 
    # chunk, it will issue an update request to the chunk database. Upon 
    # receipt, this update request is deferred until update time. The chunk 
    # database updater takes responsibiity for applying any requested updates. 
    # This pattern ensures that the chunk database remains constant during 
    # propagation time. It also helps ensure that updates remain consistent 
    # even if multiple constructs issue update requests to the same chunk 
    # database. 

    nacs = Structure(
        name=subsystem("nacs"),
        assets=Assets(chunk_db=chunk_db)
    )

    with nacs:

        Construct(
            name=features("main"),
            process=MaxNodes(
                sources=[buffer("stimulus")]
            )
        )

        Construct(
            name=flow_bt("main"), 
            process=BottomUp( 
                source=features("main"),
                chunks=nacs.assets.chunk_db
            ) 
        )

        Construct(
            name=chunks("main"),
            process=MaxNodes(
                sources=[flow_bt("main")]
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
            process=BoltzmannSelector(
                source=chunks("main"),
                temperature=0.01,
                threshold=0.0
            )
        )

        # The bottom level ('bl') terminus introduces a new emitter, the 
        # `ChunkExtractor`. As suggested by the name, this emitter extracts 
        # chunks capturing the state of the bottom level. More precisely, it 
        # first applies a thresholding function to feature activations in the 
        # bottom level. Then, it looks for a chunk whose form matches exactly 
        # the features above threshold. If a match is found, the corresponding 
        # chunk is emitted as output (fully activated). If no match is found, 
        # a new chunk is named and emitted, and a request is sent to the chunk 
        # database to add the new chunk.

        Construct(
            name=terminus("bl"),
            process=ChunkExtractor(
                source=features("main"),
                threshold=0.9,
                chunks=nacs.assets.chunk_db,
                prefix="bl"
            )
        )

        Construct(
            name=updater("cdb"),
            process=ChunkDBUpdater(chunks=nacs.assets.chunk_db)
        )

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
        feature("word", "/apple/"): 1.0,
        feature("color", "red"): 1.0,
        feature("shape", "round"): 1.0,
        feature("size", "medium"): 1.0,
        feature("texture", "smooth"): 1.0
    },
    {
        feature("word", "/orange/"): 1.0,
        feature("color", "orange"): 1.0,
        feature("shape", "round"): 1.0,
        feature("size", "medium"): 1.0,
        feature("texture", "grainy"): 1.0
    },
    {
        feature("word", "/banana/"): 1.0,
        feature("color", "yellow"): 1.0,
        feature("shape", "oblong"): 1.0,
        feature("size", "medium"): 1.0,
        feature("texture", "spotty"): 1.0
    },
    {
        feature("word", "/plum/"): 1.0,
        feature("color", "purple"): 1.0,
        feature("shape", "round"): 1.0,
        feature("size", "small"): 1.0,
        feature("texture", "smooth"): 1.0
    }
]

# In the loop below, we present each stimulus in turn and print the state of 
# the agent at each step. The agent will automatically extract chunks. The 
# final chunk database is printed on loop termination.

for i, s in enumerate(stimuli):
    print("Presentation {}".format(i + 1))

    stimulus.process.input(s)
    alice.step()

    pprint(alice.output)

print("Learned Chunks:")
pprint(nacs.assets.chunk_db)

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
        feature("word", "/orange/"): .85,
    }
]

# To prevent formation of new chunks, we set the query strengths to be below 
# the chunk inclusion threshold. This is a simple, but hacky, solution. A more 
# elegant approach would be to use some kind of control mechanism (e.g. a 
# `ControlledExtractor`).

# Once again we run a loop, presenting the stimuli and printing the agent 
# state at each step.

for i, s in enumerate(queries):
    print("Presentation {}".format(i + 1))

    stimulus.process.input(s)
    alice.step()

    pprint(alice.output)


##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - The basics of learning and updater use, and
#   - A recipe for chunk extraction from the state of the bottom level. 
