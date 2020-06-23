from pyClarion import *
from typing import cast

alice = Agent(
    name="Alice",
    assets=Assets(chunks=Chunks()),
    updaters={
        "chunk_adder": ChunkAdder(
            template=ChunkAdder.Template(
                matches=MatchSpec(
                    ctype=ConstructType.flow_xt,
                    constructs={buffer("Stimulus")}
                ),
                propagator=MaxNode()
            ),
            response=response("Extractor"),
            subsystem=subsystem("NACS")
        )
    }
)

stimulus = Buffer(name="Stimulus", propagator=Stimulus())
wm = Buffer(
    name="WM",
    matches={subsystem("NACS")},
    propagator=WorkingMemory(
        source=subsystem("NACS"),
        chunks=alice.assets.chunks
    )
)
alice.add(stimulus, wm)

nacs = Subsystem(
    name="NACS",
    matches={buffer("Stimulus"), buffer("WM")},
    cycle=NACSCycle()
)
alice.add(nacs)

nacs.add(
    Flow(
        name=flow_bt("Main"), 
        matches=ConstructType.feature, 
        propagator=BottomUp(chunks=alice.assets.chunks) # type: ignore
    ),
    Flow(
        name=flow_tb("Main"), 
        matches=ConstructType.chunk, 
        propagator=TopDown(chunks=alice.assets.chunks) # type: ignore
    )
)

fnodes = [
    Node(
        name=feature(dim, val), 
        matches=MatchSpec(
            ctype=ConstructType.flow_xb, 
            constructs={buffer("Stimulus")}
        ), 
        propagator=MaxNode()
    ) for dim, val in [
        ("fruit", "banana"),
        ("fruit", "kiwi"),
        ("fruit", "blueberry"),
        ("fruit", "dragon fruit"),
        ("fruit", "orange"),
        ("fruit", "strawberry"),
        ("price", "very cheap"),
        ("price", "cheap"),
        ("price", "fair"),
        ("price", "expensive"),
        ("price", "very expensive"),
    ]
]
nacs.add(*fnodes)

# As mentioned, we need to create a special response construct that produces 
# new chunk recommendations. This is achieved with a `ChunkExtractor` object,
# which assumes that chunks are stored in a `Chunks` object.

nacs.add(
    Response(
        name="Retriever",
        matches=MatchSpec(
            ctype=ConstructType.chunk, 
            constructs={buffer("Stimulus")}
        ),
        propagator=FilteredR(
            base=BoltzmannSelector(temperature=.1),
            input_filter=buffer("Stimulus"))
    ),
    Response(
        name="Extractor",
        matches=MatchSpec(
            ctype=ConstructType.feature, 
            constructs={buffer("Stimulus")}
        ),
        propagator=ChunkExtractor(
            chunks=alice.assets.chunks,
            name="state",
            filter=MatchSpec(ctype=ConstructType.feature),
            threshold=0.9
        )
    )
)

# Agent setup is now complete!

##################
### Simulation ###
##################

stimulus_states = [
    {
        feature("fruit", "dragon fruit"): 1.0,
        feature("price", "expensive"): 1.0
    },
    {
        feature("fruit", "orange"): 1.0,
        feature("price", "expensive"): 1.0
    },
    {
        feature("fruit", "orange"): 1.0,
        feature("price", "expensive"): 1.0
    }
]
wm_channel = [response("Extractor"), response("Extractor"), None] 

for i, (stimulus_state, channel) in enumerate(zip(stimulus_states, wm_channel)):
    print("Presentation {}".format(i + 1))
    alice.propagate(args={buffer("Stimulus"): {"stimulus": stimulus_state}})
    alice.learn()
    cast(WorkingMemory, wm.propagator).update_on_next(channel)
    print(nacs.output.pstr())
    print(wm.output.pstr())

print("Learned Chunks:")
alice.assets.chunks.pprint()

##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - The basics of using updaters for learning, and
#   - A recipe for chunk extraction from the state of the bottom level. 
