from pyClarion import *
from typing import cast
import pprint

alice = Structure(
    name=agent("Alice"),
    emitter=AgentCycle(),
    assets=Assets(chunks=Chunks())
)

with alice:

    stimulus = Construct(name=buffer("Stimulus"), emitter=Stimulus())

    WMSLOTS = 3
    wm = Construct(
        name=buffer("WM"),
        emitter=WorkingMemory(
            controller=(subsystem("ACS"), terminus("WM")),
            source=subsystem("NACS"),
            interface=WorkingMemory.Interface(
                write_dlbs=tuple("wm-w{}".format(i) for i in range(WMSLOTS)),
                standby="standby",
                clear="clear",
                channel_map=(
                    ("retrieve", terminus("retrieval")),
                    ("extract",  terminus("bl-state"))
                ),
                reset_dlb="wm-reset",
                reset_vals=("standby", "release"),
                read_dlbs=tuple("wm-r{}".format(i) for i in range(WMSLOTS)),
                read_vals=("standby", "open"),
            ) 
        ),
        updater=WorkingMemory.StateUpdater()
    )

    # This default activation can be worked into the WM object, simplifying 
    # agent construction...

    wm_defaults = Construct(
        name=buffer("WM-defaults"),
        emitter=ConstantBuffer(
            strengths={f: 0.5 for f in wm.emitter.interface.defaults}
        )
    )

    acs = Structure(
        name=subsystem("ACS"),
        emitter=ACSCycle(
            matches=MatchSet(
                constructs={
                    buffer("Stimulus"), buffer("WM"), buffer("WM-defaults")
                }
            )
        )
    )

    with acs:

        for f in wm.emitter.interface.features:
            Construct(
                name=f, 
                emitter=MaxNode(
                    matches=MatchSet(
                        ctype=ConstructType.flow_xb, 
                        constructs={
                            buffer("Stimulus"), 
                            buffer("WM-defaults")
                        }
                    )
                )
            ) 

        Construct(
            name=terminus("WM"),
            emitter=ActionSelector(
                temperature=.01,
                dims=list(wm.emitter.interface.dims)
            )
        )

    nacs = Structure(
        name=subsystem("NACS"),
        emitter=NACSCycle(
            matches={buffer("Stimulus"), buffer("WM"), buffer("WM-defaults")}
        ),
        updater=ChunkAdder(
            chunks=alice.assets.chunks,
            terminus=terminus("bl-state"),
            emitter=MaxNode(
                MatchSet(
                    ctype=ConstructType.flow_xt,
                    constructs={buffer("Stimulus")}
                ),
            ),
            prefix="bl-state"
        )
    )

    with nacs:

        Construct(
            name=flow_bt("Main"), 
            emitter=BottomUp(chunks=alice.assets.chunks) 
        )

        Construct(
            name=flow_tb("Main"), 
            emitter=TopDown(chunks=alice.assets.chunks)
        )

        fspecs = [
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

        for dlb, val in fspecs:
            Construct(
                name=feature(dlb, val), 
                emitter=MaxNode(
                    matches=MatchSet(
                        ctype=ConstructType.flow_xb, 
                        constructs={buffer("Stimulus")}
                    )
                )
            ) 

        # As mentioned, we need to create a special terminus construct that 
        # produces new chunk recommendations. This is achieved with a 
        # `ChunkExtractor` object, which assumes that chunks are stored in a 
        # `Chunks` object.

        Construct(
            name=terminus("retrieval"),
            emitter=FilteredT(
                base=BoltzmannSelector(
                    temperature=.1,
                    matches=MatchSet(ctype=ConstructType.chunk)
                ),
                filter=buffer("Stimulus")
            )
        )

        Construct(
            name=terminus("bl-state"),
            emitter=ThresholdSelector(threshold=0.9)
        )

# Agent setup is now complete!

##################
### Simulation ###
##################

print(
    "Each simulation example consists of two propagation cycles.\n"
    "In the first, the WM is updated through the ACS; the commands are shown.\n"
    "In the secondg, we probe the WM output to demonstrate the effect.\n"
)

# standby (empty wm)
print("Standby (Empty WM)")

d = {
    feature("fruit", "dragon fruit"): 1.0,
    feature("price", "expensive"): 1.0,
}

stimulus.emitter.input(d)
alice.propagate()
alice.update()

print(
    "Step 1: {} -> {}".format(
        wm.emitter.controller, 
        alice[wm.emitter.controller].output
    )
)

alice.propagate()
alice.update()

print("Step 2: {} -> {}\n".format(buffer("WM"), alice.output[buffer("WM")]))

# open empty (should do nothing)
print("Open (Empty WM; does nothing)")

d = {feature("wm-r1", "open"): 1.0}

stimulus.emitter.input(d)
alice.propagate()
alice.update()

print(
    "Step 1: {} -> {}".format(
        wm.emitter.controller, 
        alice[wm.emitter.controller].output
    )
)

alice.propagate(kwds={})
alice.update()

print("Step 2: {} -> {}\n".format(buffer("WM"), alice.output[buffer("WM")]))

# single write
print("Single Write")

d = {
    feature("fruit", "dragon fruit"): 1.0,
    feature("price", "expensive"): 1.0,
    feature("wm-w0", "retrieve"): 1.0,
    feature("wm-r0", "open"): 1.0
}

stimulus.emitter.input(d)
alice.propagate()
alice.update()

print(
    "Step 1: {} -> {}".format(
        wm.emitter.controller, 
        alice[wm.emitter.controller].output
    )
)

alice.propagate()
alice.update()

print("Step 2: {} -> {}\n".format(buffer("WM"), alice.output[buffer("WM")]))

# reset
print("Reset")

d = {
    feature("fruit", "dragon fruit"): 1.0,
    feature("price", "expensive"): 1.0,
    feature("wm-reset", "release"): 1.0
}

stimulus.emitter.input(d)
alice.propagate()
alice.update()

print(
    "Step 1: {} -> {}".format(
        wm.emitter.controller, 
        alice[wm.emitter.controller].output
    )
)

alice.propagate(kwds={})
alice.update()

print("Step 2: {} -> {}\n".format(buffer("WM"), alice.output[buffer("WM")]))


# double write
print("Double Write")

d = {
    feature("fruit", "banana"): 1.0,
    feature("price", "expensive"): 1.0,
    feature("wm-w0", "retrieve"): 1.0,
    feature("wm-r0", "open"): 1.0,
    feature("wm-w1", "extract"): 1.0,
    feature("wm-r1", "open"): 1.0
}

stimulus.emitter.input(d)
alice.propagate()
alice.update()

print(
    "Step 1: {} -> {}".format(
        wm.emitter.controller, 
        alice[wm.emitter.controller].output
    )
)

alice.propagate()
alice.update()

print("Step 2: {} -> {}\n".format(buffer("WM"), alice.output[buffer("WM")]))

# Open Slot 1, removing it
print("Open Slot 1 only, removing Slot 0 from output")

d = {feature("wm-r1", "open"): 1.0}

stimulus.emitter.input(d)
alice.propagate()
alice.update()

print(
    "Step 1: {} -> {}".format(
        wm.emitter.controller, 
        alice[wm.emitter.controller].output
    )
)

alice.propagate()
alice.update()

print("Step 2: {} -> {}\n".format(buffer("WM"), alice.output[buffer("WM")]))


# single delete
print("Single Delete (clear & open slot 0)")

d = {
    feature("wm-w0", "clear"): 1.0,
    feature("wm-r0", "open"): 1.0
}

stimulus.emitter.input(d)
alice.propagate()
alice.update()

print(
    "Step 1: {} -> {}".format(
        wm.emitter.controller, 
        alice[wm.emitter.controller].output
    )
)

alice.propagate()
alice.update()

print("Step 2: {} -> {}\n".format(buffer("WM"), alice.output[buffer("WM")]))