from pyClarion import *
from typing import cast
import pprint
import logging

logging.basicConfig(level=logging.INFO)

wm_interface = WorkingMemory.Interface(
    slots=7,
    prefix="wm",
    write_marker="w",
    read_marker="r",
    reset_marker="re",
    standby="standby",
    clear="clear",
    mapping={
        "retrieve": terminus("retrieval"),
        "extract":  terminus("bl-state")
    },
    reset_vals=("standby", "release"),
    read_vals=("standby", "open"),
) 

alice = Structure(
    name=agent("alice"),
    emitter=AgentCycle(),
    assets=Assets(
        chunks=Chunks(),
        wm_interface=wm_interface
    )
)

with alice:

    stimulus = Construct(
        name=buffer("stimulus"), 
        emitter=Stimulus()
    )

    WMSLOTS = 3
    wm = Construct(
        name=buffer("wm"),
        emitter=WorkingMemory(
            controller=(subsystem("acs"), terminus("wm")),
            source=subsystem("nacs"),
            interface=alice.assets.wm_interface
        )
    )

    # This default activation can be worked into the WM object, simplifying 
    # agent construction...

    wm_defaults = Construct(
        name=buffer("wm-defaults"),
        emitter=Constants(
            strengths={f: 0.5 for f in wm_interface.defaults}
        )
    )

    acs = Structure(
        name=subsystem("acs"),
        emitter=ACSCycle(
            sources={
                buffer("stimulus"), buffer("wm"), buffer("wm-defaults")
            }
        )
    )

    with acs:

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={
                    buffer("stimulus"), 
                    buffer("wm-defaults")
                }
            )
        )

        Construct(
            name=terminus("wm"),
            emitter=ActionSelector(
                source=features("main"),
                temperature=.01,
                client_interface=alice.assets.wm_interface
            )
        )

    nacs = Structure(
        name=subsystem("nacs"),
        emitter=NACSCycle(
            sources={
                buffer("stimulus")
            }
        ),
        updater=ChunkAdder(
            chunks=alice.assets.chunks,
            terminus=terminus("bl-state"),
            prefix="bl-state"
        )
    )

    with nacs:

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
                    buffer("stimulus"), 
                    flow_bt("main")
                }
            )
        )

        Construct(
            name=flow_bt("main"), 
            emitter=BottomUp(
                source=features("main"),
                chunks=alice.assets.chunks
            ) 
        )

        Construct(
            name=flow_tb("main"), 
            emitter=TopDown(
                source=chunks("main"),
                chunks=alice.assets.chunks
            )
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

        Construct(
            name=terminus("retrieval"),
            emitter=Filtered(
                base=BoltzmannSelector(
                    source=chunks("main"),
                    temperature=.1
                ),
                filter=buffer("stimulus")
            )
        )

        Construct(
            name=terminus("bl-state"),
            emitter=ThresholdSelector(
                source=features("main"),
                threshold=0.9
            )
        )

# Agent setup is now complete!

##################
### Simulation ###
##################

print(
    "Each simulation example consists of two propagation cycles.\n"
    "In the first, the WM is updated through the ACS; the commands are shown.\n"
    "In the second, we probe the WM output & state to demonstrate the effect.\n"
)

alice.start()

# standby (empty wm)
print("Standby (Empty WM)")
print()

d = {
    feature("fruit", "dragon fruit"): 1.0,
    feature("price", "expensive"): 1.0,
}

stimulus.emitter.input(d)
alice.step()

print("Step 1: {} ->".format(wm.emitter.controller))
pprint.pprint(alice[wm.emitter.controller].output)
print()

alice.step()

print("Step 2: {} ->".format(buffer("wm")))
pprint.pprint(alice.output[buffer("wm")])
print()

print("Current WM state.")
pprint.pprint([cell.store for cell in wm.emitter.cells])
print()

# open empty (should do nothing)
print("Open (Empty WM; does nothing)")
print()

d = {feature(("wm", "r", 1), "open"): 1.0}

stimulus.emitter.input(d)
alice.step()

print("Step 1: {} ->".format(wm.emitter.controller))
pprint.pprint(alice[wm.emitter.controller].output)
print()

alice.step()

print("Step 2: {} ->".format(buffer("wm")))
pprint.pprint(alice.output[buffer("wm")])
print()

print("Current WM state.")
pprint.pprint([cell.store for cell in wm.emitter.cells])
print()


# single write
print("Single Write & Open Corresponding Slot")
print()

d = {
    feature("fruit", "dragon fruit"): 1.0,
    feature("price", "expensive"): 1.0,
    feature(("wm", "w", 0), "retrieve"): 1.0,
    feature(("wm", "r", 0), "open"): 1.0
}

stimulus.emitter.input(d)
alice.step()

print("Step 1: {} ->".format(wm.emitter.controller))
pprint.pprint(alice[wm.emitter.controller].output)
print()

alice.step()

print("Step 2: {} ->".format(buffer("wm")))
pprint.pprint(alice.output[buffer("wm")])
print()

print("Current WM state.")
pprint.pprint([cell.store for cell in wm.emitter.cells])
print()


# reset
print("Reset")
print()

d = {
    feature("fruit", "dragon fruit"): 1.0,
    feature("price", "expensive"): 1.0,
    feature("wm-reset", "release"): 1.0
}

stimulus.emitter.input(d)
alice.step()

print("Step 1: {} ->".format(wm.emitter.controller))
pprint.pprint(alice[wm.emitter.controller].output)
print()

alice.step()

print("Step 2: {} ->".format(buffer("wm")))
pprint.pprint(alice.output[buffer("wm")])
print()

print("Current WM state.")
pprint.pprint([cell.store for cell in wm.emitter.cells])
print()


# double write
print("Double Write")
print()

d = {
    feature("fruit", "banana"): 1.0,
    feature("price", "expensive"): 1.0,
    feature(("wm", "w", 0), "retrieve"): 1.0,
    feature(("wm", "r", 0), "open"): 1.0,
    feature(("wm", "w", 1), "extract"): 1.0,
    feature(("wm", "r", 1), "open"): 1.0
}

stimulus.emitter.input(d)
alice.step()

print("Step 1: {} ->".format(wm.emitter.controller))
pprint.pprint(alice[wm.emitter.controller].output)
print()

alice.step()

pprint.pprint([cell.store for cell in wm.emitter.cells])

print("Step 2: {} ->".format(buffer("wm")))
pprint.pprint(alice.output[buffer("wm")])
print()

print("Current WM state.")
pprint.pprint([cell.store for cell in wm.emitter.cells])
print()


# Open Slot 1, removing it
print("Open Slot 1 only, removing Slot 0 from output")
print()

d = {feature(("wm", "r", 1), "open"): 1.0}

stimulus.emitter.input(d)
alice.step()

print("Step 1: {} ->".format(wm.emitter.controller))
pprint.pprint(alice[wm.emitter.controller].output)
print()

alice.step()

print("Step 2: {} ->".format(buffer("wm")))
pprint.pprint(alice.output[buffer("wm")])
print()

print("Current WM state.")
pprint.pprint([cell.store for cell in wm.emitter.cells])
print()


# single delete
print("Single Delete (clear & open slot 0)")
print()

d = {
    feature(("wm", "w", 0), "clear"): 1.0,
    feature(("wm", "r", 0), "open"): 1.0
}

stimulus.emitter.input(d)
alice.step()

print("Step 1: {} ->".format(wm.emitter.controller))
pprint.pprint(alice[wm.emitter.controller].output)
print()

alice.step()

print("Step 2: {} ->".format(buffer("wm")))
pprint.pprint(alice.output[buffer("wm")])
print()

print("Current WM state.")
pprint.pprint([cell.store for cell in wm.emitter.cells])