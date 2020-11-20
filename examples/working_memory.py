from pyClarion import (
    feature, chunk, terminus, features, chunks, buffer, subsystem, agent, 
    flow_tb, flow_bt,
    Construct, Structure,
    AgentCycle, ACSCycle, NACSCycle,
    WorkingMemory, Stimulus, Constants, TopDown, BottomUp, MaxNodes,
    Filtered, ActionSelector, BoltzmannSelector, ThresholdSelector,
    Assets, Chunks, ChunkAdder,
    pprint
)

import logging


logging.basicConfig(level=logging.INFO)


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
    reset_vals=("standby", "reset"),
    read_vals=("standby", "read"),
)


nacs_cdb = Chunks()

nacs_cdb.link(
    chunk("APPLE"),
    feature("lexeme", "/apple/"),
    feature("color", "red"),
    feature("shape", "round"),
    feature("size", "medium"),
    feature("texture", "smooth")
)

nacs_cdb.link(
    chunk("ORANGE"),
    feature("lexeme", "/orange/"),
    feature("color", "orange"),
    feature("shape", "round"),
    feature("size", "medium"),
    feature("texture", "grainy")
)

nacs_cdb.link(
    chunk("BANANA"),
    feature("lexeme", "/banana/"),
    feature("color", "yellow"),
    feature("shape", "oblong"),
    feature("size", "medium"),
    feature("texture", "spotty")
)

nacs_cdb.link(
    chunk("PLUM"),
    feature("lexeme", "/plum/"),
    feature("color", "purple"),
    feature("shape", "round"),
    feature("size", "small"),
    feature("texture", "smooth")
) 


alice = Structure(
    name=agent("alice"),
    emitter=AgentCycle(),
    assets=Assets(
        chunks=nacs_cdb,
        wm_interface=wm_interface
    )
)

with alice:

    stimulus = Construct(
        name=buffer("stimulus"), 
        emitter=Stimulus()
    )

    acs_ctrl = Construct(
        name=buffer("acs_ctrl"), 
        emitter=Stimulus()
    )

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
                buffer("acs_ctrl"), 
                buffer("wm"), 
                buffer("wm-defaults")
            }
        )
    )

    with acs:

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={
                    buffer("acs_ctrl"), 
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
                buffer("wm"),
                buffer("stimulus")
            }
        )
    )

    with nacs:

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={
                    buffer("stimulus"), 
                    buffer("wm"),
                    flow_tb("main")
                }
            )
        )

        Construct(
            name=chunks("in"),
            emitter=MaxNodes(
                sources={
                    buffer("stimulus"),
                    buffer("wm")
                }
            )
        )

        Construct(
            name=chunks("out"),
            emitter=MaxNodes(
                sources={
                    chunks("in"), 
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
                source=chunks("in"),
                chunks=alice.assets.chunks
            )
        )

        Construct(
            name=terminus("retrieval"),
            emitter=Filtered(
                base=BoltzmannSelector(
                    source=chunks("out"),
                    temperature=.1
                ),
                sieve=buffer("wm")
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

stimulus.emitter.input({
    feature("color", "green"): 1.0,
    feature("shape", "round"): 1.0,
    feature("size", "medium"): 1.0,
    feature("texture", "smooth"): 1.0
})
acs_ctrl.emitter.input({})
alice.step()

print("Step 1:") 
print("{}:".format(wm.emitter.controller))
pprint(alice[wm.emitter.controller].output)
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

alice.step()

print("Step 2:")
print("{}:".format(buffer("wm")))
pprint(alice.output[buffer("wm")])
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

print("Current WM state.")
pprint([cell.store for cell in wm.emitter.cells])
print()


# open empty (should do nothing)
print("Open (Empty WM; does nothing)")
print()

stimulus.emitter.input({})
acs_ctrl.emitter.input({
    feature(("wm", "r", 1), "read"): 1.0
})
alice.step()

print("Step 1:") 
print("{}:".format(wm.emitter.controller))
pprint(alice[wm.emitter.controller].output)
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

alice.step()

print("Step 2:")
print("{}:".format(buffer("wm")))
pprint(alice.output[buffer("wm")])
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

print("Current WM state.")
pprint([cell.store for cell in wm.emitter.cells])
print()


# single write
print("Single Write & Open Corresponding Slot")
print()

stimulus.emitter.input({
    feature("color", "green"): 1.0,
    feature("shape", "round"): 1.0,
    feature("size", "medium"): 1.0,
    feature("texture", "smooth"): 1.0
})
acs_ctrl.emitter.input({
    feature(("wm", "w", 0), "retrieve"): 1.0,
    feature(("wm", "r", 0), "read"): 1.0
})
alice.step()

print("Step 1:") 
print("{}:".format(wm.emitter.controller))
pprint(alice[wm.emitter.controller].output)
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

alice.step()

print("Step 2:")
print("{}:".format(buffer("wm")))
pprint(alice.output[buffer("wm")])
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

print("Current WM state.")
pprint([cell.store for cell in wm.emitter.cells])
print()


# reset
print("Reset")
print()

stimulus.emitter.input({
    feature("color", "green"): 1.0,
    feature("shape", "round"): 1.0,
    feature("size", "medium"): 1.0,
    feature("texture", "smooth"): 1.0
})
acs_ctrl.emitter.input({
    feature(("wm", "re"), "reset"): 1.0
})
alice.step()

print("Step 1:") 
print("{}:".format(wm.emitter.controller))
pprint(alice[wm.emitter.controller].output)
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

alice.step()

print("Step 2:")
print("{}:".format(buffer("wm")))
pprint(alice.output[buffer("wm")])
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

print("Current WM state.")
pprint([cell.store for cell in wm.emitter.cells])
print()


# double write
print("Double Write")
print()

stimulus.emitter.input({
    feature("color", "green"): 1.0,
    feature("shape", "oblong"): 1.0,
    feature("size", "medium"): 1.0,
    feature("texture", "smooth"): 1.0
})
acs_ctrl.emitter.input({
    feature(("wm", "w", 0), "retrieve"): 1.0,
    feature(("wm", "r", 0), "read"): 1.0,
    feature(("wm", "w", 1), "extract"): 1.0,
    feature(("wm", "r", 1), "read"): 1.0
})
alice.step()

print("Step 1:") 
print("{}:".format(wm.emitter.controller))
pprint(alice[wm.emitter.controller].output)
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

alice.step()

print("Step 2:")
print("{}:".format(buffer("wm")))
pprint(alice.output[buffer("wm")])
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

print("Current WM state.")
pprint([cell.store for cell in wm.emitter.cells])
print()


# Open Slot 1, removing it
print("Open Slot 1 only, removing Slot 0 from output")
print()

stimulus.emitter.input({})
acs_ctrl.emitter.input({
    feature(("wm", "r", 1), "read"): 1.0
})
alice.step()

print("Step 1:") 
print("{}:".format(wm.emitter.controller))
pprint(alice[wm.emitter.controller].output)
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

alice.step()

print("Step 2:")
print("{}:".format(buffer("wm")))
pprint(alice.output[buffer("wm")])
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

print("Current WM state.")
pprint([cell.store for cell in wm.emitter.cells])
print()


# single delete
print("Single Delete (clear & open slot 0)")
print()

stimulus.emitter.input({})
acs_ctrl.emitter.input({
    feature(("wm", "w", 0), "clear"): 1.0,
    feature(("wm", "r", 0), "read"): 1.0
})
alice.step()

print("Step 1:") 
print("{}:".format(wm.emitter.controller))
pprint(alice[wm.emitter.controller].output)
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

alice.step()

print("Step 2:")
print("{}:".format(buffer("wm")))
pprint(alice.output[buffer("wm")])
print("{}:".format(subsystem("nacs")))
pprint(alice.output[subsystem("nacs")])
print()

print("Current WM state.")
pprint([cell.store for cell in wm.emitter.cells])
