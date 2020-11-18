"""Demonstrates selection & control of reasoning methods."""


from pyClarion import *
import pprint


gate_interface = FilteringRelay.Interface(
    mapping={
        "nacs-stim": flow_in("stimulus"),
        "nacs-assoc": flow_tt("associations"),
        "nacs-bt": flow_bt("main")
    },
    vals=(0, 1)
)

chunk_db = Chunks()
rule_db = Rules()

rule_db.link(chunk("FRUIT"), chunk("APPLE")) # type: ignore

chunk_db.link( # type: ignore
    chunk("APPLE"), 
    feature("color", "#ff0000"), 
    feature("color", "#008000"),
    feature("tasty", True)
)

chunk_db.link( # type: ignore
    chunk("JUICE"),
    feature("tasty", True),
    feature("state", "liquid")
)

chunk_db.link( # type: ignore
    chunk("FRUIT"),
    feature("tasty", True),
    feature("sweet", True)
)


alice = Structure(
    name=agent("Alice"),
    emitter=AgentCycle(),
    assets=Assets(
        gate_interface=gate_interface
    )
)

with alice:

    stimulus = Construct(
        name=buffer("stimulus"), 
        emitter=Stimulus()
    )

    gate = Construct(
        name=buffer("gate"),
        emitter=FilteringRelay(
            controller=(subsystem("acs"), terminus("nacs")),
            interface=alice.assets.gate_interface
        )
    )

    defaults = Construct(
        name=buffer("defaults"),
        emitter=Constants(
            strengths={f: 0.5 for f in alice.assets.gate_interface.defaults}
        )
    )

    acs = Structure(
        name=subsystem("acs"),
        emitter=ACSCycle(
            sources={
                buffer("stimulus"), 
                buffer("defaults")
            }
        )
    )

    with acs:

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={
                    buffer("stimulus"), 
                    buffer("defaults")
                }
            )
        )

        Construct(
            name=terminus("nacs"),
            emitter=ActionSelector(
                source=features("main"),
                client_interface=alice.assets.gate_interface,
                temperature=0.01
            )
        )

    nacs = Structure(
        name=subsystem("nacs"),
        emitter=NACSCycle(       
            sources={
                buffer("stimulus"), 
                buffer("gate")
            }
        ),
        assets=Assets(
            chunk_db=chunk_db, 
            rule_db=rule_db
        )
    )

    with nacs:

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={
                    flow_tb("main")
                }
            )
        )

        Construct(
            name=chunks("main"),
            emitter=MaxNodes(
                sources={
                    buffer("stimulus"), 
                    flow_bt("main"), 
                    flow_tt("associations")
                }
            )
        )

        Construct(
            name=flow_in("stimulus"),
            emitter=Gated(
                base=Repeater(source=buffer("stimulus")),
                gate=buffer("gate")
            )
        )

        Construct(
            name=flow_tt("associations"),
            emitter=Gated(
                base=AssociativeRules(
                    source=chunks("main"),
                    rules=nacs.assets.rule_db
                ),
                gate=buffer("gate")
            ) 
        )

        Construct(
            name=flow_bt("main"), 
            emitter=Gated(
                base=BottomUp(
                    source=features("main"),
                    chunks=nacs.assets.chunk_db
                ),
                gate=buffer("gate") 
            )
        )

        Construct(
            name=flow_tb("main"), 
            emitter=TopDown(
                source=chunks("main"),
                chunks=nacs.assets.chunk_db
            ) 
        )

        Construct(
            name=terminus("retrieval"),
            emitter=Filtered(
                base=BoltzmannSelector(
                    source=chunks("main"),
                    temperature=.1
                ),
                sieve=flow_in("stimulus")
            )
        )


##################
### Simulation ###
##################

alice.start()

print("CYCLE 1: Open stimulus only.") 

stimulus.emitter.input({feature("nacs-stim", 1.0): 1.0})
alice.step()
print(
    "Step 1: {} -> {}".format(
        gate.emitter.controller, 
        alice[gate.emitter.controller].output
    )
)

stimulus.emitter.input({chunk("APPLE"): 1.})
alice.step()
print("Step 2: {} ->".format(subsystem("nacs")))
pprint.pprint(alice[subsystem("nacs")].output)
print()


print("CYCLE 2: Open stimulus & associations only.")

stimulus.emitter.input({
    feature("nacs-stim", 1.): 1., 
    feature("nacs-assoc", 1.): 1.
})
alice.step()
print(
    "Step 1: {} -> {}".format(
        gate.emitter.controller, 
        alice[gate.emitter.controller].output
    )
)

stimulus.emitter.input({chunk("APPLE"): 1.})
alice.step()
print("Step 2: {} ->".format(subsystem("nacs")))
pprint.pprint(alice[subsystem("nacs")].output)
print()

print("CYCLE 3: Open stimulus & bottom-up only.")

stimulus.emitter.input({
    feature("nacs-stim", 1.): 1.,
    feature("nacs-bt", 1.): 1.
})
alice.step()
print(
    "Step 1: {} -> {}".format(
        gate.emitter.controller, 
        alice[gate.emitter.controller].output
    )
)

stimulus.emitter.input({chunk("APPLE"): 1.})
alice.step()
print("Step 2: {} ->".format(subsystem("nacs")))
pprint.pprint(alice[subsystem("nacs")].output)
print()
