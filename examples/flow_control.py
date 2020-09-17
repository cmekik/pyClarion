"""Demonstrates selection & control of reasoning methods."""


from pyClarion import *
import pprint


alice = Structure(
    name=agent("Alice"),
    emitter=AgentCycle(),
    assets=Assets(chunks=Chunks())
)

with alice:

    stimulus = Construct(
        name=buffer("stimulus"), 
        emitter=Stimulus()
    )

    gate = Construct(
        name=buffer("flow-gate"),
        emitter=FilteringRelay(
            controller=(subsystem("ACS"), terminus("NACS")),
            interface=FilteringRelay.Interface(
                clients=(
                    flow_in("stimulus"), 
                    flow_tt("associations"), 
                    flow_bt("main")
                ),
                dlbs=("nacs-stim", "nacs-assoc", "nacs-bt"),
                vals=(0, 1)
            )
        )
    )

    defaults = Construct(
        name=buffer("defaults"),
        emitter=ConstantBuffer(
            strengths={f: 0.5 for f in gate.emitter.interface.defaults}
        )
    )


    acs = Structure(
        name=subsystem("ACS"),
        emitter=ACSCycle(
            matches=MatchSet(
                constructs={buffer("stimulus"), buffer("defaults")}
            )
        )
    )

    with acs:

        Construct(
            name=terminus("NACS"),
            emitter=ActionSelector(
                dims=gate.emitter.interface.dims,
                temperature=0.01
            )
        )

        for f in gate.emitter.interface.features:
            Construct(
                name=f, 
                emitter=MaxNode(
                    matches=MatchSet(ctype=ConstructType.buffer)
                )
            ) 

    nacs = Structure(
        name=subsystem("NACS"),
        emitter=NACSCycle(        
            matches=MatchSet(
                constructs={
                    buffer("stimulus"), buffer("defaults"), buffer("flow-gate")
                }
            )
        ),
        assets=Assets(rules=Rules())
    )

    with nacs:

        Construct(
            name=flow_in("stimulus"),
            emitter=GatedA(
                base=Repeater(source=buffer("stimulus")),
                gate=buffer("flow-gate")
            )
        )

        Construct(
            name=flow_tt("associations"),
            emitter=GatedA(
                base=AssociativeRules(rules=nacs.assets.rules),
                gate=buffer("flow-gate")
            ) 
        )

        Construct(
            name=flow_bt("main"), 
            emitter=GatedA(
                base=BottomUp(chunks=alice.assets.chunks),
                gate=buffer("flow-gate") 
            )
        )

        Construct(
            name=flow_tb("main"), 
            emitter=TopDown(chunks=alice.assets.chunks) 
        )

        Construct(
            name=terminus("retrieval"),
            emitter=FilteredT(
                base=BoltzmannSelector(
                    temperature=.1,
                    matches=MatchSet(ctype=ConstructType.chunk)
                ),
                filter=flow_in("stimulus")
            )
        )

        fspecs = [
            ("color", "#ff0000"), # red
            ("color", "#008000"), # green
            ("tasty", True),
            ("state", "liquid"),
            ("sweet", True)
        ]

        for dlb, val in fspecs:
            Construct(
                name=feature(dlb, val), 
                emitter=MaxNode(
                    matches=MatchSet(
                        ctype=ConstructType.flow_xb,
                        constructs={flow_in("stimulus")}
                    )
                )
            ) 

        cnames = ["FRUIT", "APPLE", "JUICE"]

        for cname in cnames:
            Construct(
                name=chunk(cname),
                emitter=MaxNode(
                    matches=MatchSet(
                        ctype=ConstructType.flow_xt,
                        constructs={flow_in("stimulus")}
                    )
                )
            ) 

nacs.assets.rules.link(chunk("FRUIT"), chunk("APPLE")) # type: ignore

alice.assets.chunks.link( # type: ignore
    chunk("APPLE"), 
    feature("color", "#ff0000"), 
    feature("color", "#008000"),
    feature("tasty", True)
)

alice.assets.chunks.link( # type: ignore
    chunk("JUICE"),
    feature("tasty", True),
    feature("state", "liquid")
)

alice.assets.chunks.link( # type: ignore
    chunk("FRUIT"),
    feature("tasty", True),
    feature("sweet", True)
)


##################
### Simulation ###
##################

print("CYCLE 1: Open stimulus only.") 

stimulus.emitter.input({feature("nacs-stim", 1.0): 1.0})
alice.propagate()
alice.update()
print(
    "Step 1: {} -> {}".format(
        gate.emitter.controller, 
        alice[gate.emitter.controller].output
    )
)

alice.propagate(kwds={buffer("stimulus"): {"stimulus": {chunk("APPLE"): 1.}}})
alice.update()
print("Step 2: {} -> {}\n".format(subsystem("NACS"), alice[subsystem("NACS")].output))


print("CYCLE 2: Open stimulus & associations only.")

alice.propagate(
    kwds={
        buffer("stimulus"): {
            "stimulus": {
                feature("nacs-stim", 1.): 1.,
                feature("nacs-assoc", 1.): 1.
            }
        }
    }
)
alice.update()
print(
    "Step 1: {} -> {}".format(
        gate.emitter.controller, 
        alice[gate.emitter.controller].output
    )
)

alice.propagate(kwds={buffer("stimulus"): {"stimulus": {chunk("APPLE"): 1.}}})
alice.update()
print("Step 2: {} -> {}\n".format(subsystem("NACS"), alice[subsystem("NACS")].output))


print("CYCLE 3: Open stimulus & bottom-up only.")

alice.propagate(
    kwds={
        buffer("stimulus"): {
            "stimulus": {
                feature("nacs-stim", 1.): 1.,
                feature("nacs-bt", 1.): 1.
            }
        }
    }
)
alice.update()
print(
    "Step 1: {} -> {}".format(
        gate.emitter.controller, 
        alice[gate.emitter.controller].output
    )
)

alice.propagate(kwds={buffer("stimulus"): {"stimulus": {chunk("APPLE"): 1.}}})
alice.update()
print("Step 2: {} -> {}\n".format(subsystem("NACS"), alice[subsystem("NACS")].output))
