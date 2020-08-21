"""Demonstrates selection & control of reasoning methods."""


from pyClarion import *
import pprint


alice = Structure(
    name=agent("Alice"),
    emitter=AgentCycle(),
    assets=Assets(chunks=Chunks())
)

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
            dims=("nacs-stim", "nacs-assoc", "nacs-bt"),
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
alice.add(stimulus, gate, defaults)

acs = Structure(
    name=subsystem("ACS"),
    emitter=ACSCycle(
        matches=MatchSet(
            constructs={buffer("stimulus"), buffer("defaults")}
        )
    )
)
alice.add(acs)

acs.add(
    Construct(
        name=terminus("NACS"),
        emitter=ActionSelector(
            dims=gate.emitter.interface.dims,
            temperature=0.01
        )
    )
)

fnodes = [
    Construct(
        name=f, 
        emitter=MaxNode(
            matches=MatchSet(ctype=ConstructType.buffer)
        )
    ) 
    for f in gate.emitter.interface.features
]
acs.add(*fnodes)

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
alice.add(nacs)

nacs.add(
    Construct(
        name=flow_in("stimulus"),
        emitter=GatedA(
            base=Repeater(source=buffer("stimulus")),
            gate=buffer("flow-gate")
        )
    ),
    Construct(
        name=flow_tt("associations"),
        emitter=GatedA(
            base=AssociativeRules(rules=nacs.assets.rules),
            gate=buffer("flow-gate")
        ) 
    ),
    Construct(
        name=flow_bt("main"), 
        emitter=GatedA(
            base=BottomUp(chunks=alice.assets.chunks),
            gate=buffer("flow-gate") 
        )
    ),
    Construct(
        name=flow_tb("main"), 
        emitter=TopDown(chunks=alice.assets.chunks) 
    ),
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
)

fnodes = [
    Construct(
        name=feature(dim, val), 
        emitter=MaxNode(
            matches=MatchSet(
                ctype=ConstructType.flow_xb,
                constructs={flow_in("stimulus")}
            )
        )
    ) for dim, val in [
        ("color", "#ff0000"), # red
        ("color", "#008000"), # green
        ("tasty", True),
        ("state", "liquid"),
        ("sweet", True)
    ]
]
cnodes = [
    Construct(
        name=chunk(name),
        emitter=MaxNode(
            matches=MatchSet(
                ctype=ConstructType.flow_xt,
                constructs={flow_in("stimulus")}
            )
        )
    ) for name in ["FRUIT", "APPLE", "JUICE"]
]

nacs.add(*(fnodes + cnodes))

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

alice.propagate(
    kwds={
        buffer("stimulus"): {"stimulus": {feature("nacs-stimulus", 1.0): 1.0}}
    }
)
alice.update()
pprint.pprint(alice.output)

alice.propagate(kwds={buffer("stimulus"): {"stimulus": {chunk("APPLE"): 1.}}})
alice.update()
pprint.pprint(alice.output)


print("CYCLE 2: Open stimulus & associations only.")

alice.propagate(
    kwds={
        buffer("stimulus"): {
            "stimulus": {
                feature("nacs-stimulus", 1.): 1.,
                feature("nacs-associations", 1.): 1.
            }
        }
    }
)
alice.update()
pprint.pprint(alice.output)

alice.propagate(kwds={buffer("stimulus"): {"stimulus": {chunk("APPLE"): 1.}}})
alice.update()
pprint.pprint(alice.output)


print("CYCLE 3: Open stimulus & bottom-up only.")

alice.propagate(
    kwds={
        buffer("stimulus"): {
            "stimulus": {
                feature("nacs-stimulus", 1.): 1.,
                feature("nacs-bt", 1.): 1.
            }
        }
    }
)
alice.update()
pprint.pprint(alice.output)

alice.propagate(kwds={buffer("stimulus"): {"stimulus": {chunk("APPLE"): 1.}}})
alice.update()
pprint.pprint(alice.output)
