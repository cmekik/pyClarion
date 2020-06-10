from pyClarion.base.realizers import Agent, Subsystem, MatchArg, UpdaterArg
from pyClarion.components.assets import BasicAgentAssets, NACSAssets
from pyClarion.components.cycles import NACSCycle
from typing import TypeVar


class BasicAgent(Agent[BasicAgentAssets]):

    _CRt = TypeVar("_CRt", bound="BasicAgent")
    
    def __init__(
        self: _CRt, 
        name: str, 
        matches: MatchArg = None, 
        assets: BasicAgentAssets = None, 
        updaters: UpdaterArg[_CRt] = None
    ) -> None:

        super().__init__(
            name=name, 
            matches=matches, 
            assets=assets if assets is not None else BasicAgentAssets(), 
            updaters=updaters
        )


class NACS(Subsystem[NACSAssets]):

    _CRt = TypeVar("_CRt", bound="NACS")

    def __init__(
        self: _CRt, 
        matches: MatchArg = None, 
        assets: NACSAssets = None, 
        updaters: UpdaterArg[_CRt] =None
    ) -> None:

        super().__init__(
            name="NACS", 
            matches=matches, 
            propagator=NACSCycle(), 
            assets=assets if assets is not None else NACSAssets(), 
            updaters=updaters
        )
