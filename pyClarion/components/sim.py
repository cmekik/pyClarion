from ..system import Process
from ..knowledge import Family


class Environment(Process):
    """
    A simulation environment.
    
    Initializes shared keyspaces for multi-agent simulations.
    """

    def __init__(self, name: str, **families: Family) -> None:
        super().__init__(name)
        for name, family in families.items():
            self.system.root[name] = family


class Agent(Process):
    """
    A simulated agent.
    
    Initializes keyspaces specific to an agent.
    """

    def __init__(self, name: str, **families: Family) -> None:
        super().__init__(name)
        for name, family in families.items():
            self.system.root[name] = family