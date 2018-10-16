from abc import ABC, abstractmethod
from typing import List, cast
from pyClarion.base.symbols import Agent, Subsystem, Buffer
from pyClarion.base.realizers.abstract import ContainerConstructRealizer
from pyClarion.base.realizers.subsystem import SubsystemRealizer
from pyClarion.base.updates import UpdateManager
from pyClarion.base.utils import check_construct


class AgentRealizer(ContainerConstructRealizer):

    def __init__(self, construct: Agent) -> None:

        check_construct(construct, Agent)
        super().__init__(construct)
        self._update_managers : List[UpdateManager] = []

    def propagate(self) -> None:
        
        for construct, realizer in self.items():
            if isinstance(construct, Buffer):
                realizer.propagate()
        for construct, realizer in self.items():
            if isinstance(construct, Subsystem):
                realizer.propagate()

    def execute(self) -> None:

        for construct, realizer in self.items():
            if isinstance(construct, Subsystem):
                cast(SubsystemRealizer, realizer).execute()

    def learn(self) -> None:

        for update_manager in self.update_managers:
            update_manager.update()

    def attach(self, *update_managers: UpdateManager) -> None:

        for update_manager in update_managers:
            self._update_managers.append(update_manager)
            update_manager.assign(self.dict.keys(), self.get)

    @property
    def update_managers(self) -> List[UpdateManager]:
        
        return list(self._update_managers)
