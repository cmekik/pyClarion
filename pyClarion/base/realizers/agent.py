from abc import ABC, abstractmethod
from typing import Iterable
from pyClarion.base.symbols import Agent, Subsystem, Buffer
from pyClarion.base.realizers.abstract import ContainerConstructRealizer
from pyClarion.base.revisions import RevisionManager


class AgentRealizer(ContainerConstructRealizer):

    @abstractmethod
    def cycle(self) -> None:
        pass

    @abstractmethod
    def revise(self) -> None:
        pass

    @property
    @abstractmethod
    def specialists(self) -> Iterable[RevisionManager]:
        pass

    @property
    @abstractmethod
    def actions(self):
        pass
