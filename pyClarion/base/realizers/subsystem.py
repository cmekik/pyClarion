from abc import abstractmethod
from typing import Dict, Any, Iterator, Iterable, Type, Optional, Hashable
from pyClarion.base.symbols import Node, Flow, Appraisal, Behavior, Subsystem
from pyClarion.base.utils import check_construct, may_connect
from pyClarion.base.packets import ActivationPacket
from pyClarion.base.realizers.abstract import ContainerConstructRealizer
from pyClarion.base.realizers.basic import AppraisalRealizer, BehaviorRealizer
from pyClarion.base.links import (
    SubsystemInputMonitor, SubsystemOutputView, PullMethod
)


class MissingAppraisalError(AttributeError):
    pass


class MissingBehaviorError(AttributeError):
    pass


class SubsystemRealizer(ContainerConstructRealizer[Subsystem]):
    """A network of interconnected nodes and flows."""

    def __init__(self, construct: Subsystem) -> None:

        check_construct(construct, Subsystem)
        super().__init__(construct)
        self._appraisal: Optional[Appraisal] = None
        self._behavior: Optional[Behavior] = None
        self.input = SubsystemInputMonitor(self._watch, self._drop)
        self.output = SubsystemOutputView(self._view)        

    def __setitem__(self, key: Any, value: Any) -> None:

        if self._appraisal and isinstance(key, Appraisal):
            raise Exception("Appraisal already set")
        if self._behavior and isinstance(key, Behavior):
            raise Exception("Behavior already set")

        super().__setitem__(key, value)

        for construct, realizer in self.dict.items():
            if may_connect(source=construct, target=key):
                value.input.watch(construct, realizer.output.view)
            if may_connect(source=key, target=construct):
                realizer.input.watch(key, value.output.view)
        if isinstance(key, Node):
            for buffer, pull_method in self.input.input_links.items():
                value.input.watch(buffer, pull_method)
        elif isinstance(key, Appraisal):
            self._appraisal = key
        elif isinstance(key, Behavior):
            self._behavior = key

    def __delitem__(self, key: Any) -> None:

        super().__delitem__(key)

        for construct, realizer in self.dict.items():
            if may_connect(key, construct):
                realizer.input.drop(key)
        if isinstance(key, Appraisal):
            self._appraisal = None
        elif isinstance(key, Behavior):
            self._behavior = None

    def _watch(self, identifier: Hashable, pull_method: PullMethod) -> None:

        for node in self.nodes:
            self[node].input.watch(identifier, pull_method)

    def _drop(self, identifier: Hashable) -> None:

        for node in self.nodes:
            self[node].input.drop(identifier)

    def _view(self, keys: Iterable[Node] = None) -> ActivationPacket:

        return self.appraisal.output.view(keys)

    def execute(self) -> None:
        try:
            self.behavior.propagate()
        except MissingBehaviorError:
            pass

    @property
    def nodes(self) -> Iterable[Node]:
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, Node)
        }

    @property
    def flows(self) -> Iterable[Flow]:
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, Flow)
        }

    @property
    def appraisal(self) -> AppraisalRealizer:
        
        if self._appraisal:
            return self.__getitem__(self._appraisal)
        else:
            raise MissingAppraisalError()

    @property
    def behavior(self) -> BehaviorRealizer:
        
        if self._behavior:
            return self.__getitem__(self._behavior)
        else:
            raise MissingBehaviorError()
