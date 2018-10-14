from dataclasses import dataclass
from pyClarion.base.symbols import Node, Flow, Appraisal, Behavior, Buffer
from pyClarion.base.utils import check_construct
from pyClarion.base.processors import Channel, Junction, Selector, Effector, Source
from pyClarion.base.realizers.abstract import BasicConstructRealizer


class NodeRealizer(BasicConstructRealizer[Node]):

    def __init__(self, construct: Node, junction: Junction) -> None:
        
        check_construct(construct, Node)
        super().__init__(construct)
        self.junction: Junction = junction
        self._init_io()

    def propagate(self) -> None:

        inputs = self.input.pull([self.construct])
        output = self.junction(*inputs)
        self.output.update(output)


class FlowRealizer(BasicConstructRealizer[Flow]):

    def __init__(
        self, construct: Flow, junction: Junction, channel: Channel
    ) -> None:
        
        check_construct(construct, Flow)
        super().__init__(construct)
        self.junction: Junction = junction
        self.channel: Channel = channel
        self._init_io()

    def propagate(self):

        inputs = self.input.pull()
        combined = self.junction(*inputs)
        output = self.channel(combined)
        self.output.update(output)


class AppraisalRealizer(BasicConstructRealizer[Appraisal]):

    def __init__(
        self, construct: Appraisal, junction: Junction, selector: Selector
    ) -> None:
        
        check_construct(construct, Appraisal)
        super().__init__(construct)
        self.junction: Junction = junction
        self.selector: Selector = selector
        self._init_io()

    def propagate(self):

        inputs = self.input.pull()
        combined = self.junction(*inputs)
        output = self.selector(combined)
        self.output.update(output)


class BufferRealizer(BasicConstructRealizer[Buffer]):

    def __init__(
        self, construct: Buffer, source: Source
    ) -> None:
        
        check_construct(construct, Buffer)
        super().__init__(construct)
        self.source: Source = source
        self._init_io(has_input=False)

    def propagate(self):

        output = self.source()
        self.output.update(output)


class BehaviorRealizer(BasicConstructRealizer[Behavior]):
    
    def __init__(
        self, construct: Behavior, effector: Effector
    ) -> None:
        
        check_construct(construct, Behavior)
        super().__init__(construct)
        self.effector: Effector = effector
        self._init_io(has_output=False)

    def propagate(self):

        input_ = self.input.pull()
        if input_:
            self.effector(*input_)