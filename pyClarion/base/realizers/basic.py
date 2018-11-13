from dataclasses import dataclass
from pyClarion.base.symbols import Node, Flow, Appraisal, Behavior, Buffer
from pyClarion.base.packets import DefaultActivation
from pyClarion.base.utils import check_construct
from pyClarion.base.processors import Channel, Junction, Selector, Effector, Source
from pyClarion.base.realizers.abstract import BasicConstructRealizer


class NodeRealizer(BasicConstructRealizer[Node]):
    """Realizer for Node constructs."""

    def __init__(self, construct: Node, junction: Junction) -> None:
        """Initialize a new node realizer.

        :param construct: Client node.
        :param junction: Junction for combining or selecting output value 
            recommendations for client node.
        """
        
        check_construct(construct, Node)
        super().__init__(construct)
        self.junction: Junction = junction
        self._init_io()

    def propagate(self) -> None:

        inputs = self.input.pull([self.construct])
        output = self.junction(*inputs)
        self.output.update(output)


class FlowRealizer(BasicConstructRealizer[Flow]):
    """Realizer for Flow constructs."""

    def __init__(
        self, 
        construct: Flow, 
        junction: Junction, 
        channel: Channel, 
        default_activation: DefaultActivation
    ) -> None:
        """
        Initialize a new flow realizer.
        
        :param construct: Client flow.
        :param junction: Combines input packets.
        :param channel: Computes output.
        :param default_activation: Computes default outputs.
        """

        check_construct(construct, Flow)
        super().__init__(construct)
        self.junction: Junction = junction
        self.channel: Channel = channel
        self._init_io(default_activation=default_activation)

    def propagate(self):

        inputs = self.input.pull()
        combined = self.junction(*inputs)
        output = self.channel(combined)
        self.output.update(output)


class AppraisalRealizer(BasicConstructRealizer[Appraisal]):
    """Realizer for Appraisal constructs."""

    def __init__(
        self, construct: Appraisal, junction: Junction, selector: Selector
    ) -> None:
        """
        Initialize a new appraisal realizer.
        
        :param construct: Client appraisal.
        :param junction: Combines incoming input packets.
        :param selector: Computes output.
        """

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
    """Realizer for Buffer constructs."""

    def __init__(
        self, 
        construct: Buffer, 
        source: Source, 
        default_activation: DefaultActivation
    ) -> None:
        """
        Initialize a new buffer realizer.
        
        :param construct: Client buffer.
        :param source: Computes output.
        :param default_activation: Computes default output.
        """

        check_construct(construct, Buffer)
        super().__init__(construct)
        self.source: Source = source
        self._init_io(has_input=False, default_activation=default_activation)

    def propagate(self):

        output = self.source()
        self.output.update(output)


class BehaviorRealizer(BasicConstructRealizer[Behavior]):
    """Realizer for a Behavior construct."""
    
    def __init__(
        self, construct: Behavior, effector: Effector
    ) -> None:
        """
        Initialize a new behavior realizer.
        
        :param construct: Client behavior.
        :param effector: Executes action callbacks.
        """
        
        check_construct(construct, Behavior)
        super().__init__(construct)
        self.effector: Effector = effector
        self._init_io(has_output=False)

    def propagate(self):

        input_ = self.input.pull()
        self.effector(*input_)
