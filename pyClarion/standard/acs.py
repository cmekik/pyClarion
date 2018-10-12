from pyClarion.base.processors import Channel
from pyClarion.base.realizers.subsystem import SubsystemRealizer
from pyClarion.base.symbols import Chunk, Microfeature
from pyClarion.base.packets import ActivationPacket

class RuleCollection(object):
    """A collection of rules used by AssociativeRuleChannel and descendants.
    """
    pass


class InterlevelAssociation(object):
    """Represents connections between chunk nodes and their respective 
    microfeature nodes.
    """
    pass


class AssociativeRulesChannel(Channel[float]):
    """
    Represents associative rules known to an agent.

    Contains rules that have the form
        condition_chunk -> action_chunk

    Only one condition chunk per rule.

    Generally speaking, activation of action_chunk is equal to some weight 
    multiplied by the activation of the conditon chunk.

    May need to define and enforce a format for representing rules.

    More advanced features:
        - Rule BLAs
        - Subtypes: fixed, irl, rer
            Subtypes may have learning rules and dedicated statistics associated 
            with them; methods for updating knowledge base according to learning 
            rules and stats should be implemented within the class. 
    """
    pass


class BottomUpChannel(Channel[float]):
    """Activates condition chunks on the basis of microfeature activations.
    """
    pass


class QNetChannel(Channel[float]):
    """A simple q-net based channel activating action chunks based on 
    microfeature activations.

    Should have a learn method. See below.
    """
    
    def learn(self, inputs, reinforcement):
        """Update q-net using given input and reinforcement values."""
        pass


class TopDownChannel(Channel[float]):
    """Not sure what this does. Ignore for now.
    """

    pass


class ACSRealizer(SubsystemRealizer):
    pass


if __name__ == "__main__":
    """
    All weights == 1:

    Chunk("POMEGRANATE") -> Chunk("EAT")
    Chunk("APRICOT") -> Chunk("PIT")
    Chunk("APPLE") -> Chunk("PIT")
    Chunk("BANANA") -> Chunk("PEEL")
    """

    inputs = [
        ActivationPacket(
            {
                Chunk("POMEGRANATE"): .3,
                Chunk("APRICOT"): .9,
                Chunk("APPLE"): .0,
                Chunk("BANANA"): .2
            }
        ),
        ActivationPacket(
            {
                Chunk("POMEGRANATE"): .5,
                Chunk("APPLE"): .7,
                Chunk("BANANA"): .0
            }
        )
    ]

    outputs = [
        ActivationPacket(
            {
                Chunk("EAT"): .3,
                Chunk("PIT"): .9,
                Chunk("PEEL"): .2,
            }
        ),
        ActivationPacket(
            {
                Chunk("EAT"): .5,
                Chunk("PIT"): .7,
                Chunk("PEEL"): .0,
            }
        )
    ]

    rule_channel: AssociativeRulesChannel = AssociativeRulesChannel(...)
    for ipt, opt in zip(inputs, outputs):
        rule_channel(ipt) == opt

