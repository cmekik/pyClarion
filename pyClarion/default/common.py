"""This module defines commonly used constructs in the default implementation 
of Clarion.

Classes/functions defined here are used by at least two subsystems.
"""

from ..base import node
from ..base import activation
from ..base import subject

import numpy as np
import typing as T
import numbers as nums
import collections


class Chunk(node.Chunk):
    """A default Clarion chunk.

    Weights are initialized to 1 by default.
    """

    def __init__(
        self,
        microfeatures: node.FeatureSet,
        dim2weight: node.Dim2Num = None,
        label: str = None
    ) -> None:
        """Initialize a default Clarion chunk.
        """

        if dim2weight is None:
            dim2weight = self.initialize_weights(microfeatures)
        super().__init__(microfeatures, dim2weight, label)

    @staticmethod
    def initialize_weights(microfeatures : node.FeatureSet) -> node.Dim2Num:
        """Initialize top-down weights.
        If input is None, weights are initialized to 1.0.
        Args:
            dim2weight: A mapping from each chunk dimension to its top-down 
                weight.
        """
        
        dim2weight : node.Dim2Num = dict()
        for f in microfeatures:
            dim2weight[f.dim] = 1.
        return dim2weight


class ActivationMap(activation.ActivationMap):
    """Default Clarion Activation Dict.
    """

    @staticmethod
    def default_activation() -> float:
        """Default activation is 0.
        """

        return 0.


class TopDownActivationMap(ActivationMap, activation.TopDownActivationMap):
    """Represents result of a top-down activation flow.
    """
    pass


class BottomUpActivationMap(ActivationMap, activation.BottomUpActivationMap):
    """Represents result of a bottom-up activation flow.
    """
    pass


class TopLevelActivationMap(ActivationMap, activation.TopLevelActivationMap):
    """Represents result of a top-level activation flow.
    """
    pass


class BottomLevelActivationMap(
    ActivationMap, activation.BottomLevelActivationMap
):
    """Represents result of a bottom-level activation flow.
    """
    pass


class TopDown(activation.TopDown):
    """A default Clarion top-down (from a chunk to its microfeatures) 
    activation channel.

    For details, see Section 3.2.2.3 para 2 of Sun (2016).

    References:
        Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(self, chunk : node.Chunk) -> None:
        """Set up the top-down activation flow from chunk to its microfeatures.

        kwargs:
            chunk : Source chunk for top-down activation.
        """
        
        self.chunk = chunk
        self.val_counts = self.count_values(self.chunk.microfeatures)

    def __call__(
        self, input_map : activation.ActivationMap
    ) -> activation.TopDownActivationMap:
        """Compute bottom-level activations given current chunk strength.

        Note: If the expected chunk is missing from the input, this channel 
        outputs an empty mapping.

        kwargs:
            input_map : A mapping from nodes (Chunks and/or Features) to 
            input activations.
        """


        strength = input_map[self.chunk]
        activations = TopDownActivationMap()

        for f in self.chunk.microfeatures:
            w_dim = self.chunk.dim2weight[f.dim] 
            n_val = self.val_counts[f.dim]
            activations[f] = strength * w_dim / n_val        

        return activations

    @staticmethod
    def count_values(microfeatures: node.FeatureSet) -> node.Dim2Num:
        """Count the number of features in each dimension
        
        kwargs:
            microfeatures : A set of dv-pairs.
        """
        
        counts : T.DefaultDict[T.Hashable, int] = collections.defaultdict(
            lambda: 0
        )
        for f in microfeatures:
            counts[f.dim] += 1
        return counts


class BottomUp(activation.BottomUp):
    """A default Clario bottom-up (from microfeatures to a chunk) activation 
    channel.

    For details, see Section 3.2.2.3 para 3 of Sun (2016).

    References:
        Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(self, chunk : node.Chunk) -> None:
        """Set up the bottom-up activation flow to chunk from its microfeatures.

        kwargs:
            chunk : Target chunk for bottom-up activation.
        """
        
        self.chunk = chunk

    def __call__(
        self, input_map : activation.ActivationMap
    ) -> activation.BottomUpActivationMap:
        """Compute chunk strength given current bottom-level activations.

        Note: If an expected (micro)feature is missing from the input, its 
        activation is assumed to be 0.

        kwargs:
            input_map : A mapping from nodes (Chunks and/or Features) to 
            input activations.
        """

        # For each dimension, pick the maximum available activation of the 
        # target chunk microfeatures in that dimension. 
        dim2activation : node.Dim2Num = collections.defaultdict(
            ActivationMap.default_activation
        )
        for f in self.chunk.microfeatures:
            if dim2activation[f.dim] < input_map[f]:
                dim2activation[f.dim] = input_map[f]

        # Compute chunk strength based on dimensional activations.
        strength = 0.
        for dim in dim2activation:
            strength += self.chunk.dim2weight[dim] * dim2activation[dim]
        else:
            strength /= sum(self.chunk.dim2weight.values()) ** 1.1
        return BottomUpActivationMap({self.chunk: strength})


class Rule(activation.TopLevel):
    """An basic Clarion associative rule.

    Rules have the form:
        chunk_1 chunk_2 chunk_3 -> chunk_4
    Chunks in the left-hand side are condition chunks, the single chunk in the 
    right-hand side is the conclusion chunk. The strength of the conclusion 
    chunk resulting from rule application is a weighted sum of the strengths 
    of the condition chunks.

    This implementation is based on Chapter 3 of Sun (2016). 

    References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(
        self,
        condition2weight : node.Chunk2Num,
        conclusion_chunk : node.Chunk
    ) -> None:
        """Initialize a Clarion associative rule.

        kwargs:
            condition2weight : A mapping from condition chunks to their weights.
            conclusion_chunk : The conclusion chunk of the rule.
        """
        
        self.condition2weight = condition2weight
        self.conclusion_chunk = conclusion_chunk

    def __call__(
        self, input_map : activation.ActivationMap
    ) -> activation.TopLevelActivationMap:
        """Return strength of conclusion chunk resulting from an application of 
        current associative rule.

        Note: If an expected chunk is missing from the input, its activation is 
        assumed to be 0.

        kwargs:
            input_map : A mapping from nodes (Chunks and/or Features) to 
            input activations.
        """
        
        conclusion_strength = 0. 
        for condition in self.condition2weight:
            condition_strength = input_map[condition]
            conclusion_strength += (
                condition_strength * self.condition2weight[condition]
            )
        return TopLevelActivationMap(
            {self.conclusion_chunk: conclusion_strength}
        )


class MaxJunction(activation.Junction):
    """An activation junction returning max activations for all input nodes.
    """

    def __call__(
        self, *input_maps : activation.ActivationMap
    ) -> activation.ActivationMap:
        """Return the maximum activation value for each input node.

        kwargs:
            input_maps : A set of mappings from chunks and/or (micro)features 
            to activations.
        """

        node_set = node.get_nodes(*input_maps)

        activations = ActivationMap()
        for n in node_set:
            for input_map in input_maps:
                if activations[n] < input_map[n]:
                    activations[n] = input_map[n]
        return activations


class BoltzmannSelector(activation.Selector):
    """Select a chunk according to a Boltzmann distribution.
    """

    def __init__(self, temperature: float) -> None:
        """Initialize a BoltzmannSelector.

        kwargs:
            chunks : A set of (potentially) actionable chunks.
            temperature : Temperature of the Boltzmann distribution.
        """

        super().__init__()
        self.temperature = temperature

    def __call__(
        self, 
        input_map: activation.ActivationMap, 
        actionable_chunks: node.ChunkSet
    ) -> node.ChunkSet:
        """Identify chunks that are currently actionable based on their 
        strengths according to a Boltzmann distribution.

        Note: If an expected input chunk is missing, it is assumed to have 
        activation 0.

        kwargs:
            chunk2strength : A mapping from chunks to their strengths.
        """

        terms = dict()
        divisor = 0.
        for chunk in actionable_chunks:
            terms[chunk] = np.exp(input_map[chunk] / self.temperature)
            divisor += terms[chunk]
        chunk_list = list(actionable_chunks)
        probabilities = [terms[chunk] / divisor for chunk in chunk_list]
        choice = np.random.choice(chunk_list, p=probabilities)

        return {choice}


class BLA(subject.Statistic):
    """Keeps track of base-level activations (BLAs).

    Implemented according to Sun (2016) Chapter 3. See Section 3.2.1.3 (p. 62)
    and also Section 3.2.2.2 (p. 77).

    Warning: This class has no knowledge of the chunks, rules and other 
    constructs associated with the statistics it tracks.

    References:
        Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(
        self,  
        initial_activation : float = 0., 
        amplitude : float = 2., 
        decay_rate : float = .5, 
        density: float = 0.
    ) -> None:
        """Initialize a BLA instance.

        Warning: By default, a timestamp is not added at initialization. If a 
        timestamp is required at creation time, call self.add_timestamp.
        """
        
        self.initial_activation = initial_activation
        self.amplitude = amplitude
        self.decay_rate = decay_rate
        self.density = density

        self.timestamps : T.List = []

    def update(self, current_time : float) -> None:
        """Record current time as an instance of use and/or activation for 
        associated construct.
        """

        self.timestamps.append(current_time)

    def compute_bla(self, current_time : float) -> float:
        """Compute the current BLA.

        Warning: Will result in division by zero if called immediately after 
        update.
        """

        summation_terms = [
            (current_time - t) ** (- self.decay_rate) 
            for t in self.timestamps
        ]
        bla = (
            self.initial_activation + (self.amplitude  *  sum(summation_terms))
        )
        return bla

    def below_density(self, current_time : float) -> bool:
        """Return true if BLA is below density.
        """

        return self.compute_bla(current_time) < self.density


class MatchStatistics(subject.Statistic):
    """Tracks positive and negative match statistics.

    Implemented according to Sun (2016) Chapter 3. See Section 3.3.2.1 (p. 90).

    Warning: This class is not responsible for testing positive or negative 
    match criteria, and has no knowledge of the chunks and actions associated 
    with the statistics it tracks.  

    References:
        Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(self) -> None:
        """Initialize tracking for a set of match statistics.
        """

        self.positive_matches = 0.
        self.negative_matches = 0.

    def update(self, positive_match : bool) -> None:
        """Update current match statistics.

        kwargs:
            positive_match : True if the positivity criterion is satisfied, 
            false otherwise.
        """

        if positive_match:
            self.positive_matches += 1.
        else: 
            self.negative_matches += 1.

    def discount(self, multiplier : float) -> None:
        """Discount match statistics.
        """
        
        self.positive_matches *= multiplier
        self.negative_matches *= multiplier