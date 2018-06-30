from chunk import Chunk, Chunk2Float

class Rule(object):
    """A basic Clarion associative rule.

    Rules have the form:
        chunk_1 chunk_2 chunk_3 -> chunk_4
    Chunks in the left-hand side are condition chunks, the single chunk in the 
    right-hand side is the conclusion chunk. The strength of the conclusion 
    chunk resulting from rule application is a weighted sum of the strengths 
    of the condition chunks.

    Only essential features of Clarion rules are represented in this class. 
    More advanced or specialized features may be added using specialized Mixin 
    classes (see, e.g., bla.BLAMixin).

    This implementation is based on Chapter 3 of Sun (2016). 

    References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(
        self,
        chunk2weight : Chunk2Float,
        conclusion_chunk : Chunk
    ) -> None:
        """Initialize a Clarion associative rule.

        kwargs:
            chunk2weight : A mapping from condition chunks to their weights.
            conclusion_chunk : The conclusion chunk of the rule.
        """
        
        self.chunk2weight = chunk2weight
        self.conclusion_chunk = conclusion_chunk
    
    def apply(self, chunk2strength : Chunk2Float) -> float:
        """Return strength of conclusion chunk resulting from an application of 
        current associative rule.

        kwargs:
            chunk2strength : A mapping from chunks to their current strengths.
        """
        
        strength = 0. 
        for chunk in self.chunk2weight:
            try:
                strength += chunk2strength[chunk] * self.chunk2weight[chunk]
            except KeyError:
                continue
        return strength