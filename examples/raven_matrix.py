"""A very simple example of NACS-style operation on an item in the style of 
Raven's Progressive Matrices. This example uses only the basic constructs of 
Feature, Chunk, and Rule. It is meant to demonstrate how these components can 
be combined to give rise to the kinds of processes that are normally controlled 
by various subsystems such as ACS, NACS, MS, and MCS. Learning is not treated 
in this example.

The item matrix has the following structure:

    triangle  square    circle
    circle    triangle  square
    square    circle    ?

The alternatives are: triangle, circle, square.

The goal is to choose the alternative that best completes the matrix.

For simplicity, this example only uses row-wise reasoning.
"""

from feature import Feature
from chunk import Chunk
from rule import Rule

####### (MICRO)FEATURES #######

class SequenceType(Feature):
    MATRIX = 0
    ALTERNATIVE = 1

class SequenceNum(Feature):
    S1 = 1
    S2 = 2

class Axis(Feature):
    ROW = 0
    COL = 1

class Alternative(Feature):
    A1 = 1
    A2 = 2
    A3 = 3

class ShapeDistribution(Feature):
    ABSENT = 0
    PRESENT = 1

####### CHUNKS #######

alt1 = Chunk(
    microfeatures = {Alternative.A1},
    label = "Alternative 1"
)

alt2 = Chunk(
    microfeatures = {Alternative.A2},
    label = "Alternative 2"
)

alt3 = Chunk(
    microfeatures = {Alternative.A3},
    label = "Alternative 3"
)

matseq1 = Chunk(
    microfeatures = {
        SequenceType.MATRIX,
        Axis.ROW,
        SequenceNum.S1,
        ShapeDistribution.PRESENT
    },
    label = "Matrix Sequence 1"
)

matseq2 = Chunk(
    microfeatures = {
        SequenceType.MATRIX,
        Axis.ROW,
        SequenceNum.S2,
        ShapeDistribution.PRESENT
    },
    label = "Matrix Sequence 2"
)

altseq1 = Chunk(
    microfeatures = {
        SequenceType.ALTERNATIVE,
        Axis.ROW,
        Alternative.A1,
        ShapeDistribution.PRESENT
    },
    label = "Alternative Sequence 1"
)

altseq2 = Chunk(
    microfeatures = {
        SequenceType.ALTERNATIVE,
        Axis.ROW,
        Alternative.A2,
        ShapeDistribution.ABSENT
    },
    label = "Alternative Sequence 2"
)

altseq3 = Chunk(
    microfeatures = {
        SequenceType.ALTERNATIVE,
        Axis.ROW,
        Alternative.A3,
        ShapeDistribution.ABSENT
    },
    label = "Alternative Sequence 3"
)

chunks = {
    alt1,
    alt2,
    alt3,
    matseq1,
    matseq2,
    altseq1,
    altseq2,
    altseq3
}

####### RULES #######

mat2alt1 = Rule(
    chunk2weight = {
        matseq1 : .5,
        matseq2 : .5
    },
    conclusion_chunk = alt1
)

mat2alt2 = Rule(
    chunk2weight = {
        matseq1 : .5,
        matseq2 : .5
    },
    conclusion_chunk = alt2
)

mat2alt3 = Rule(
    chunk2weight = {
        matseq1 : .5,
        matseq2 : .5
    },
    conclusion_chunk = alt3
)

rules = {
    mat2alt1,
    mat2alt2,
    mat2alt3
}

####### PROCESSING EXAMPLE #######

# Step 1: Set the strength of one of the alternative rows to maximum.

chunk2strength = {
    altseq1 : 1.
}

# Step 2: Use SBR to activate any similar chunks.

## Step 2.1: Top-Down Activation
    # Note: For simplicity, I assume that all other chunk activations are 
    # 0. or that they are suppressed by MCS. Normally, top-down activation 
    # would affect all active chunks. In this case, the only active chunk is
    # altseq1.

top_down = altseq1.top_down(chunk2strength[altseq1])

## Step 2.2: Bottom-Up Activation
    # The resulting activations from the top-down step are now used for 
    # bottom up activation. Note that normally activations would spread 
    # throughout the top and bottom levels. In this case, there are no rules 
    # connecting alternative sequences to other sequences, and no implicit 
    # connections. Thus this step is essentially skipped. 

bottom_up = dict()
for chunk in chunks:
    bottom_up[chunk] = chunk.bottom_up(top_down)

# The result of this operation is a chunk2strength mapping denoting 
# activation due to similarity.

# Step 3: Filter Setup
    # Note that chunk alt1 will be activated at this step due to the shared 
    # microfeature Alternative.A<x> between altseq<x> and alt<x>. This is 
    # important because it allows the subject to isolate which alternative 
    # it cares about during this particular reasoning episode. The next 
    # step will, without intervention, activate all chunks representing 
    # matrix alternatives. There is no way to attribute the results of the 
    # current episode to the correct alternative without taking into account 
    # alternative chunk activations at this step. This is something that 
    # can be handled by the MCS; for simplicity, I do it manually.

# By 'relevant' I mean relevant to the current reasoning episode.
relevant_chunks = {chunk for chunk in bottom_up if bottom_up[chunk] > 0.} 

# Step 4: Rule Application
    # Note: This step normally would pick the strength with maximal 
    # activation. However, since no two rules have the same activation 
    # chunk, this selection process has been omitted.

conc2strength = dict()
for rule in rules:
    conc2strength[rule.conclusion_chunk] = rule.apply(bottom_up)

# Step 5: Filtering
    # See notes in Step 3 about isolating the correct alternative. In 
    # this step, the correct alternative is isolated according to the 
    # considerations discussed in that section. 

result = dict()
for chunk in relevant_chunks:
    try:
        result[chunk] = conc2strength[chunk]
    except KeyError:
        continue

# The result thus obtainde should just be the activation value of alt<x>, 
# the alternative related to the original alternative sequence under 
# consideration. This result can now be stored in WM or episodic memory for 
# action selection once the other alternatives have been selected.

####### PROCESSING OF OTHER ALTERNATIVES #######
    # I have wrapped the above in a simple static method, so that we can 
    # skip repetition.

def process_alternative_sequence(altseq, chunks, rules):

    # Step 1: Set the strength of one of the alternative rows to maximum.

    chunk2strength = {
        altseq : 1.
    }

    # Step 2: Use SBR to activate any similar chunks.

    ## Step 2.1: Top-Down Activation

    top_down = altseq.top_down(chunk2strength[altseq])

    ## Step 2.2: Bottom-Up Activation

    bottom_up = dict()
    for chunk in chunks:
        bottom_up[chunk] = chunk.bottom_up(top_down)

    # Step 3: Filter Setup

    relevant_chunks = {
        chunk for chunk in bottom_up if bottom_up[chunk] > 0.
    } 

    # Step 4: Rule Application

    conc2strength = dict()
    for rule in rules:
        conc2strength[rule.conclusion_chunk] = rule.apply(bottom_up)

    # Step 5: Filtering
    
    result = dict()
    for chunk in relevant_chunks:
        try:
            result[chunk] = conc2strength[chunk]
        except KeyError:
            continue

    return result

result2 = process_alternative_sequence(altseq2, chunks, rules)
result3 = process_alternative_sequence(altseq3, chunks, rules)

####### ALTERNATIVE SELECTION #######

results = {**result, **result2, **result3} 

choice = None
for chunk in results:
    try:
        if results[chunk] > choice:
            choice = chunk
    except TypeError:
        choice = chunk 

# At the end of all this reasoning, we choose an alternative, here using a 
# naive selection method (pick the first chunk attaining maximal 
# activation). Given the setup of the scenario, this chunk should be chunk 
# alt1 (the correct response).