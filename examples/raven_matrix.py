"""A very simple example of NACS-style operation on an item in the style of 
Raven's Progressive Matrices. This example uses only the basic constructs. It 
is meant to demonstrate how these components can be combined to give rise to 
the kinds of processes that are normally controlled by various subsystems such 
as ACS, NACS, MS, and MCS. Learning is not treated in this example.

The item matrix has the following structure:

    triangle  square    circle
    circle    triangle  square
    square    circle    ?

The alternatives are: triangle, circle, square.

The goal is to choose the alternative that best completes the matrix.

For simplicity, this example only uses row-wise reasoning.
"""


from pyClarion.base.node import Feature, Chunk, all_chunks
from pyClarion.base.activation import propagate
from pyClarion.base.action import Handler 
from pyClarion.default.common import (
    TopDown, BottomUp, Rule, MaxJunction, BoltzmannSelector
)
from enum import auto


####### (MICRO)FEATURES #######

class SequenceType(Feature):
    MATRIX = auto()
    ALTERNATIVE = auto()

class SequenceNum(Feature):
    S1 = auto()
    S2 = auto()

class Axis(Feature):
    ROW = auto()
    COL = auto()

class Alternative(Feature):
    A1 = auto()
    A2 = auto()
    A3 = auto()

class ShapeDistribution(Feature):
    ABSENT = auto()
    PRESENT = auto()


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


####### ACTIVATION CHANNELS #######

# TOP-DOWN LINKS

top_downs = {TopDown(chunk) for chunk in chunks}

# BOTTOM-UP LINKS

bottom_ups = {BottomUp(chunk) for chunk in chunks}

# RULES

mat2alt1 = Rule(
    chunk2weight = {altseq1 : 1.,},
    conclusion_chunk = alt1
)

mat2alt2 = Rule(
    chunk2weight = {altseq2 : 1.},
    conclusion_chunk = alt2
)

mat2alt3 = Rule(
    chunk2weight = {altseq3 : 1.,},
    conclusion_chunk = alt3
)

rules = {
    mat2alt1,
    mat2alt2,
    mat2alt3
}

# JUNCTION

max_junction = MaxJunction()


####### ACTION HANDLING #######

class ResponseTracker(object):
    """A simple object for tracking trial outcomes.
    """

    def __init__(self, correct):
        """Initialize response tracker.

        kwargs:
            correct : The chunk representing the correct response.
        """

        self.response = None
        self.correct = correct

    def record(self, response):
        """Record the subject's response.

        kwargs:
            response : The subject's response.
        """

        self.response = response

    def evaluate_outcome(self):
        """Return an evaluation of the subject's response.
        """

        if self.response:
            if self.response == self.correct:
                return "Response was correct."
            else:
                return "Response was incorrect."
        else:
            return "No response yet."

# INITIALIZE RESPONSE TRACKER

response_tracker = ResponseTracker(correct=alt1)

# DEFINE ACTION CHUNKS

action_chunks = {
    alt1, alt2, alt3
}

# MAP ACTION CHUNKS TO CALLBACKS
action_callbacks = {
    alt1 : lambda: response_tracker.record(alt1),
    alt2 : lambda: response_tracker.record(alt2),
    alt3 : lambda: response_tracker.record(alt3),
}

# ACTION SELECTOR

boltzmann_selector = BoltzmannSelector(action_chunks, temperature=0.1)

# ACTION HANDLER

execute_action = Handler(action_callbacks)


####### PROCESSING EXAMPLE #######

# Step 1: Activate matrix sequence chunks.

initial = {matseq1 : 1., matseq2 : 1.}

# Step 2: Use SBR. 
    # Activate any similar alternative sequence chunks.

    ## Step 2.1: Top-Down Activation
        # Since two chunks are simultaneously used as the source of top-down 
        # activation, some of the microfeatures may receive two activation 
        # values. This is sorted out through the use of a junction. In this 
        # case, a max-junction is used. The junction effectively combines the 
        # activations coming from these two sources.

top_down = propagate(initial, top_downs, max_junction)

    ## Step 2.2: Bottom-Up Activation
        # The resulting activations from the top-down step are now used for 
        # bottom up activation. Normally, activations would spread throughout 
        # the top and bottom levels before this step. In this case, there are 
        # no rules connecting matrix sequences to other sequences, and no 
        # implicit connections. Thus this step is essentially skipped. 

bottom_up = propagate(top_down, bottom_ups, max_junction)

# The result of SBR is a chunk to strength mapping denoting alternative sequence 
# activation due to similarity.

# Step 3: Rule Application
    # Activate alternative chunks based on activation of corresponding 
    # alternative sequence chunks.

results = propagate(bottom_up, rules, max_junction)

####### ALTERNATIVE SELECTION #######
    # At the end of reasoning, we choose an alternative, here using a boltzmann 
    # distribution. Given the setup of the scenario and model, this chunk should 
    # likely be chunk alt1 (the correct response). A correct response is not
    # guaranteed due to the random nature of response selection.

choice = boltzmann_selector(results)
execute_action(choice)

# Now we can check if the response was correct using response_tracker.

outcome = response_tracker.evaluate_outcome() 