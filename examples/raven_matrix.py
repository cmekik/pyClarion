"""An example of non-action-centered processing on a problem in the style of 
Raven's Progressive Matrices. This example demonstrates how basic pyClarion 
components can be combined to capture cognitive processing. Learning 
is not treated in this example.

Consider the following matrix:

    triangle  square    circle
    circle    triangle  square
    square    circle    ?

The alternatives are: triangle, circle, square.

The goal is to choose the alternative that best completes the matrix.

For simplicity, this example only uses row-wise reasoning.
"""

import typing as T
from pyClarion.base.node import (
    Microfeature, ChunkSet, Chunk2Callable, Node2Float
)
from pyClarion.base.activation import propagate, ChannelSet
from pyClarion.base.subject import execute_actions
from pyClarion.default.common import (
    Chunk, TopDown, BottomUp, Rule, MaxJunction, BoltzmannSelector
)


####### (MICRO)FEATURES #######

matseq = Microfeature("Sequence Type", "mat")
altseq = Microfeature("Sequence Type", "alt")

seq_1 = Microfeature("Sequence Number", "1")
seq_2 = Microfeature("Sequence Number", "2")

ax_row = Microfeature("Axis", "row")
ax_col = Microfeature("Axis", "col") 

alt_1 = Microfeature("Alternative", "1")
alt_2 = Microfeature("Alternative", "2")
alt_3 = Microfeature("Alternative", "3")

shp_dist = Microfeature("Shape Distribution", "present")
shp_cte = Microfeature("Shape Distribution", "absent")


####### CHUNKS #######

ch_alt1 = Chunk(
    microfeatures = {alt_1},
    label = "Alternative 1"
)

ch_alt2 = Chunk(
    microfeatures = {alt_2},
    label = "Alternative 2"
)

ch_alt3 = Chunk(
    microfeatures = {alt_3},
    label = "Alternative 3"
)

matseq1 = Chunk(
    microfeatures = {
        matseq,
        ax_row,
        seq_1,
        shp_dist
    },
    label = "Matrix Sequence 1"
)

matseq2 = Chunk(
    microfeatures = {
        matseq,
        ax_row,
        seq_2,
        shp_dist
    },
    label = "Matrix Sequence 2"
)

altseq1 = Chunk(
    microfeatures = {
        altseq,
        ax_row,
        alt_1,
        shp_dist
    },
    label = "Alternative Sequence 1"
)

altseq2 = Chunk(
    microfeatures = {
        altseq,
        ax_row,
        alt_2,
        shp_cte
    },
    label = "Alternative Sequence 2"
)

altseq3 = Chunk(
    microfeatures = {
        altseq,
        ax_row,
        alt_3,
        shp_cte
    },
    label = "Alternative Sequence 3"
)

chunks = {
    ch_alt1,
    ch_alt2,
    ch_alt3,
    matseq1,
    matseq2,
    altseq1,
    altseq2,
    altseq3
}


####### ACTIVATION CHANNELS #######

# TOP-DOWN LINKS

top_downs : ChannelSet = {TopDown(chunk) for chunk in chunks}

# BOTTOM-UP LINKS

bottom_ups : ChannelSet = {BottomUp(chunk) for chunk in chunks}

# RULES

mat2alt1 = Rule(
    condition2weight = {altseq1 : 1.},
    conclusion_chunk = ch_alt1
)

mat2alt2 = Rule(
    condition2weight = {altseq2 : 1.},
    conclusion_chunk = ch_alt2
)

mat2alt3 = Rule(
    condition2weight = {altseq3 : 1.},
    conclusion_chunk = ch_alt3
)

rules : ChannelSet = {
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

response_tracker = ResponseTracker(correct=ch_alt1)

# DEFINE ACTION CHUNKS

action_chunks : ChunkSet = {
    ch_alt1, ch_alt2, ch_alt3
}

# MAP ACTION CHUNKS TO CALLBACKS

action_callbacks : Chunk2Callable= {
    ch_alt1 : lambda: response_tracker.record(ch_alt1),
    ch_alt2 : lambda: response_tracker.record(ch_alt2),
    ch_alt3 : lambda: response_tracker.record(ch_alt3),
}

# ACTION SELECTOR

boltzmann_selector = BoltzmannSelector(temperature=0.1)


####### PROCESSING EXAMPLE #######

# Step 1: Activate matrix sequence chunks.

initial : Node2Float = {matseq1 : 1., matseq2 : 1.}

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

choice = boltzmann_selector(results, action_chunks)
execute_actions(choice, action_callbacks)

# Now we can check if the response was correct using response_tracker.

outcome = response_tracker.evaluate_outcome() 