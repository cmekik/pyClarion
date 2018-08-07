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
from pyClarion.base.activation import propagate, ChannelSet, ActivationDict
from pyClarion.base.subject import execute_actions
from pyClarion.default.common import (
    Chunk, TopDown, BottomUp, Rule, MaxJunction, BoltzmannSelector
)


####### (MICRO)FEATURES #######

matseq = Microfeature("sequence type", "matrix")
altseq = Microfeature("sequence type", "alternative")

seq_1 = Microfeature("sequence number", "1")
seq_2 = Microfeature("sequence number", "2")

ax_row = Microfeature("axis", "row")
ax_col = Microfeature("axis", "column") 

alt_1 = Microfeature("alternative", "triangle")
alt_2 = Microfeature("alternative", "circle")
alt_3 = Microfeature("alternative", "square")

shp_dist = Microfeature("shape distribution", "present")
shp_cte = Microfeature("shape distribution", "absent")


####### CHUNKS #######

ch_alt1 = Chunk(
    microfeatures = {alt_1},
    label = "triangle"
)

ch_alt2 = Chunk(
    microfeatures = {alt_2},
    label = "circle"
)

ch_alt3 = Chunk(
    microfeatures = {alt_3},
    label = "square"
)

matseq1 = Chunk(
    microfeatures = {
        matseq,
        ax_row,
        seq_1,
        shp_dist
    },
    label = "triangle square circle"
)

matseq2 = Chunk(
    microfeatures = {
        matseq,
        ax_row,
        seq_2,
        shp_dist
    },
    label = "circle triangle square"
)

altseq1 = Chunk(
    microfeatures = {
        altseq,
        ax_row,
        alt_1,
        shp_dist
    },
    label = "square circle triangle"
)

altseq2 = Chunk(
    microfeatures = {
        altseq,
        ax_row,
        alt_2,
        shp_cte
    },
    label = "square circle circle"
)

altseq3 = Chunk(
    microfeatures = {
        altseq,
        ax_row,
        alt_3,
        shp_cte
    },
    label = "square circle square"
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

top_down_channels : ChannelSet = {TopDown(chunk) for chunk in chunks}

# BOTTOM-UP LINKS

bottom_up_channels : ChannelSet = {BottomUp(chunk) for chunk in chunks}

# RULES

altseq1_2_alt1 = Rule(
    condition2weight = {altseq1 : 1.},
    conclusion_chunk = ch_alt1
)

altseq2_2_alt2 = Rule(
    condition2weight = {altseq2 : 1.},
    conclusion_chunk = ch_alt2
)

altseq3_2_alt3 = Rule(
    condition2weight = {altseq3 : 1.},
    conclusion_chunk = ch_alt3
)

top_level_channels : ChannelSet = {
    altseq1_2_alt1,
    altseq2_2_alt2,
    altseq3_2_alt3
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

initial_activations = ActivationDict(matseq1=1., matseq2=1.) 

# Step 2: Use SBR. 
    # Activate any similar alternative sequence chunks.

    ## Step 2.1: Top-Down Activation
        # Since two chunks are simultaneously used as the source of top-down 
        # activation, some of the microfeatures may receive two activation 
        # values. This is sorted out through the use of a junction. In this 
        # case, a max-junction is used. The junction effectively combines the 
        # activations coming from these two sources.

top_down_activations = propagate(
    initial_activations, top_down_channels, max_junction
)

    ## Step 2.2: Bottom-Up Activation
        # The resulting activations from the top-down step are now used for 
        # bottom up activation. Normally, activations would spread throughout 
        # the top and bottom levels before this step. In this case, there are 
        # no rules connecting matrix sequences to other sequences, and no 
        # implicit connections. Thus this step is essentially skipped. 

bottom_up_activations = propagate(
    top_down_activations, bottom_up_channels, max_junction
)

# The result of SBR is a chunk to strength mapping denoting alternative sequence 
# activation due to similarity.

# Step 3: Rule Application
    # Activate alternative chunks based on activation of corresponding 
    # alternative sequence chunks.

top_level_activations = propagate(
    bottom_up_activations, top_level_channels, max_junction
)

####### ALTERNATIVE SELECTION #######
    # At the end of reasoning, we choose an alternative, here using a boltzmann 
    # distribution. Given the setup of the scenario and model, this chunk should 
    # likely be chunk alt1 (the correct response). A correct response is not
    # guaranteed due to the random nature of response selection.

choice = boltzmann_selector(top_level_activations, action_chunks)
execute_actions(choice, action_callbacks)

# Now we can check if the response was correct using response_tracker.

outcome = response_tracker.evaluate_outcome() 