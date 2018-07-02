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

from nodes import Feature, Chunk
from activation import TopDown, BottomUp, Rule, ActivationFilter, MaxJunction
from action import BoltzmannSelector, ActionHandler

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

####### ACTIVATION CHANNELS #######

# TOP-DOWN LINKS

top_downs = {TopDown(chunk) for chunk in chunks}

# BOTTOM-UP LINKS

bottom_ups = {BottomUp(chunk) for chunk in chunks}

# RULES

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

# FILTERS

reasoning_filter = ActivationFilter({})

# JUNCTIONS

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

execute_action = ActionHandler(action_callbacks)

####### PROCESSING EXAMPLE #######

# Step 1: Activate an alternative sequence chunk.

chunk2strength = {altseq1 : 1.}

# Step 2: Use SBR. 
    # Activate any similar chunks.

## Step 2.1: Top-Down Activation
    # Note: Top-down activation affects all active chunks that are not 
    # suppressed by MCS. In this case, the only active chunk is altseq1.

top_down = dict()
for td in top_downs:
    top_down.update(td(chunk2strength))

## Step 2.2: Bottom-Up Activation
    # The resulting activations from the top-down step are now used for 
    # bottom up activation. Normally, activations would spread throughout the 
    # top and bottom levels before this step. In this case, there are no rules 
    # connecting alternative sequences to other sequences, and no implicit 
    # connections. Thus this step is essentially skipped. 

bottom_up = dict()
for bu in bottom_ups:
    bottom_up.update(bu(top_down))

# The result of this operation is a chunk2strength mapping denoting 
# activation due to similarity.

# Step 3: Filter Setup
    # Chunk alt1 will be activated at this step due to the shared microfeature 
    # Alternative.A1 between altseq1 and alt1. This is important because it 
    # allows the subject to isolate which alternative it cares about during this 
    # particular reasoning episode. The next step will, without intervention 
    # from MCS, activate all chunks representing matrix alternatives. There is 
    # no way to attribute the results of the current episode to the correct 
    # alternative without taking into account alternative chunk activations at 
    # this step. This is something that can be handled by the MCS in a more 
    # detailed simulation; here, I do it manually.

reasoning_filter.nodes = {chunk for chunk in bottom_up if bottom_up[chunk] > 0.} 

# Step 4: Rule Application
    # Note: This step normally would pick, for each chunk, the maximal resulting 
    # strength. However, since no two rules have the same conclusion chunk, 
    # this selection process has been omitted.

conc2strength = dict()
for rule in rules:
    conc2strength.update(rule(bottom_up))

# Step 5: Filtering
    # See notes in Step 3 about isolating the correct alternative. In this 
    # step, the correct alternative is isolated, according to the considerations 
    # discussed in that section, by filtering out all chunks except those found 
    # to be relevant to the present episode. 

result1 = reasoning_filter(conc2strength)

# The result of filtering should just be the activation value of alt1, the 
# alternative that generates the initial alternative sequence when plugged into 
# the blank. This result can now be stored in WM or episodic memory for action 
# selection once the other alternatives have been processed.

####### PROCESSING OF OTHER ALTERNATIVES #######
    # I have wrapped the above in a function, so that we can skip repetition.

def process_alternative_sequence(
    altseq, chunks, top_downs, bottom_ups, rules, reasoning_filter
):
    
    chunk2strength = {altseq : 1.}
    
    top_down = dict()
    for td in top_downs:
        top_down.update(td(chunk2strength))
    
    bottom_up = dict()
    for bu in bottom_ups:
        bottom_up.update(bu(top_down))

    reasoning_filter.nodes = {
        chunk for chunk in bottom_up if bottom_up[chunk] > 0.
    }

    conc2strength = dict()
    for rule in rules:
        conc2strength.update(rule(bottom_up)) 

    result = reasoning_filter(conc2strength)
    
    return result

result2 = process_alternative_sequence(
    altseq2, chunks, top_downs, bottom_ups, rules, reasoning_filter
)
result3 = process_alternative_sequence(
    altseq3, chunks, top_downs, bottom_ups, rules, reasoning_filter
)

####### ALTERNATIVE SELECTION #######
    # At the end of all this reasoning, we choose an alternative, here using a 
    # boltzmann distribution. Given the setup of the scenario and model, this 
    # chunk should likely be chunk alt1 (the correct response). A correct 
    # response is not guaranteed due to the random nature of response selection.

results = max_junction(result1, result2, result3) 
choice = boltzmann_selector(results)
execute_action(choice)

# Now we can check if the response was correct using response_tracker.

correct = response_tracker.evaluate_outcome() 