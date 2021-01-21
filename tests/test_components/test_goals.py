from pyClarion import (
    pprint, feature, chunk, buffer, subsystem, terminus, agent, 
    Chunks, BLAs, ActionSelector, nd
)
from pyClarion.components.goals import GoalStay

import unittest
import unittest.mock as mock


class TestGoalStay(unittest.TestCase):

    def test_goal_stay_interface_init_succeeds_under_normal_input(self):

        # baseline; make sure it works then test w/ pathological inputs
        interface = GoalStay.Interface(
            name="gctl",
            goals=(
                feature("goal", "select"),
                feature("goal", "analyze"),
                feature("goal", "evaluate"),
                feature("goal_object", "attribute"),
                feature("goal_object", "response"),
                feature("goal_object", "pattern")
            )
        )

    def test_goal_buffer_push(self):

        # TODO: Add assertions...

        interface = GoalStay.Interface(
            name="gctl",
            goals=(
                feature("goal", "select"),
                feature("goal", "analyze"),
                feature("goal", "evaluate"),
                feature("gobj", "attribute"),
                feature("gobj", "response"),
                feature("gobj", "pattern")
            )
        )

        chunks = Chunks()
        blas = BLAs(density=1.0)

        gb = GoalStay(
            controller=(subsystem("acs"), terminus("gb_actions")),
            source=(subsystem("ms"), terminus("goal_selection")),
            interface=interface,
            chunks=chunks,
            blas=blas
        )
        
        input_ = nd.NumDict({
            feature(("gctl", ".cmd"), ".w"): 1.0,
            feature(("gctl", "goal"), "analyze"): 1.0,
            feature(("gctl", "gobj"), "pattern"): 1.0
        })        
        inputs = {
            (subsystem("acs"), terminus("gb_actions")): input_,
            (subsystem("ms"), terminus("goal_selection")): nd.NumDict(default=0)
        }

        output = gb.call(inputs)
        chunks.step()

        # pprint(output)
        # pprint(chunks)
        # pprint(blas)

        input_ = nd.NumDict({
            feature(("gctl", ".cmd"), ".w"): 1.0,
            feature(("gctl", "goal"), "evaluate"): 1.0,
            feature(("gctl", "gobj"), "attribute"): 1.0
        })        
        inputs = {
            (subsystem("acs"), terminus("gb_actions")): input_,
            (subsystem("ms"), terminus("goal_selection")): nd.NumDict(default=0)
        }

        output = gb.call(inputs)
        chunks.step()

        # pprint(output)
        # pprint(chunks)
        # pprint(blas)

        input_ = nd.NumDict({
            feature(("gctl", ".cmd"), ".f"): 1.0,
            feature(("gctl", "goal"), "analyze"): 1.0,
            feature(("gctl", "gobj"), "pattern"): 1.0
        })
        inputs = {
            (subsystem("acs"), terminus("gb_actions")): input_,
            (subsystem("ms"), terminus("goal_selection")): nd.NumDict({
                chunk(".goal_1"): 1.0
            })
        }

        output = gb.call(inputs)
        chunks.step()

        # pprint(output)
        # pprint(chunks)
        # pprint(blas)
