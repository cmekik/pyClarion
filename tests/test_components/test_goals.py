from pyClarion import (
    SimpleDomain, pprint, feature, chunk, buffer, subsystem, terminus, Chunks, 
    BLAs, ActionSelector, nd
)
from pyClarion.components.goals import GoalStay

import unittest
import unittest.mock as mock


class TestGoalStay(unittest.TestCase):

    def test_goal_stay_interface_init_succeeds_under_normal_input(self):

        goal_domain = SimpleDomain([
            feature("goal", "select"),
            feature("goal", "analyze"),
            feature("goal", "evaluate"),
            feature("goal_object", "attribute"),
            feature("goal_object", "response"),
            feature("goal_object", "pattern")
        ])

        # baseline; make sure it works then test w/ pathological inputs
        interface = GoalStay.Interface(goals=goal_domain)

    def test_goal_buffer_push(self):

        goal_domain = SimpleDomain([
            feature("goal", "select"),
            feature("goal", "analyze"),
            feature("goal", "evaluate"),
            feature("goal_object", "attribute"),
            feature("goal_object", "response"),
            feature("goal_object", "pattern")
        ])

        interface = GoalStay.Interface(goals=goal_domain)

        chunks = Chunks()
        blas = BLAs(density=1.0)

        input_ = nd.NumDict({
            feature(("gb", "set"), "push"): 1.0,
            feature(("gb", "set", "goal"), "analyze"): 1.0,
            feature(("gb", "set", "goal_object"), "pattern"): 1.0
        })
        inputs = {
            subsystem("acs"): {
                terminus("gb_actions"): input_
            },
            subsystem("ms"): {
                terminus("goal_selection"): nd.NumDict(default=0)
            }
        }

        gb = GoalStay(
            controller=(subsystem("acs"), terminus("gb_actions")),
            source=(subsystem("ms"), terminus("goal_selection")),
            interface=interface,
            chunks=chunks,
            blas=blas
        )
        gb.entrust(buffer("gb"))

        output = gb.call(inputs)
        blas.step()
        chunks.step()

        output = gb.call(inputs)
        blas.step()
        chunks.step()

        input_ = nd.NumDict({
            feature(("gb", "set"), "fail"): 1.0,
            feature(("gb", "set", "goal"), "analyze"): 1.0,
            feature(("gb", "set", "goal_object"), "pattern"): 1.0
        })
        inputs = {
            subsystem("acs"): {
                terminus("gb_actions"): input_
            },
            subsystem("ms"): {
                terminus("goal_selection"): nd.NumDict({
                    chunk("goal_1"): 1.0
                })
            }
        }

        output = gb.call(inputs)
        blas.step()
        chunks.step()

        # pprint(output)
        # pprint(chunks)
        # pprint(blas)
