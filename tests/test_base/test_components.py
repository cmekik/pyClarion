import pyClarion.base as clb
import pyClarion.numdicts as nd

import unittest
import unittest.mock as mock


class TestProcess(unittest.TestCase):

    @mock.patch.object(clb.Process, "_serves", clb.ConstructType.chunks)
    def test_check_inputs_accepts_good_input_structure(self):

        process = clb.Process(
            expected=[clb.buffer("wm"), clb.terminus("selection")]
        )

        inputs = {
            clb.buffer("wm"): nd.NumDict(default=0),
            clb.terminus("selection"): nd.NumDict(default=0),
            clb.terminus("selection2"): nd.NumDict(default=0)
        }
        process.check_inputs(inputs)

    @mock.patch.object(clb.Process, "_serves", clb.ConstructType.chunks)
    def test_check_inputs_rejects_incomplete_input(self):

        process = clb.Process(
            expected=[clb.chunks("in"), clb.terminus("selection")]
        )

        with self.assertRaises(RuntimeError):
            inputs = {
                # clb.buffer("wm"): nd.NumDict(default=0),
                clb.terminus("selection"): nd.NumDict(default=0),
                clb.terminus("selection2"): nd.NumDict(default=0)
            }
            process.check_inputs(inputs)


class TestWrappedProcess(unittest.TestCase):
    pass
