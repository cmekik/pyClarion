import pyClarion.base as clb
import pyClarion.numdicts as nd

import unittest
import unittest.mock as mock


class TestProcess(unittest.TestCase):

    def test_check_inputs_accepts_good_input_structure(self):

        process = clb.Process(
            expected=[
                clb.chunks("in"),
                (clb.subsystem("nacs"), clb.terminus("selection"))
            ]
        )

        inputs = {
            clb.chunks("in"): nd.NumDict(default=0),
            clb.subsystem("nacs"): {
                clb.terminus("selection"): nd.NumDict(default=0),
                clb.terminus("selection2"): nd.NumDict(default=0)
            }
        }

        process.check_inputs(inputs)

    def test_check_inputs_rejects_incorrect_input_structure(self):

        process = clb.Process(
            expected=[
                clb.chunks("in"),
                (clb.subsystem("nacs"), clb.terminus("selection"))
            ]
        )

        with self.subTest(msg="Missing key."):
            with self.assertRaises(RuntimeError):
                inputs = {
                    # clb.chunks("in"): nd.NumDict(default=0),
                    clb.subsystem("nacs"): {
                        clb.terminus("selection"): nd.NumDict(default=0),
                        clb.terminus("selection2"): nd.NumDict(default=0)
                    }
                }
                process.check_inputs(inputs)

        with self.subTest(msg="Too deep."):
            with self.assertRaises(RuntimeError):
                inputs = {
                    clb.chunks("in"): {
                        clb.chunks("in"): nd.NumDict(default=0)
                    },
                    clb.subsystem("nacs"): {
                        clb.terminus("selection"): nd.NumDict(default=0),
                        clb.terminus("selection2"): nd.NumDict(default=0)
                    }
                }
                process.check_inputs(inputs)

        with self.subTest(msg="Too shallow."):
            with self.assertRaises(RuntimeError):
                inputs = {
                    clb.chunks("in"): nd.NumDict(default=0),
                    clb.subsystem("nacs"): nd.NumDict(default=0)
                }
                process.check_inputs(inputs)

    def test_extract_input_correctly_extracts_input(self):

        # TODO: Check if inputs are taken from correct keys. - Can

        process = clb.Process(
            expected=[
                clb.chunks("in"),
                (clb.subsystem("nacs"), clb.terminus("selection"))
            ]
        )

        inputs = {
            clb.chunks("in"): nd.NumDict(default=0),
            clb.subsystem("nacs"): {
                clb.terminus("selection"): nd.NumDict(default=0),
                clb.terminus("selection2"): nd.NumDict(default=0)
            }
        }

        chunks_in, nacs_selection = process.extract_inputs(inputs)

        self.assertIsInstance(chunks_in, nd.NumDict)
        self.assertIsInstance(nacs_selection, nd.NumDict)


class TestWrappedProcess(unittest.TestCase):

    pass