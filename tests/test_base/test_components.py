import pyClarion.base as clb
import pyClarion.numdicts as nd
from pyClarion import feature

import unittest
import unittest.mock as mock
from unittest.mock import PropertyMock


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

class TestDomainMethods(unittest.TestCase):

    def test_config_calls_update_only_at_end_of_with_block(self):

        with mock.patch('pyClarion.base.Domain.update'):
            with mock.patch('pyClarion.base.Domain._config', ("A", "B", "C")):
                mockDom = clb.Domain(
                    features=(
                        feature("x", "a"),
                        feature("y", "b"),
                        feature("z", "c"),
                        feature("d"),
                        feature("e")
                    )
                )

                with mockDom.config():
                    mockDom.update.assert_not_called()
                    mockDom.A = "a"
                    mockDom.update.assert_not_called()
                    mockDom.B = "b"
                    mockDom.update.assert_not_called()
                    mockDom.C = "c"
                    mockDom.update.assert_not_called()
                mockDom.update.assert_called()

    def test_lock_disallows_mutation_of_domain(self):

        with mock.patch('pyClarion.base.Domain._config', ("A", "B", "C")):
            mockDom = clb.Domain(
                features=(
                    feature("x", "a"),
                    feature("y", "b"),
                    feature("z", "c"),
                    feature("d"),
                    feature("e")
                )
            )

            mockDom.lock()
            with self.subTest(msg="run_with_config()"):
                with mockDom.config():

                    with self.assertRaisesRegex(RuntimeError, "Cannot mutate locked domain."):
                        mockDom.A = "a"

            with self.subTest(msg="run_without_config()"):
                with self.assertRaisesRegex(RuntimeError, "Cannot mutate locked domain."):
                        mockDom.B = "b"

    def test_disjoint_recognize_overlaps(self):

        dom1 = clb.Domain(
            features=(
                feature("x", "a"),
                feature("y", "b"),
                feature("z", "c"),
                feature("d"),
                feature("e")
            )
        )

        dom2 = clb.Domain(
            features=(
                feature("z", "a"),
                feature("x", "b"),
                feature("y", "c"),
                feature("f"),
                feature("g")
            )
        )

        dom3 = clb.Domain(
            features=(
                feature("x", "d"),
                feature("b", "y"),
                feature("z", "c"),
                feature("f")
            )
        )

        dom0 = clb.Domain(
            features=()
        )

        dom4 = clb.Domain(
            features=(
                feature("1", "2"),
                feature("3", "4"),
                feature("5"),
                feature("6")
            )
        )

        with self.subTest(msg="different domains with same size"):
            self.assertEqual(clb.Domain.disjoint(dom1, dom2), True)

        with self.subTest(msg="different domains with different size"):
            self.assertEqual(clb.Domain.disjoint(dom1, dom3), False)

        with self.subTest(msg="more than 2 domains as argument"):
            self.assertEqual(clb.Domain.disjoint(dom1, dom2, dom4), True)

        with self.subTest(msg="one of the domains being empty"):
            self.assertEqual(clb.Domain.disjoint(dom1, dom0), True)

        with self.subTest(msg="only 1 domain as argument"):
            self.assertEqual(clb.Domain.disjoint(dom1), True)

        with self.subTest(msg="without argument"):
            with self.assertRaisesRegex(ValueError, "disjoint\(\) doesn't accept 0 argument"):
                clb.Domain.disjoint()


class TestInterfaceMethods(unittest.TestCase):

    def test_parse_commands(self):
        with self.subTest(msg="different domains with same size"):
            test_interface = pcl.Interface(
                cmds=(
                    pcl.feature("up", 0), 
                    pcl.feature("up", 1), 
                    pcl.feature("down", 0), 
                    pcl.feature("down", 1)
                ),
            )
            data = nd.NumDict({pcl.feature("up", 1): 1.0, 
                                pcl.feature("down", 0): 1.0}, default=0)

            # preconditions on the input
            assert data.default == 0
            assert set(data.values()) == {1.0} # activations must be 1.0 or default
            assert no_duplicate_dims(data) # each cmd dim should appear exactly once in data

            res = test_interface.parse_commands(data);
            assert (pcl.feature("up", 1) in res) or (pcl.feature("up", 0) in res)
            assert (pcl.feature("down", 1) in res) or (pcl.feature("down", 0) in res)


if __name__ == "__main__":
    unittest.main()