import pyClarion.base as clb
import pyClarion.numdicts as nd
import pyClarion.domain as Domain

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

class TestDomainMethods(unittest.TestCase):

    def test_lock_disallows_mutation_of_domain(self):

        dom = Domain(
        features=(
            feature("x", "a"),
            feature("y", "b"),
            feature("z", "c"),
            feature("d"),
            feature("e")
        )

        dom.lock()

    )

    def test_disjoint_recognize_overlaps(self):

        dom1 = Domain(
        features=(
            feature("x", "a"),
            feature("y", "b"),
            feature("z", "c"),
            feature("d"),
            feature("e")
        )

        dom2 = Domain(
        features=(
            feature("z", "a"),
            feature("x", "b"),
            feature("y", "c"),
            feature("f"),
            feature("g")
        )

        dom3 = Domain(
        features=(
            feature("x", "d"),
            feature("b", "y"),
            feature("z", "c"),
            feature("f")
        )

        dom0 = Domain(
        features=(
        )

        dom4 = Domain(
        features=(
            feature("1", "2"),
            feature("3", "4"),
            feature("5"),
            feature("6")
        )

        self.assertEqual(Domain.disjoint(dom1, dom2), True)
        self.assertEqual(Domain.disjoint(dom1, dom3), False)
        self.assertEqual(Domain.disjoint(dom1, dom2, dom4), True)
        self.assertEqual(Domain.disjoint(dom1, dom0), True)
        self.assertEqual(Domain.disjoint(dom1), True)
        self.assertEqual(Domain.disjoint(), True)