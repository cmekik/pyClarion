import pyClarion.base as clb
import pyClarion.numdicts as nd
from pyClarion import feature

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

    def test_config(self):

        dom = clb.Domain(
            features=(
                feature("x", "a"),
                feature("y", "b"),
                feature("z", "c"),
                feature("d"),
                feature("e")
            )
        )

        with dom.config():
            dom._features[0] = feature("")

    def test_lock_disallows_mutation_of_domain(self):

        dom = clb.Domain(
            features=(
                feature("x", "a"),
                feature("y", "b"),
                feature("z", "c"),
                feature("d"),
                feature("e")
            )
        )

        dom.lock()
        # how to modify dom?
        
        #dom = clb.Domain(
        #    features=(
        #        feature("d")
        #    )
        #)
        # domain __init__ doesn't accept single argument (single feature)

        #fs = dom.features() # tuple not callable?


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

        self.assertEqual(clb.Domain.disjoint(dom1, dom2), True)
        self.assertEqual(clb.Domain.disjoint(dom1, dom3), False)
        self.assertEqual(clb.Domain.disjoint(dom1, dom2, dom4), True)
        self.assertEqual(clb.Domain.disjoint(dom1, dom0), True)
        #self.assertEqual(clb.Domain.disjoint(dom1), True) # ???
        #self.assertEqual(clb.Domain.disjoint(), True) # argument cannot be empty


if __name__ == "__main__":
    unittest.main()