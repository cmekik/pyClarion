import unittest

from pyClarion.numdicts import Key


@unittest.skip("very broken")
class KeySpaceTestCase(unittest.TestCase):
    
    def test_key_addition(self):
        ksp = KeySpace()
        ksp.foo; ksp.bar; ksp.baz
        self.assertTrue(all(key in ksp for key in ["foo", "bar", "baz"]))

    def test_product_containment(self):
        ksp = KeySpace()
        ksp.foo; ksp.bar; ksp.baz
        self.assertTrue(all(prod in ksp for prod in [
            "(foo,bar)", "(foo,baz)", "(bar,baz)", "(foo,bar,baz)"]))

    def test_key_removal(self):
        ksp = KeySpace()
        ksp.foo; ksp.bar; ksp.baz
        del ksp.foo
        self.assertFalse("foo" in ksp)
        self.assertTrue("bar" in ksp and "baz" in ksp)
    
    def test_product_deletion(self):
        ksp = KeySpace()
        ksp.foo; ksp.bar; ksp.baz
        del ksp.bar
        self.assertTrue("(foo,baz)" in ksp)
        self.assertFalse(any(prod in ksp for prod in [
            "(foo,bar)", "(bar,baz)", "(foo,bar,baz)"]))

    def test_membership_check(self):
        ksp = KeySpace()
        ksp.foo.bar; ksp.foo.baz; ksp.qux.xyz
        self.assertTrue("foo:bar" in ksp)
        self.assertTrue("(foo,qux):(,xyz)" in ksp)
        self.assertFalse("(foo,bar):(baz,qux)" in ksp)

    def test_iter(self):
        ksp = KeySpace()
        ksp.foo.bar; ksp.foo.baz; ksp.qux.xyz
        lst = list(ksp)
        self.assertTrue(lst == ["foo", "qux"])


@unittest.skip("Not Implemented")
class IndexTestCase(unittest.TestCase):
    ...

if __name__ == "__main__":
    unittest.main()