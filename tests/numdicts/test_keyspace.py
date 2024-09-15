import unittest

from pyClarion.numdicts import KeySpace, Key, bind, unbind


class KeySpaceTestCase(unittest.TestCase):
    
    def test_key_addition(self):
        ksp = KeySpace()
        ksp.foo; ksp.bar; ksp.baz
        self.assertTrue(all(key in ksp for key in ["foo", "bar", "baz"]))

    def test_product_space_addition(self):
        ksp = KeySpace()
        ksp.foo; ksp.bar; ksp.baz
        bind(ksp.foo, ksp.bar, ksp.baz)
        self.assertTrue(all(prod in ksp for prod in [
            "(foo,bar)", "(foo,baz)", "(bar,baz)", "(foo,bar,baz)"]))

    def test_key_removal(self):
        ksp = KeySpace()
        ksp.foo; ksp.bar; ksp.baz
        del ksp.foo
        self.assertFalse("foo" in ksp)
        self.assertTrue("bar" in ksp and "baz" in ksp)
    
    def test_product_space_deletion(self):
        ksp = KeySpace()
        ksp.foo; ksp.bar; ksp.baz
        bind(ksp.foo, ksp.bar, ksp.baz)
        del ksp.bar
        self.assertTrue("(foo,baz)" in ksp)
        self.assertFalse(any(prod in ksp for prod in [
            "(foo,bar)", "(bar,baz)", "(foo,bar,baz)"]))

    def test_membership_check(self):
        ksp = KeySpace()
        ksp.foo.bar; ksp.foo.baz; ksp.qux.xyz
        bind(ksp.foo.bar, ksp.foo.baz)
        bind(ksp.foo, ksp.qux)
        self.assertTrue("foo:bar" in ksp)
        self.assertTrue("(foo,qux):(,xyz)" in ksp)
        self.assertFalse("(foo,bar):(baz,qux)" in ksp)

    def test_enumeration(self):
        ksp = KeySpace()
        ksp.foo.bar; ksp.foo.baz; ksp.qux.xyz
        bind(ksp.foo.bar, ksp.foo.baz)
        bind(ksp.foo, ksp.qux)
        l_1 = list(ksp._iter_(1))
        self.assertTrue(l_1 == [Key("foo"), Key("qux"), Key("(foo,qux)")])
        l_2 = list(ksp._iter_(2))
        self.assertTrue(l_2 == [
            Key("foo:bar"), 
            Key("foo:baz"), 
            Key("foo:(bar,baz)"), 
            Key("qux:xyz"), 
            Key("(foo,qux):(bar,xyz)"), 
            Key("(foo,qux):(baz,xyz)"), 
            Key("(foo,qux):((bar,baz),xyz)")])


@unittest.skip("Not Implemented")
class IndexTestCase(unittest.TestCase):
    ...

if __name__ == "__main__":
    unittest.main()