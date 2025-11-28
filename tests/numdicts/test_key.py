import unittest

from pyClarion.numdicts.keys import ValidationError, Key, KeyForm


class KeyRepresentationTestCase(unittest.TestCase):

    def setUp(self):
        self.valid_keys = [
            ("", (("", 0),)),
            ("a:b:c:d", (("", 1), ("a", 1), ("b", 1), ("c", 1), ("d", 0))),
            ("(a,b,c,d)", (("", 4), ("a", 0), ("b", 0), ("c", 0), ("d", 0))),
            ("a:(b,c):(d,(e,f))",
                (("", 1), ("a", 2), ("b", 1), ("c", 2), ("d", 0), ("e", 0), 
                    ("f", 0))),
            ("a:(b,c,d):(e,(f,g),):(,(,h),)",
                (("", 1), ("a", 3), ("b", 1), ("c", 2), ("d", 0), ("e", 0), 
                    ("f", 0), ("g", 1), ("h", 0))),
            ("(a,b):(c,(d,e)):(,(f,(g,h))):(,(,(,i)))",
                (("", 2), ("a", 1), ("b", 2), ("c", 0), ("d", 1), ("e", 2), 
                    ("f", 0), ("g", 0), ("h", 1), ("i", 0)))]
        self.invalid_keys = [
            ("a:(b, c)",           "Space character in key stirng"),
            ("a::b",               "Two ':' in a row"),
            ("a:(b,,c)",           "Two ',' in a row in fresh parens"),
            ("a:()",               "Empty parens"),
            ("a:(b)",              "Redundant parens"),
            ("a:b,c):(d,e)",       "Missing '(' in fresh parens"),
            ("a:(b,c):d,e)",       "Missing '(' in established parens"),
            ("a:(b,c:(d,e)",       "Missing ')' in fresh parens"),
            ("a:(b,c):(d,e",       "Missing ')' in established parens"),
            ("a:(b,c,d):(e,f)",    "Missing ',' in established parens"),
            ("a:(b,c)(d,e)",       "Missing ':' after fresh parens"),
            ("a:(b,c):(d,e)(f,g)", "Missing ':' after established parens"),
            ("a:(b,c),d",          "Unexpected ',' after fresh parens"),
            ("a:((b,c),d)",        "Fresh parens in fresh parens")
        ]

    def test_parse_of_valid_key_strings(self):
        for s, k in self.valid_keys:
            with self.subTest(s=s, k=k):
                self.assertEqual(Key(s), k)

    def test_serialization_of_key_data(self):
        for s, k in self.valid_keys:
            with self.subTest(s=s, k=k):
                self.assertEqual(Key.__str__(k), s)

    def test_error_on_bad_key_string_syntax(self):
        for s, msg in self.invalid_keys:
            with self.subTest(msg, s=s):
                self.assertRaises(ValidationError, Key, s)


class KeyPartialOrderingTestCase(unittest.TestCase):

    def setUp(self):
        self.equal_keys = [
            "", 
            "a", 
            "a:b:c:d", 
            "a:(b,c):(d,(e,f))", 
            "a:(b,c,d):(e,(f,g),):(,(,h),)",
            "(a,b):(c,(d,e)):(,(f,(g,h))):(,(,(,i)))"]
        self.ordered_pairs = [
            ("",             "a"),
            ("a",            "a:b"),
            ("a:b:d",        "a:(b,c):(d,(e,f))"),
            ("a:b:d",        "a:(b,b):(,d)"),
            ("a:b",          "(a,a):(b,b)")]
        self.unordered_pairs = [
            ("a:b",          "a:c"),
            ("a:b:g",        "a:(b,c):(d,(e,f))"),
            ("a:(g,c):(,e)", "a:(b,c):(d,(e,f))"),
            ("(a,a):(b,c)",  "a:(b,c)")]

    def test_comparison_of_equal_keys(self):
        for s in self.equal_keys:
            k1, k2 = Key(s), Key(s)
            with self.subTest(k1=k1, k2=k2):
                self.assertTrue(k1 <= k2)

    def test_comparison_of_in_order_keys(self):
        for s1, s2 in self.ordered_pairs:
            k1, k2 = Key(s1), Key(s2)
            with self.subTest(k1=k1, k2=k2):
                self.assertTrue(k1 <= k2)

    def test_comparison_of_out_of_order_keys(self):
        for s1, s2 in self.ordered_pairs:
            k1, k2 = Key(s1), Key(s2)
            with self.subTest(k1=k1, k2=k2):
                self.assertFalse(k2 <= k1)

    def test_comparison_of_unordered_keys(self):
        for s1, s2 in self.unordered_pairs:
            k1, k2 = Key(s1), Key(s2)
            with self.subTest(k1=k1, k2=k2):
                self.assertFalse(k1 <= k2)
                self.assertFalse(k2 <= k1)


unittest.skip("Incomplete/needs updating")
class KeyFormPartialOrderingTestCase(unittest.TestCase):
    
    def setUp(self):
        self.ordered_pairs = [
            (("a", (2,)), ("a:b", (1,))),
            #(("a", (1,)), ("a:(b,c)", (1, 2))), # Think about this one
            #(("a:b", (0,)), ("a:(b,c)", (0, 2)))
            ]
        self.unordered_pairs = [
            (("a", (2,)), ("a:b", (0,))),
            (("a", (2,)), ("b:c", (1,))),
            #(("a", (5,)), ("a:(b,c)", (1, 2))),
            #(("a", (2,)), ("b:(c,d)", (1, 2))),
            #(("a:b", (3,)), ("a:(b,c)", (0, 2))),
            #(("a:b", (0,)), ("a:(c,d)", (0, 2)))
            ] 

    def test_comparison_of_in_order_keyforms(self):
        for (s1, h1), (s2, h2) in self.ordered_pairs:
            kf1, kf2 = KeyForm(Key(s1), h1), KeyForm(Key(s2), h2)
            with self.subTest(kf1=kf1, kf2=kf2):
                self.assertTrue(kf1 <= kf2)

    def test_comparison_of_out_of_order_keyforms(self):
        for (s1, h1), (s2, h2) in self.ordered_pairs:
            kf1, kf2 = KeyForm(Key(s1), h1), KeyForm(Key(s2), h2)
            with self.subTest(kf1=kf1, kf2=kf2):
                self.assertFalse(kf2 <= kf1)

    def test_comparison_of_unordered_keyforms(self):
        for (s1, h1), (s2, h2) in self.unordered_pairs:
            kf1, kf2 = KeyForm(Key(s1), h1), KeyForm(Key(s2), h2)
            with self.subTest(kf1=kf1, kf2=kf2):
                self.assertFalse(kf1 <= kf2)
                self.assertFalse(kf2 <= kf1)

class KeyFormFromKeyTestCase(unittest.TestCase):

   def test(self):
    x = KeyForm(Key("(a,b):(c,)"), (1, 0))
    y = KeyForm.from_key(Key("(a,b):(c,?)"))
    self.assertEqual(x, y)

@unittest.skip("Not Implemented")
class KeyManipulationTestCase(unittest.TestCase):

    def test_key_link_method(self):
        ...

    def test_key_cut_method(self):
        ...

    def test_key_link_method_inverts_cut_method(self):
        ...

    def test_key_cut_method_inverts_link_method(self):
        ...


@unittest.skip("Not Implemented")
class KeyFormTestCase(unittest.TestCase):
    
    def test_keyform_partial_order_comparison(self):
        ...

    def test_keyform_key_reduction_method(self):
        ...

    def test_keyform_error_on_invalid_key_reduction(self):
        ...

if __name__ == "__main__":
    unittest.main()