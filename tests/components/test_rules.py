import unittest

from pyClarion import rule, chunk, chunks, ConstructType
from pyClarion.components.rules import ActionRules, Rules
from pyClarion import nd


class TestActionRules(unittest.TestCase):

    def test_call_returns_empty_numdict_when_no_rules_exist(self):

        rules = Rules(max_conds=1)

        inputs = {
            chunks(1): nd.NumDict({
                chunk("Condition A"): .7,
                chunk("Condition B"): .2,
                chunk("Condition C"): .6,
                chunk("Condition D"): .5,
                chunk("Condition E"): .3
            }, default=0)
        }

        action_rules = ActionRules(
            source=chunks(1), 
            rules=rules, 
            temperature=1 # high temperature to ensure variety
        )
        strengths = action_rules.call(inputs)

        self.assertEqual(len(strengths), 0, msg="Unexpected items in output.")

    def test_call_activates_unique_action_and_rule_pair(self):

        rules = Rules(max_conds=1)
        rules.define(rule("A"), chunk("Action 1"), chunk("Condition A"))
        rules.define(rule("B"), chunk("Action 1"), chunk("Condition B"))
        rules.define(rule("C"), chunk("Action 1"), chunk("Condition C"))
        rules.define(rule("D"), chunk("Action 2"), chunk("Condition D"))
        rules.define(rule("E"), chunk("Action 2"), chunk("Condition E"))

        inputs = {
            chunks(1): nd.NumDict({
                chunk("Condition A"): .7,
                chunk("Condition B"): .2,
                chunk("Condition C"): .6,
                chunk("Condition D"): .5,
                chunk("Condition E"): .3
            }, default=0)
        }

        action_rules = ActionRules(
            source=chunks(1), 
            rules=rules, 
            temperature=1 # high temperature to ensure variety
        )
        strengths = action_rules.call(inputs)
        above_zero = nd.threshold(strengths, th=0.0)

        self.assertEqual(
            len(above_zero), 2, 
            msg="Expected at most two items above zero activation."
        )
        if chunk("Action 1") in above_zero:
            self.assertTrue(
                rule("A") in above_zero or 
                rule("B") in above_zero or 
                rule("C") in above_zero,
                msg="Unexpected rule paired with Action 1."   
            )
        if chunk("Action 2") in above_zero:
            self.assertTrue(
                rule("D") in above_zero or 
                rule("E") in above_zero,
                msg="Unexpected rule paired with Action 2."   
            )

    def test_rule_selection_follows_boltzmann_distribution(self):
        
        rules = Rules(max_conds=1)
        rules.define(rule("A"), chunk("Action 1"), chunk("Condition A"))
        rules.define(rule("B"), chunk("Action 1"), chunk("Condition B"))
        rules.define(rule("C"), chunk("Action 1"), chunk("Condition C"))
        rules.define(rule("D"), chunk("Action 2"), chunk("Condition D"))
        rules.define(rule("E"), chunk("Action 2"), chunk("Condition E"))

        inputs = {
            chunks(1): nd.NumDict({
                chunk("Condition A"): .7,
                chunk("Condition B"): .2,
                chunk("Condition C"): .6,
                chunk("Condition D"): .5,
                chunk("Condition E"): .3
            }, default=0)
        }

        get_rule = lambda c: rule(c.cid[-1])

        action_rules = ActionRules(
            source=chunks(1), 
            rules=rules, 
            temperature=.1 # relatively high temperature to ensure variety
        )

        expected = nd.transform_keys(inputs[chunks(1)], func=get_rule)
        expected = nd.boltzmann(expected, t=.1)
        expected = nd.with_default(expected, default=None)

        N = 100

        selected = []
        is_rule = lambda sym: sym.ctype in ConstructType.rule
        for _ in range(N):
            strengths = action_rules.call(inputs)
            s = nd.keep(strengths, func=is_rule)
            s = nd.threshold(s, th=0)
            s = s.constant(val=1)
            s = nd.with_default(s, default=0) 
            selected.append(s)

        counts = nd.ew_sum(*selected)
        terms = ((counts - (N * expected)) ** 2) / (N * expected) 
        chi_square_stat = nd.val_sum(terms)
        critical_value = 9.488 # for chi square w/ 4 df @ alpha = .05

        self.assertFalse(
            critical_value < chi_square_stat,
            msg="Chi square test significant at alpha = .05."
        )


if __name__ == "__main__":
    unittest.main()