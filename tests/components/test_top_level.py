import unittest
from datetime import timedelta

from pyClarion import Agent
from pyClarion.knowledge import Family, Atoms, Atom
from pyClarion.components.elementary import Input
from pyClarion.components.stores import ChunkStore, RuleStore
from pyClarion.components.rules import FixedRules

@unittest.skip("very broken")
class ChunkStoreTestCase(unittest.TestCase):
    def test_chunk_store(self):
        class Color(Atoms):
            red: Atom
            grn: Atom
            blu: Atom
        class Shape(Atoms):
            circ: Atom
            squr: Atom
            tria: Atom
        class IO(Atoms):
            ipt: Atom
            opt: Atom
        
        s = Family()
        color = Color()
        shape = Shape()
        io = IO()
        s.color = color
        s.shape = shape
        s.io = io

        with Agent("a") as agent:
            root = agent.system.root; root.s = s
            input = Input("input", (root.s, root.s))
            store = ChunkStore("store", root.s, root.s, root.s)
            store.bu.input = input.main
        
        store.compile(
            "blue_triangle" ^
            + io.ipt ** color.blu
            + io.ipt ** shape.tria,

            "red_triangle" ^
            + io.ipt ** color.red
            + io.ipt ** shape.tria,
            
            "green_square" ^
            + io.ipt ** color.grn
            + io.ipt ** shape.squr)

        input.send(
            + io.ipt ** color.blu
            + io.ipt ** shape.tria)

        while agent.system.queue:
            agent.system.advance()        
        ...


@unittest.skip("very broken")
class RuleStoreTestCase(unittest.TestCase):

    def test_rule_store(self):
        class Heading(Atoms):
            nil: Atom
            left: Atom
            right: Atom
            up: Atom
            down: Atom
        class IO(Atoms):
            food: Atom
            danger: Atom
            move: Atom

        s = Family()
        io = IO()
        heading = Heading() 
        s.io = io; s.heading = heading

        rules = [
            "avoid_danger" ^
            + io.danger ** heading("H")
            >>
            - io.move ** heading("H"),

            "approach_food" ^
            + io.food ** heading("H")
            >>
            + io.move ** heading("H")]
        
        with Agent("agent") as agent:
            root = agent.system.root; root.s = s
            input = Input("input", (root.s, root.s))
            store = RuleStore("rules", root.s, root.s, root.s, root.s)
            store.lhs.bu.input = input.main

        store.compile(*rules)
        input.send(+ io.danger ** heading.up)

        while agent.system.queue:
            agent.system.advance()
        ...


@unittest.skip("very broken")
class FixedRuleTestCase(unittest.TestCase):

    def test_rule_store(self):
        class Heading(Atoms):
            nil: Atom
            left: Atom
            right: Atom
            up: Atom
            down: Atom
        class IO(Atoms):
            food: Atom
            danger: Atom
            move: Atom

        s = Family()
        io = IO()
        heading = Heading() 
        s.io = io; s.heading = heading

        rules = [
            "avoid_danger" ^
            + io.danger ** heading("H")
            >>
            - io.move ** heading("H"),

            "approach_food" ^
            + io.food ** heading("H")
            >>
            + io.move ** heading("H")]
        
        with Agent("agent") as agent:
            root = agent.system.root; root.s = s; root.p = Family()
            input = Input("input", (root.s, root.s))
            frs = FixedRules("frs", root.p, root.s, root.s, root.s, root.s, sd=1e-4)
            frs.rules.lhs.bu.input = input.main

        frs.rules.compile(*rules)
        input.send(+ io.danger ** heading.up)
        frs.trigger()

        event = None
        import pprint
        while agent.system.queue:
            event = agent.system.advance()
            print("Event:", event.source.__qualname__, f"(# {event.number})")
            print("Input (BL):")
            pprint.pprint(input.main[0].d)
            print("Input (TL):")
            pprint.pprint(frs.rules.lhs.bu.main[0].d)
            print("Output (TL):")
            pprint.pprint(frs.choice.main[0].d)
            print("Output (BL):")
            pprint.pprint(frs.rules.rhs.td.main[0].d)
            print()
        ...


if __name__ == "__main__":
    unittest.main()