import unittest
from datetime import timedelta

from pyClarion import Agent
from pyClarion.knowledge import Family, Atoms, Atom, Dyads, Var
from pyClarion.components.elementary import Input
from pyClarion.components.top_level import ChunkStore, RuleStore


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
            bl = Dyads(root.s, root.s) 
            input = Input("input", bl)
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

    def test_rule_store(self):
        class Heading(Atoms):
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

        H = Var("H", heading)

        rules = [
            "avoid_danger" ^
            + io.danger ** H
            >>
            - io.move ** H,

            "approach_food" ^
            + io.food ** H
            >>
            + io.move ** H]
        
        with Agent("agent") as agent:
            root = agent.system.root; root.s = s
            bl = Dyads(root.s, root.s)
            input = Input("input", bl)
            store = RuleStore("rules", root.s, root.s, root.s)
            store.lhs.bu.input = input.main

        store.compile(*rules)
        input.send(+ io.danger ** heading.up, dt=timedelta(milliseconds=1))

        while agent.system.queue:
            agent.system.advance()
        ...



if __name__ == "__main__":
    unittest.main()