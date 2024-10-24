import unittest

from pyClarion import Agent
from pyClarion.knowledge import Family, Atoms, Atom, Dyads
from pyClarion.components.elementary import Input
from pyClarion.components.top_level import ChunkStore


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
            store = ChunkStore("store", root.s, bl)
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

if __name__ == "__main__":
    unittest.main()