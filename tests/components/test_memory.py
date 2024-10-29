import unittest
from datetime import timedelta

from pyClarion import Agent, UpdateSort, path
from pyClarion.knowledge import Family, Chunks, Chunk
from pyClarion.components.memory import BaseLevel


class BaseLevelTestCase(unittest.TestCase):

    def test_base_level(self):    
        efam = Family()
        pfam = Family()
        tfam = Family()
        chunks = Chunks()
        tfam.c = chunks
        with Agent("agent") as agent:
            agent.system.root.p = pfam
            agent.system.root.e = efam
            agent.system.root.s = tfam
            base_levels = BaseLevel("bla", pfam, efam, chunks)
        
        agent.system.user_update(
            UpdateSort(chunks, add=((f"test_chunk", Chunk()),)))
        for i in range(1,11):
            agent.breakpoint(dt=timedelta(seconds=i))
        
        while agent.system.queue:
            event = agent.system.advance()
            if event.source == agent.breakpoint:
                base_levels.update()
            if event.source == base_levels.update:
                print(event.time, base_levels.main[path(chunks["test_chunk"])])
        

if __name__ == "__main__":
    unittest.main()