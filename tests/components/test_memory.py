import unittest
from datetime import timedelta

from pyClarion import Agent, UpdateSort
from pyClarion.knowledge import Family, Chunks, Chunk
from pyClarion.components.stats import BaseLevel

@unittest.skip("very broken")
class BaseLevelTestCase(unittest.TestCase):

    def test_base_level(self):    
        efam = Family()
        pfam = Family()
        tfam = Family()
        chunks = Chunks()
        tfam["c"] = chunks
        with Agent("agent", p=pfam, e=efam, t=tfam) as agent:
            base_levels = BaseLevel("bla", pfam, efam, chunks)
        
        agent.system.user_update(
            UpdateSort(chunks, add=(Chunk({}),)))
        for i in range(1,11):
            agent.breakpoint(dt=timedelta(seconds=i))
        
        while agent.system.queue:
            event = agent.system.advance()
            if event.source == agent.breakpoint:
                base_levels.update()
            if event.source == base_levels.update:
                print(event.time, base_levels.main[0][~chunks["test_chunk"]])
        

if __name__ == "__main__":
    unittest.main()