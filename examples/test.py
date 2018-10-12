from typing import Tuple, Dict, Union, NewType, List
from pyClarion.base.symbols import Chunk

T = List[Tuple[Chunk, Tuple[Tuple[Chunk, float], ...]]]

rules: T = [
    (
        Chunk("Apple"), 
        (
            (Chunk("Granny Smith"), 1.),
        )
    ),
    (
        Chunk("Apple"),
        (
            (Chunk("Fuji"), 1.),
        )
    ),
    (
        Chunk("Apple"),
        (
            (Chunk("Macintosh"), 1.),
        )
    )
]
d = [{ch: dict(weights)} for ch, weights in rules]