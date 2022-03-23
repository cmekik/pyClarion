from typing import Sequence, Tuple

from .. import dev as cld
from ..base import feature
from ..numdicts import NumDict


class Drives(cld.Process):
    """
    Maintains drive strengths.
    
    Houses deficits and baselines.
    """

    initial = NumDict()

    def __init__(self, spec: Sequence[str]) -> None:
        for f in spec:
            if not cld.ispath(f):
                raise ValueError(f"Invalid drive name '{f}'")
        self.dspec = spec 
        self.deficits: NumDict[feature] = NumDict()
        self.baselines: NumDict[feature] = NumDict()

    def call(
        self, stimuli: NumDict[feature], gains: NumDict[feature]
    ) -> NumDict[feature]:
        return self.baselines.add(
            (self.deficits
                .mul_from(stimuli, kf=cld.eye)
                .mul_from(gains, kf=cld.eye)))

    @property
    def reprs(self) -> Tuple[NumDict[feature], ...]:
        return tuple(feature(cld.prefix(d, self.prefix)) for d in self.dspec)
