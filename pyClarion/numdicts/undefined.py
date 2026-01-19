from typing import Final, final

@final
class _Undefined:
    __slots__ = ()
    def __repr__(self):
        return "Undefined"
    

Undefined: Final[_Undefined] = _Undefined()