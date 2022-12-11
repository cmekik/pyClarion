from typing import Union, List, Dict, Tuple, Sequence, Generator, TypeVar, Optional

from ..numdicts import NumDict
from ..base.constructs import Process
from ..base.symbols import F, D, V


class Receptors(Process):
    """A perceptual input channel."""
    _spec: Union[List[str], Dict[str, List[str]]]
    _data: NumDict[F]
    _reprs: NumDict[F]

    def __init__(
        self, 
        path: str = "",
        spec: Union[None, List[str], Dict[str, List[str]]] = None
    ) -> None:
        super().__init__(path)
        self._spec = spec or []
        self.__validate()
        self._data = NumDict()
        self._reprs = NumDict({f: 1.0 for f in self._init_reps()})

    def __validate(self) -> None:
        if len(self._spec) == 0:
            raise ValueError(f"Arg spec must define at least one feature.")
        if isinstance(self._spec, list):
            for d in self._spec: 
                F.validate(i=d)
        else:
            assert isinstance(self._spec, dict)
            for d, l in self._spec.items():
                if 0 < len(l):
                    for v in l: 
                        F.validate(i=(d, v))
                else:
                    F.validate(i=d)
                
    def initial(self) -> Tuple[NumDict[F], NumDict[F]]:
        return NumDict(), self._reprs

    def call(self) -> Tuple[NumDict[F], NumDict[F]]:
        data, self._data = self._data, NumDict()
        return data, self._reprs

    T = TypeVar("T", str, Tuple[str, str])
    def stimulate(self, data: Union[List[T], Dict[T, float]]) -> None:
        """
        Set perceptual stimulus levels for defined perceptual features.
        
        :param data: Stimulus data. If list, each entry is given an activation 
            value of 1.0. If dict, each key is set to an activation level equal 
            to its value.
        """
        if isinstance(data, list):
            self._data = NumDict({f: 1.0 for f in self._fseq(data)})
        elif isinstance(data, dict):
            fspecs, strengths = zip(*data.items())
            fseq = self._fseq(fspecs) # type: ignore
            self._data = NumDict({f: v for f, v in zip(fseq, strengths)})
        else:
            raise ValueError("Stimulus spec must be list or dict, "
                f"got {type(data)}.")

    T = TypeVar("T", str, Tuple[str, str])
    def _fseq(self, data: Sequence[T]) -> Generator[F[T], None, None]:
        for x in data:
            f = F(i=x, p=self.path)
            if f not in self._reprs:
                raise ValueError(f"Unexpected stimulus '{x}'")
            yield f

    def _init_reps(self) -> Tuple[F, ...]:
        if isinstance(self._spec, list):
            return tuple(F(d, p=self.path) for d in self._spec)
        else:
            assert isinstance(self._spec, dict)
            return tuple(self._reps_from_dict())

    def _reps_from_dict(self): 
        assert isinstance(self._spec, dict)   
        for d, l in self._spec.items():
            if 0 < len(l): 
                for v in l: 
                    yield F((d, v), p=self.path)
            else: 
                yield F(d, p=self.path)


class Actions(Process):
    """Represents external actions."""
    _spec: Dict[str, List[str]]
    _rwds: NumDict[D]
    _cmds: NumDict[V]

    def __init__(
        self, path: str = "", spec: Optional[Dict[str, List[str]]] = None
    ) -> None:
        super().__init__(path)
        self._spec = spec or {}
        self.__validate()
        self._rwds = NumDict({f: 0.0 for f in self._init_rwds()})
        self._cmds = NumDict({f: 1.0 for f in self._init_cmds()})

    def __validate(self) -> None:
        if len(self._spec) == 0:
            raise ValueError("At least one action dimension must be defined.")
        for d, l in self._spec.items():
            if not len(l):
                raise ValueError(f"Empty value list for dimension '{d}'.")
            for v in l:
                    F.validate((d, v))

    def initial(self) -> Tuple[NumDict[D], NumDict[V]]:
        return NumDict(), self._cmds

    def call(self) -> Tuple[NumDict[D], NumDict[V]]:
        r, self._rwds = self._rwds, NumDict()
        return r, self._cmds

    def parse_actions(self, a: NumDict[F]) -> Dict[str, str]:
        """Return a dictionary of selected action values for each dimension"""
        result = {}
        for f in self._cmds:
            if a.isolate(key=f).isclose(1.0).c == 1.0:
                d, v = f.i
                if d in result:
                    raise ValueError(f"Multiple values for action dim '{d}'")
                result[d] = v
        return result

    def reward(self, r: Dict[str, float]) -> None:
        self._rwds = NumDict({d: r.get(d.i, 0.0) for d in self._init_rwds()})

    def _init_cmds(self) -> Tuple[V, ...]:
        return tuple(F((d, v), p=self.path) for d, vs in self._spec.items() 
            for v in vs)

    def _init_rwds(self) -> Tuple[D, ...]:
        return tuple(F(d, p=self.path) for d in self._spec)
