from ...knowledge import Family, Sort, Atom, keyform
from ...numdicts import NumDict
from .base import LayerBase, Activation, Backprop


class Layer(LayerBase):
    def __init__(self, 
        name: str, 
        h_in: Sort, 
        h_out: Sort, 
        *, 
        f: Activation = NumDict.eye,
        train: Backprop.Train = Backprop.Train.ALL,
        lags: int = 0
    ) -> None:
        super().__init__(name, f, train)
        self.system.check_root(h_in, h_out)
        idx_in = self.system.get_index(keyform(h_in))
        idx_out = self.system.get_index(keyform(h_out))
        self._init(idx_in, idx_out, lags)


class SingleLayer(LayerBase):
    def __init__(self, 
        name: str, 
        d1: Family | Sort | Atom, 
        v1: Family | Sort,
        d2: Family | Sort | Atom | None = None,
        v2: Family | Sort | None = None,
        *, 
        f: Activation = NumDict.eye,
        train: Backprop.Train = Backprop.Train.ALL,
        lags: int = 0
    ) -> None:    
        d2 = d1 if d2 is None else d2
        v2 = v1 if v2 is None else v2
        super().__init__(name, f, train)
        self.system.check_root(d1, v1, d2, v2)
        idx_d1 = self.system.get_index(keyform(d1))
        idx_v1 = self.system.get_index(keyform(v1))
        idx_d2 = self.system.get_index(keyform(d2))
        idx_v2 = self.system.get_index(keyform(v2))
        self._init(idx_d1 * idx_v1, idx_d2 * idx_v2, lags)


class InputLayer(LayerBase):
    def __init__(self, 
        name: str, 
        d: Family | Sort | Atom, 
        v: Family | Sort,
        h: Sort, 
        *, 
        f: Activation = NumDict.eye,
        train: Backprop.Train = Backprop.Train.ALL,
        lags: int = 0
    ) -> None:
        super().__init__(name, f, train)
        self.system.check_root(d, v, h)
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        idx_h = self.system.get_index(keyform(h))
        self._init(idx_d * idx_v, idx_h, lags)


class OutputLayer(LayerBase):
    def __init__(self, 
        name: str, 
        h: Sort, 
        d: Family | Sort | Atom, 
        v: Family | Sort,
        *, 
        f: Activation = NumDict.eye,
        train: Backprop.Train = Backprop.Train.ALL,
        lags: int = 0
    ) -> None:
        super().__init__(name, f, train)
        self.system.check_root(h, d, v)
        idx_h = self.system.get_index(keyform(h))
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        self._init(idx_h, idx_d * idx_v, lags)