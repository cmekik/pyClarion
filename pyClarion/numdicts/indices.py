
from typing import overload, Any, Iterator
from itertools import product
from weakref import WeakSet

from.keys import Key, KeyForm
from .keyspaces import KSBase, KSRoot, KSParent, KSObserver, root, path


class Index[R: KSRoot](KSObserver):
    root: R
    kf: KeyForm
    observers: WeakSet["IndexObserver"]
    _leaves: list[int]
    _heights: dict[int, int]
    _levels: list[KSBase]

    @overload
    def __init__(self, root: R, form: KeyForm | Key | str) -> None:
        ...

    @overload
    def __init__(
        self, root: R, form: Key | str, tup: tuple[int, ...]
    ) -> None:
        ...

    def __init__(self, 
        root: R, 
        form: KeyForm | Key | str, 
        tup: tuple[int, ...] | None = None
    ) -> None:
        if isinstance(form, (Key, str)) and tup is not None:
            form = KeyForm(Key(form), tup)
        elif isinstance(form, (Key, str)):
            form = KeyForm.from_key(Key(form))
        elif not isinstance(form, KeyForm):
            raise TypeError("Unexpected input to Index.")
        form = form.strip # make sure the form contains no placeholders
        leaves, heights, levels = self._init(root, form)
        self.root = root
        self.kf = form 
        self.observers = WeakSet()
        self._leaves = leaves
        self._heights = heights
        self._levels = levels
        for ksp in self._levels:
            if isinstance(ksp, KSParent):
                self.subscribe(ksp)

    @staticmethod
    def _init(root: KSRoot, keyform: KeyForm) \
        -> tuple[list[int], dict[int, int], list[KSBase]]:
        keyspaces, parents = [], []
        leaves, hs, heights = [], iter(keyform.h), {}
        for i, (label, degree) in enumerate(keyform.k):
            if i == 0:
                keyspaces.append(root)
                parents.extend([-1, *([i] * degree)])
            else:
                level = keyspaces[parents[i]][label]
                keyspaces.append(level)
                parents.extend([i] * degree)
            if degree == 0:
                heights[i] = next(hs)
                leaves.append(i)
        return leaves, heights, keyspaces 

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Index):
            return self.root is other.root and self.kf == other.kf
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Index):
            return self.root is other.root and self.kf < other.kf
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.root, self.kf))

    def __contains__(self, key: Key) -> bool:
        return key in self.kf and key in self.root

    def __len__(self) -> int:
        l = 0
        for _ in self:
            l += 1
        return l

    def __iter__(self) -> Iterator[Key]:
        its = (self._levels[i]._iter_(self._heights[i]) for i in self._leaves)
        suites = [list(it) for it in its]
        heights = [self._heights[i] for i in self._leaves]
        for h, l in zip(heights, suites): 
            if h == 0 and not l: 
                l.append(Key()) # ensures suite not empty when h=0
        for suite in product(*suites):
            result = self.kf.k
            for i, s in zip(reversed(self._leaves), reversed(suite)):
                if s:
                    result = result.link(s, i, ())
            yield result

    def __mul__(self, other: "Index") -> "Index":
        return Index(self.root, self.kf * other.kf)

    def depends_on(self, ksp: KSBase) -> bool:
        if root(ksp) != self.root:
            raise ValueError("Incompatible keyspace: Non-identical roots")
        key, i = path(ksp), 0
        while key and not key < self.kf.k:
            key, _ = key.cut(key.size - 1)
            i += 1
        return bool(key)        


class IndexObserver:
    __slots__ = ("__weakref__",)

    def register(self, index: Index) -> None:
        index.observers.add(self)
    
    def on_add(self, index: Index, key: Key) -> None:
        pass

    def on_remove(self, index: Index, key: Key) -> None:
        pass