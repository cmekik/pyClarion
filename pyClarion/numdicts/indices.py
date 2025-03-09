
from typing import overload, Any, Iterator
from itertools import product
from weakref import WeakSet
from math import prod

from pyClarion.numdicts.keyspaces import KSChild

from.keys import Key, KeyForm
from .keyspaces import KSPath, KSRoot, KSParent, KSObserver, KeyGroup, ks_root


class Index[R: KSRoot](KSObserver):
    root: R
    kf: KeyForm
    observers: WeakSet["IndexObserver"]
    groups: dict[int, KeyGroup]

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
        self.groups = {i: KeyGroup(levels[i], heights[i]) for i in leaves}
        for ksp in levels:
            if isinstance(ksp, KSParent):
                self.subscribe(ksp)

    @staticmethod
    def _init(root: KSRoot, keyform: KeyForm) \
        -> tuple[list[int], dict[int, int], list[KSPath]]:
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
        return prod([len(g) for g in self.groups.values()])

    def __iter__(self) -> Iterator[Key]:
        suites = [list(group) for group in self.groups.values()]
        for suite in product(*suites):
            result = self.kf.k
            # We append in reverse order because this preserves indices
            for i, s in zip(reversed(self.groups), reversed(suite)):
                if s: # s == Key() when h == 0, in which case don't append. 
                    result = result.link(s, i, ())
            yield result

    def __mul__(self, other: "Index") -> "Index":
        return Index(self.root, self.kf * other.kf)

    def requires(self, ksp: KSPath) -> bool:
        if ks_root(ksp) != self.root:
            raise ValueError("Incompatible keyspace: Non-identical roots")
        key = ~ksp
        for group in self.groups.values():
            matches = key.find_in(~group.ks)
            if matches:
                return True
        return False

    def depends_on(self, ksp: KSPath) -> bool:
        if ks_root(ksp) != self.root:
            raise ValueError("Incompatible keyspace: Non-identical roots")
        key = ~ksp
        for group in self.groups.values():
            leaf = ~group.ks
            matches = leaf.find_in(key)
            if matches and key.size <= leaf.size + group.h:
                return True
        return False        
    
    def on_del(self, parent: KSParent, child: KSChild) -> None:
        if self.requires(child):
            raise RuntimeError(f"Cannot delete key {~child}: "
                f"Required by index {self}")
        if self.depends_on(child):
            for observer in self.observers:
                observer.on_del(self, ~child)


class IndexObserver:
    __slots__ = ("__weakref__",)

    def register(self, index: Index) -> None:
        index.observers.add(self)
    
    def on_add(self, index: Index, key: Key) -> None:
        pass

    def on_del(self, index: Index, key: Key) -> None:
        pass