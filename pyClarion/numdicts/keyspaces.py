from typing import get_type_hints, Any, Iterator, Self
from itertools import combinations, product
from sys import getrefcount
from weakref import WeakSet

from .exc import ValidationError
from .keys import Key, KeyForm


class KeySpace:
    _name_: Key
    _parent_: "KeySpace | None"
    _internal_refs_: int
    _members_: dict[Key, "KeySpace"]
    _products_: dict[tuple[Key, ...], "ProductSpace"]
    _subspaces_: WeakSet["Index"]
    _required_: frozenset[Key]

    def __init__(self):
        self._name_ = Key()
        self._parent_ = None
        self._members_ = {}
        self._products_ = {}
        self._subspaces_ = WeakSet()
        cls = type(self)
        for name, typ in get_type_hints(cls).items():
            if isinstance(typ, type) and issubclass(typ, KeySpace):
                setattr(self, name, typ())
        self._required_ = frozenset(self._members_)

    def __contains__(self, key: str | Key) -> bool:
        k = Key(key)
        keyspace = self
        trunk, branches = k.split()
        while trunk:
            head, trunk = trunk.cut(1) 
            try:
                keyspace = keyspace._members_[head]
            except KeyError:
                return False
        if branches:
            while branches.size:
                branches, branch = branches.cut(0, (0,))
                if branch not in keyspace:
                    return False
        return True

    def __setattr__(self, name: str, value: Any) -> None:
        if name != "_parent_" and isinstance(value, KeySpace):
            key = Key(name)
            if key in self._members_:
                raise ValidationError(
                    f"Cannot overwrite preexisting key '{name}'")
            if value._parent_ is not None:
                raise ValidationError(f"KeySpace already has parent")
            self._members_[key] = value
            value._name_ = key
            value._parent_ = self
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        key = Key(name)
        if key in self._members_: 
            if key in self._required_:
                raise ValidationError(f"Cannot remove required key '{name}'")
            keyspace = self._members_[key]
            if self._members_[key]._subspaces_: 
                raise ValidationError(f"Key {name} has dependent subspaces")
            del self._members_[key]
            self._unbind_(key)
            subspaces, keyspace = set(), self
            while keyspace._parent_ is not None:
                subspaces.update(keyspace._subspaces_)
                keyspace = keyspace._parent_
            for subspace in subspaces:
                subspace.deletions += 1
        super().__delattr__(name)

    def _bind_(self, *keys: str | Key) -> None:
        keys = tuple(Key(key) for key in keys)
        if any(key not in self._members_ for key in keys):
            raise ValueError()
        N = len(keys)
        if N <= 1:
            return
        indices = list(range(N))
        for r in range(2, N + 1):
            for m in combinations(indices, r):
                product = ProductSpace(self, *(keys[i] for i in m))
                self._products_.setdefault(tuple(keys[i] for i in m), product)

    def _unbind_(self, *keys: str | Key) -> None:
        for product in list(self._products_):
            if len(product) < len(keys):
                continue
            it = iter(product)
            for key_i in keys:
                for key_j in it:
                    if key_i == key_j:
                        break
                else:
                    break
            else:
                del self._products_[product]            

    def _iter_(self, h: int) -> Iterator[Key]:
        if not self._members_:
            return
        if h <= 1:
            yield from self._members_
            for product in self._products_.values():
                yield from product._iter_(h)
        else:
            for key, keyspace in self._members_.items():
                for suite in keyspace._iter_(h - 1):
                    yield key.link(suite, 1)
            for product in self._products_.values():
                yield from product._iter_(h)

    def __getattr__(self: Self, name: str) -> Self:
        if name.startswith("_") and name.endswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
        new = type(self)()
        self.__setattr__(name, new)
        return new


def bind(keyspace: KeySpace, *keyspaces: KeySpace) -> None:
    parent = keyspace._parent_
    if parent is None:
        raise ValidationError("Cannot bind root keyspace")
    if any(ksp._parent_ is not parent for ksp in keyspaces):
        raise ValidationError("Bound keyspaces must have identical parents")
    parent._bind_(keyspace._name_, *(ksp._name_ for ksp in keyspaces))


def unbind(keyspace: KeySpace, *keyspaces: KeySpace) -> None:
    parent = keyspace._parent_
    if parent is None:
        raise ValidationError("Cannot bind root keyspace")
    if any(ksp._parent_ is not parent for ksp in keyspaces):
        raise ValidationError("Bound keyspaces must have identical parents")
    parent._unbind_(keyspace._name_, *(ksp._name_ for ksp in keyspaces))


class ProductSpace:
    
    def __init__(self, keyspace: KeySpace, *keys: Key) -> None:
        self.keys = keys
        self.keyspaces = [keyspace._members_[k] for k in keys]
    
    def _iter_(self, h: int):
        if h <= 1:
            k = Key()
            for b in self.keys:
                k = k.link(b, 0)
            yield k
        else:
            its = [keyspace._iter_(h - 1) for keyspace in self.keyspaces]
            suites = [list(it) for it in its]
            for suite in product(*suites):
                k = Key()
                for b, s in zip(self.keys, suite):
                    k = k.link(b.link(s, 1), 0)
                yield k
    

class Index:

    def __init__(self, keyspace: KeySpace, keyform: KeyForm) -> None:
        self.keyspace = keyspace
        self.keyform = keyform 
        self.deletions = 0
        self._trace = self._init_trace()
        for ksp in self._trace[2]:
            ksp._subspaces_.add(self)

    def _init_trace(self):
        keyspaces, parents = [], []
        leaves, hs, heights = [], iter(self.keyform.h), {}
        for i, (label, degree) in enumerate(self.keyform.k):
            if i == 0:
                keyspaces.append(self.keyspace)
                parents.extend([-1, *([i] * degree)])
            else:
                level = keyspaces[parents[i]]
                try:
                    level = level._members_[Key(label)]
                except IndexError as e:
                    raise ValidationError(
                        f"Key '{self.keyform.k}' not defined") from e
                keyspaces.append(level)
                parents.extend([i] * degree)
            if degree == 0:
                try:
                    heights[i] = next(hs)
                except StopIteration as e:
                    raise ValidationError("Too few height params") from e
                leaves.append(i)
        return leaves, heights, keyspaces 

    def __contains__(self, key: Key) -> bool:
        return key in self.keyform and key in self.keyspace

    def __iter__(self) -> Iterator[Key]:
        leaves, heights, keyspaces = self._trace
        its = (keyspaces[i]._iter_(heights[i]) for i in leaves)
        suites = [list(it) for it in its]
        for suite in product(*suites):
            result = self.keyform.k
            for i, s in zip(reversed(leaves), reversed(suite)):
                result = result.link(s, i, ())
            yield result
