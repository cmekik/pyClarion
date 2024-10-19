from typing import get_type_hints, Any, Iterator, Type, overload, Self, cast
from itertools import combinations, product
from weakref import WeakSet

from .exc import ValidationError
from .keys import Key, KeyForm, sig_cache


class KeySpace:
    _name_: Key
    _parent_: "KeySpace | None"
    _members_: dict[Key, "KeySpace"]
    _products_: dict[tuple[Key, ...], "ProductSpace"]
    _indices_: WeakSet["Index"]
    _required_: frozenset[Key]

    def __init__(self):
        self._name_ = Key()
        self._parent_ = None
        self._members_ = {}
        self._products_ = {}
        self._indices_ = WeakSet()
        cls = type(self)
        for name, typ in get_type_hints(cls).items():
            if isinstance(typ, type) and issubclass(typ, KeySpace):
                setattr(self, name, typ())
        self._required_ = frozenset(self._members_)

    def __contains__(self, key: str | Key) -> bool:
        k, keyspace = Key(key), self
        while k and k[0][1] <= 1:
            node, k = k.cut(1)
            try:
                keyspace = keyspace._members_[node]
            except KeyError:
                return False
        else:
            while k.size:
                k, branch = k.cut(0, (0,))
                if branch not in keyspace:
                    return False
        return True
    
    def __getitem__(self, name: str) -> "KeySpace":
        if not name.isidentifier():
            raise ValueError(
                f"Argument {repr(name)} is not a valid Python identifier")
        return getattr(self, name)

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
        if key in self._required_:
            raise ValidationError(f"Cannot remove required key '{name}'")
        if key in self._members_ and self._members_[key]._indices_: 
            raise ValidationError(f"Key {name} has dependent subspaces")
        super().__delattr__(name)
        if key in self._members_: 
            keyspace = self._members_[key]
            del self._members_[key]
            self._unbind_(key)
            subspaces, keyspace = set(), self
            while keyspace._parent_ is not None:
                subspaces.update(keyspace._indices_)
                keyspace = keyspace._parent_
            for subspace in subspaces:
                subspace.deletions += 1

    def _bind_(self, *keys: str | Key) -> None:
        keys = tuple(Key(key) for key in keys)
        if any(key not in self._members_ for key in keys):
            raise ValueError(f"Undefined child keys in sequence {keys}")
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
        if h <= 0:
            return
        if h == 1:
            yield from self._members_
            for product in self._products_.values():
                yield from product._iter_(h)
        else:
            for key, keyspace in self._members_.items():
                for suite in keyspace._iter_(h - 1):
                    yield key.link(suite, 1)
            for product in self._products_.values():
                yield from product._iter_(h)

    def __getattr__(self, name: str) -> "KeySpace":
        if name.startswith("_") and name.endswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
        new = type(self)()
        self.__setattr__(name, new)
        return new


class GenericKeySpace[C: KeySpace](KeySpace):
    
    @property
    def _child_type_(self) -> Type[C]:
        raise NotImplementedError()

    def __getitem__(self, name: str) -> C:
        return cast(C, super().__getitem__(name))

    def __setattr__(self, name: str, value: Any) -> None:
        if (name != "_parent_" and isinstance(value, KeySpace) 
            and not isinstance(value, self._child_type_)):
            raise TypeError(f"Keyspace of type {type(self).__name__} "
                f"expected subspace of type {self._child_type_.__name__} "
                f"but got {type(value).__name__} instead")
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> C:
        if name.startswith("_") and name.endswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
        new = self._child_type_()
        self.__setattr__(name, new)
        return new


def root(ksp: KeySpace) -> KeySpace:
    ret = ksp
    while ret._parent_ is not None:
        ret = ret._parent_
    return ret


def path(ksp: KeySpace) -> Key:
    ret = ksp._name_
    while ksp._parent_ is not None:
        assert len(ksp._name_) == 2
        ksp = ksp._parent_
        ret = ksp._name_.link(ret, ksp._name_.size)
    return ret


def parent(ksp: KeySpace) -> KeySpace:
    if ksp._parent_ is None:
        raise ValueError("Keyspace is root (has no parent)")
    return ksp._parent_


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

    @overload
    def __new__(cls: type[Self], root: KeySpace, form: KeyForm) -> Self:
        ...

    @overload
    def __new__(cls: type[Self], 
        root: KeySpace, 
        form: Key | str, 
        tup: tuple[int, ...]
    ) -> Self:
        ...

    @sig_cache # TODO: Make cache weak
    def __new__(
        cls: Type[Self], 
        root: KeySpace, 
        form: KeyForm | Key | str, 
        tup: tuple[int, ...] | None = None) -> Self:
        return super().__new__(cls)

    @overload
    def __init__(self, root: KeySpace, form: KeyForm) -> None:
        ...

    @overload
    def __init__(self, 
        root: KeySpace, 
        form: Key | str, 
        tup: tuple[int, ...]
    ) -> None:
        ...

    def __init__(self, 
        root: KeySpace, 
        form: KeyForm | Key | str, 
        tup: tuple[int, ...] | None = None
    ) -> None:
        if isinstance(form, (Key, str)):
            if tup is None:
                raise ValueError("No depth tuple passed in")
            form = KeyForm(Key(form), tup)
        self.root = root
        self.keyform = form 
        self.deletions = 0
        self._trace = self._init_trace()
        self._auto_bind()
        for ksp in self._trace[2]:
            ksp._indices_.add(self)

    def _auto_bind(self):
        keyspaces, parents = [], []
        s = 0
        for i, (label, degree) in enumerate(self.keyform.k):
            if i == 0:
                level = self.root
                keyspaces.append(level)
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
            if 1 < degree:
                children = [self.keyform.k[s + j + 1][0] for j in range(degree)]
                level._bind_(*children)
            s += degree

    def _init_trace(self):
        keyspaces, parents = [], []
        leaves, hs, heights = [], iter(self.keyform.h), {}
        for i, (label, degree) in enumerate(self.keyform.k):
            if i == 0:
                keyspaces.append(self.root)
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
        return key in self.keyform and key in self.root

    def __iter__(self) -> Iterator[Key]:
        leaves, heights, keyspaces = self._trace
        its = (keyspaces[i]._iter_(heights[i]) for i in leaves)
        suites = [list(it) for it in its]
        for suite in product(*suites):
            result = self.keyform.k
            for i, s in zip(reversed(leaves), reversed(suite)):
                result = result.link(s, i, ())
            yield result

    def depends_on(self, ksp: KeySpace) -> bool:
        if root(ksp) != self.root:
            raise ValueError("Incompatible keyspace: Non-identical roots")
        key, i = path(ksp), 0
        kf = KeyForm(key, (i,))
        while key and not kf < self.keyform.k:
            key, _ = key.cut(key.size)
            i += 1
            kf = KeyForm(key, (i,))
        return bool(key)        
