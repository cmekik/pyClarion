from typing import get_type_hints, Any, Iterator, Type, ClassVar, overload
from itertools import product
from weakref import WeakSet
import warnings

from .exc import ValidationError
from .keys import Key, KeyForm


class KeySpaceBase[P: "KeySpaceBase", T: "KeySpaceBase"]:
    _name_: Key
    _parent_: P | None = None
    _ptype_: Type[P]
    _mtype_: Type[T]
    _members_: dict[Key, T]
    _indices_: WeakSet["Index"]
    _required_: frozenset[Key]

    def __init__(self, ptype: Type[P], mtype: Type[T]):
        self._name_ = Key()
        self._ptype_ = ptype
        self._mtype_ = mtype
        self._members_ = {}
        self._indices_ = WeakSet()
        cls = type(self)
        for name, typ in get_type_hints(cls).items():
            if isinstance(typ, type) and issubclass(typ, KeySpace):
                setattr(self, name, typ())
        self._required_ = frozenset(self._members_)

    def __iter__(self) -> Iterator[str]:
        for k in self._members_:
            yield k[1][0]

    def __contains__(self, key: "str | Key | KeySpaceBase") -> bool:
        if isinstance(key, KeySpaceBase):
            return key in self._members_.values()
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

    def __getitem__(self, name: str) -> T:
        if name.isidentifier():
            return getattr(self, name)
        raise ValueError(f"Argument {repr(name)} is not a Python identifier")
    
    def __setitem__(self, name: str, value: T) -> None:
        if isinstance(value, self._mtype_):
            setattr(self, name, value)
        raise TypeError(f"{type(self).__name__}.__setitem__ expects value " 
            f"of type {self._mtype_.__name__}, but got a value of type "
            f"{type(value).__name__} instead")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_parent_":
            if self._parent_ is not None:
                raise ValidationError("Keyspace already has parent")
            if not isinstance(value, self._ptype_):
                raise TypeError()
        elif isinstance(value, KeySpaceBase):
            if (key := Key(name)) in self._members_:
                raise ValidationError(f"Cannot replace existing key '{name}'")
            if not isinstance(value, self._mtype_):
                raise TypeError()
            if not isinstance(self, value._ptype_):
                raise TypeError()
            self._members_[key] = value
            value._name_ = key
            value._parent_ = self
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if (key := Key(name)) in self._required_:
            raise ValidationError(f"Cannot remove required key '{name}'")
        if key in self._members_ and self._members_[key]._indices_: 
            raise ValidationError(f"Key {name} has dependent subspaces")
        super().__delattr__(name)
        if key in self._members_: 
            keyspace = self._members_[key]
            del self._members_[key]
            subspaces, keyspace = set(), self
            while keyspace._parent_ is not None:
                subspaces.update(keyspace._indices_)
                keyspace = keyspace._parent_
            for subspace in subspaces:
                subspace.deletions += 1


class KeySpace(KeySpaceBase["KeySpaceBase", "KeySpaceBase"]):

    def __init__(self):
        super().__init__(type(self), type(self))           

    def __getattr__(self, name: str) -> "KeySpace":
        if name.startswith("_") and name.endswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
        new = type(self)()
        self.__setattr__(name, new)
        return new


def root(ksp: KeySpaceBase) -> KeySpaceBase:
    ret = ksp
    while ret._parent_ is not None:
        ret = ret._parent_
    return ret


def path(ksp: KeySpaceBase) -> Key:
    ret = ksp._name_
    while ksp._parent_ is not None:
        assert len(ksp._name_) == 2
        ksp = ksp._parent_
        ret = ksp._name_.link(ret, ksp._name_.size)
    return ret


def parent(ksp: KeySpaceBase) -> KeySpaceBase:
    if ksp._parent_ is None:
        raise ValueError("Keyspace is root (has no parent)")
    return ksp._parent_


def _iter(ksp: KeySpaceBase, h: int) -> Iterator[Key]:
    if not ksp._members_:
        return
    if h <= 0:
        return
    if h == 1:
        yield from ksp._members_
    else:
        for key, child in ksp._members_.items():
            for suite in _iter(child, h - 1):
                yield key.link(suite, 1)


def bind(keyspace: KeySpace, *keyspaces: KeySpace) -> None:
    warnings.warn(
        "Function 'bind()' is deprecated as explicit declaration of compound "
        "keys is no longer required. This function does nothing but remains "
        "available for backwards compatibility.", DeprecationWarning)
    ...


def unbind(keyspace: KeySpace, *keyspaces: KeySpace) -> None:
    warnings.warn(
        "Function 'unbind()' is deprecated as explicit declaration of compound "
        "keys is no longer required. This function does nothing but remains "
        "available for backwards compatibility.", DeprecationWarning)
    ...
    

class Index[T: KeySpaceBase]:
    root: T
    keyform: KeyForm
    deletions: int
    _leaves: list[int]
    _heights: dict[int, int]
    _levels: list[KeySpaceBase]

    @overload
    def __init__(self, root: T, form: KeyForm | Key | str) -> None:
        ...

    @overload
    def __init__(
        self, root: T, form: Key | str, tup: tuple[int, ...]
    ) -> None:
        ...

    def __init__(self, 
        root: T, 
        form: KeyForm | Key | str, 
        tup: tuple[int, ...] | None = None
    ) -> None:
        if isinstance(form, (Key, str)) and tup is not None:
            form = KeyForm(Key(form), tup)
        elif isinstance(form, (Key, str)):
            form = KeyForm.from_key(Key(form))
        elif not isinstance(form, KeyForm):
            raise TypeError("Unexpected input to Index.")
        leaves, heights, levels = self._init(root, form)
        self.root = root
        self.keyform = form 
        self.deletions = 0
        self._leaves = leaves
        self._heights = heights
        self._levels = levels
        for ksp in self._levels:
            ksp._indices_.add(self)

    @staticmethod
    def _init(root: KeySpaceBase, keyform: KeyForm) \
        -> tuple[list[int], dict[int, int], list[KeySpaceBase]]:
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
            return self.root is other.root and self.keyform == other.keyform
        return NotImplemented
    
    def __hash__(self) -> int:
        return hash((self.root, self.keyform))

    def __contains__(self, key: Key) -> bool:
        return key in self.keyform and key in self.root

    def __iter__(self) -> Iterator[Key]:
        its = (_iter(self._levels[i], self._heights[i]) for i in self._leaves)
        suites = [list(it) for it in its]
        for suite in product(*suites):
            result = self.keyform.k
            for i, s in zip(reversed(self._leaves), reversed(suite)):
                result = result.link(s, i, ())
            yield result

    def depends_on(self, ksp: KeySpaceBase) -> bool:
        if root(ksp) != self.root:
            raise ValueError("Incompatible keyspace: Non-identical roots")
        key, i = path(ksp), 0
        kf = KeyForm(key, (i,))
        while key and not kf < self.keyform.k:
            key, _ = key.cut(key.size)
            i += 1
            kf = KeyForm(key, (i,))
        return bool(key)        
