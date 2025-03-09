from typing import get_type_hints, Any, Iterator, Type, ClassVar, overload
from itertools import product
from weakref import WeakSet
import warnings

from .exc import ValidationError
from .keys import Key, KeyForm


class KSBase:
    """Base class for all keyspaces."""
    _name_: str

    def __invert__(self) -> Key:
        """Return a symbolic representation of self."""
        return Key(self._name_)

    def _iter_(self, h: int) -> Iterator[Key]:
        return
        yield


class KSParent[M: "KSChild"](KSBase):
    """Base class for parent keyspaces."""
    _members_: dict[Key, M]
    _observers_: WeakSet["KSObserver"]

    def __iter__(self) -> Iterator[str]:
        for k in self._members_:
            yield k[1][0]

    def _iter_(self, h: int) -> Iterator[Key]:
        if h <= 0:
            return
        if h == 1:
            yield from self._members_
        else:
            for key, child in self._members_.items():
                for suite in child._iter_(h - 1):
                    yield key.link(suite, 1)

    def __contains__(self, key: "str | Key | KSChild") -> bool:
        if isinstance(key, KSChild):
            return key in self._members_.values()
        k, ksp = Key(key), self
        while k and k[0][1] <= 1:
            node, k = k.cut(1)
            if not isinstance(ksp, KSParent):
                return False
            try:
                ksp = ksp._members_[node]
            except KeyError:
                return False
        else:
            while k.size:
                k, branch = k.cut(0, (0,))
                if not isinstance(ksp, KSParent) or branch not in ksp:
                    return False
        return True

    def __getitem__(self, name: str | Key) -> M:
        return self._members_[Key(name)]

    def __setitem__(self, name: str, value: M) -> None:
        if not name or any(not s.isidentifier() for s in name.split(".")):
            raise ValueError(f"Invalid keyspace name: '{name}'")
        try:
            value._parent_
        except AttributeError:
            value._name_ = name
            value._parent_ = self
            self._members_[Key(name)] = value
            for obs in self._observers_:
                obs.on_add(self, value)
        else:
            raise ValidationError(f"{value} already has a parent")

    def __delitem__(self, name: str) -> None:
        child = self._members_[(key := Key(name))]
        for obs in self._observers_:
            obs.on_del(self, child)
        del self._members_[key]


class KSChild(KSBase):
    """Base class for child keyspaces."""
    _parent_: KSParent

    def __invert__(self) -> Key:
        p = ~self._parent_
        k = super().__invert__()
        return p.link(k, p.size)


class KSObserver:
    """Base class for keyspace observers."""

    def subscribe(self, parent: "KSParent") -> None:
        """Register self as an observer of parent keyspace."""
        parent._observers_.add(self)

    def on_add(self, parent: "KSParent", child: "KSChild") -> None:
        """Called when a child keyspace is added to parent."""
        pass

    def on_del(self, parent: "KSParent", child: "KSChild") -> None:
        """Called when child keyspace is deleted from parent."""
        pass


class KSRoot[M: KSChild](KSParent[M]):
    "A generic keyspace root"
    def __init__(self):
        self._name_ = ""
        self._members_ = {}
        self._observers_ = WeakSet()


class KSNode[M: "KSChild"](KSChild, KSParent[M]):
    "A generic keyspace node"
    def __init__(self, name: str = ""):
        self._name_ = name
        self._members_ = {}
        self._observers_ = WeakSet()


def root(keyspace: KSBase) -> KSBase | None:
    ret = keyspace
    while isinstance(ret, KSChild):
        try:
            ret = ret._parent_
        except AttributeError:
            return None
    return ret


def path(keyspace: KSBase) -> Key:
    return ~keyspace


def parent(keyspace: KSChild) -> KSParent:
    return keyspace._parent_


def crawl(keyspace: KSParent, path: str | Key) -> KSBase:
    key = Key(path)
    if any(1 < deg for _, deg in key):
        raise ValueError(f"Key {key} is not a path")
    ksp = keyspace
    for label, _ in key[1:]:
        if not isinstance(keyspace, KSParent):
            raise ValidationError(f"{key} not a member of {keyspace}")
        ksp = ksp[label]
    return ksp 


def bind(__1: Key, __2: Key, *__others: Key) -> Key:
    ret = __1.link(__2, 0)
    for __i in __others:
        ret = ret.link(__i, 0)
    return ret
