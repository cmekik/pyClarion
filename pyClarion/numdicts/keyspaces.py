import abc
from typing import Iterator
from weakref import WeakSet
from itertools import product

from .exc import ValidationError
from .keys import Key


class KSBase(abc.ABC):
    """Abstract base class for all keyspaces."""
    
    @abc.abstractmethod
    def __contains__(self, key: str | Key) -> bool:
        pass

    @abc.abstractmethod
    def __invert__(self) -> Key:
        pass

    @abc.abstractmethod
    def __mul__(self, other) -> "KSProduct":
        pass


class KSPath(KSBase):
    """Abstract base class for all elementary keyspaces."""

    _name_: str

    def __invert__(self) -> Key:
        """Return a symbolic representation of self."""
        return Key(self._name_)

    def __mul__(self, other) -> "KSProduct":
        if isinstance(other, KSPath):
            return KSProduct(self, other)
        if isinstance(other, KSProduct):
            return KSProduct(self, *other.paths)
        return NotImplemented

    def _iter_(self, h: int) -> Iterator[Key]:
        return
        yield


class KSParent[M: "KSChild"](KSPath):
    """
    Base class for parent keyspaces.
    
    Do not instantiate this class directly.
    """
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

    def __contains__(self, key: str | Key) -> bool:
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


class KSChild(KSPath):
    """
    Base class for child keyspaces.
    
    Do not instantiate this class directly.
    """
    _parent_: KSParent

    def __contains__(self, key: str | Key) -> bool:
        return Key(key) == ~self

    def __invert__(self) -> Key:
        try:
            p = ~self._parent_
        except AttributeError:
            return super().__invert__()
        k = super().__invert__()
        return p.link(k, p.size)


class KSRoot[M: KSChild](KSParent[M]):
    """
    A generic keyspace root.
    
    Do not instantiate this class directly.
    """
    def __init__(self):
        self._name_ = ""
        self._members_ = {}
        self._observers_ = WeakSet()


class KSNode[M: "KSChild"](KSChild, KSParent[M]):
    """
    A generic keyspace node.
    
    Do not instantiate this class directly.
    """
    def __init__(self, name: str = ""):
        self._name_ = name
        self._members_ = {}
        self._observers_ = WeakSet()


class KSProduct(KSBase):
    """A product of elementary keyspaces."""
    paths: tuple[KSPath, ...]

    def __init__(self, *paths: KSPath) -> None:
        self.paths = paths

    def __contains__(self, key: str | Key) -> bool:
        branches = Key(key).split()
        if len(branches) != len(self.paths):
            return False
        return all(k in ks for k, ks in zip(branches, self.paths))

    def __invert__(self) -> Key:
        k = ~self.paths[0]
        for p in self.paths[1:]:
            k = k * ~p
        return k
    
    def __mul__(self, other) -> "KSProduct":
        if isinstance(other, KSPath):
            return KSProduct(*self.paths, other)
        if isinstance(other, KSProduct):
            return KSProduct(*self.paths, *other.paths)
        return NotImplemented

    def _iter_(self, *hs: int) -> Iterator[Key]:
        iterables = [KeyGroup(ks, h) if isinstance(ks, KSParent) else ()
            for ks, h in zip(self.paths, hs, strict=True)]
        for keys in product(*iterables):
            k = keys[0]
            for k_i in keys[1:]:
                k = k * k_i
            yield k        
        return


class KeyGroup:
    """
    A grouping of elementary keys within a keyspace. 
    
    The group is identified by the host keyspace and an integer 
    designating the height of the keyspace relative to the units.
    """
    __slots__ = ("ks", "h")
    ks: KSPath
    h: int

    def __init__(self, ks: KSPath, h: int) -> None:
        self.ks = ks
        self.h = h

    def __contains__(self, obj: str | Key) -> bool:
        k = Key(obj)
        return k in self.ks and (~self.ks).size + self.h == k.size

    def __len__(self) -> int:
        if self.h <= 0:
            return 1
        elif isinstance(self.ks, KSParent):
            return sum([len(KeyGroup(ks, self.h - 1)) 
                for ks in self.ks._members_.values()])
        else:
            return 0

    def __iter__(self) -> Iterator[Key]:
        if 0 < self.h and isinstance(self.ks, KSParent):
            yield from self.ks._iter_(self.h)
        elif self.h == 0:
            yield ~self.ks


class KSObserver:
    """
    Base class for keyspace observers.
    
    Do not instantiate this class directly.
    """

    def subscribe(self, parent: "KSParent") -> None:
        """Register self as an observer of parent keyspace."""
        parent._observers_.add(self)

    def on_add(self, parent: "KSParent", child: "KSChild") -> None:
        """Called when a child keyspace is added to parent."""
        pass

    def on_del(self, parent: "KSParent", child: "KSChild") -> None:
        """Called when child keyspace is deleted from parent."""
        pass


def ks_root(ks: KSBase) -> KSRoot | None:
    """Return the root of keyspace ks if it exists, otherwise return None."""
    ret = ks
    while isinstance(ret, KSChild):
        try:
            ret = ret._parent_
        except AttributeError:
            return None
    if isinstance(ret, KSRoot):
        return ret
    return None


def ks_parent(ks: KSChild) -> KSParent:
    """Return parent of child keyspace ks."""
    return ks._parent_


def ks_crawl(ks: KSParent, path: str | Key) -> KSBase:
    """Find and return the keyspace located at path relative to ks."""
    key = Key(path)
    if any(1 < deg for _, deg in key):
        raise ValueError(f"Key {key} is not a path")
    ksp = ks
    for label, _ in key[1:]:
        if not isinstance(ks, KSParent):
            raise ValidationError(f"{key} not a member of {ks}")
        ksp = ksp[label]
    return ksp 
