from typing import Iterator, Self, Protocol, ClassVar, overload, cast
from weakref import WeakSet
from itertools import product

from .exc import ValidationError
from .keys import Key, KeyForm


class KSProtocol(Protocol):
    """Protocol for keyspace classes."""
    
    def __contains__(self, key: str | Key) -> bool:
        """True iff key is a member of self"""
        ...

    def __invert__(self) -> Key:
        """Return a Key instance representing self"""
        ...

    def __mul__(self, other) -> "KSProduct":
        """Return the keyspace obtained by crossing self with other"""
        ...

    def _keyform_(self, *hs: int) -> KeyForm:
        """Construct a keyform from self."""
        ...

    def _iter_(self, *hs: int) -> Iterator[Key]:
        """Iterate keys in self."""
        ...


class KSProduct[*Ts](KSProtocol):
    """A product of elementary keyspaces."""
    paths: tuple[*Ts]

    def __init__(self, *paths: *Ts) -> None:
        # This check is necessary b/c TypeVarTuple does not support bounds as
        # of the time of writing
        for ks in paths:
            if not isinstance(ks, KSPath):
                raise TypeError(f"Arguments to {type(self).__name__} must be "
                    f"of type {KSPath}")
        self.paths = paths

    @property
    def _paths(self) -> tuple["KSPath", ...]:
        # This property works around lack of bound support on TypeVarTuples
        return cast(tuple[KSPath, ...], self.paths)

    def __contains__(self, key: str | Key) -> bool:
        branches = Key(key).split()
        if len(branches) != len(self.paths):
            return False
        return all(k in ks for k, ks in zip(branches, self._paths))

    def __invert__(self) -> Key:
        k = ~self._paths[0]
        for p in self._paths[1:]:
            k = k * ~p
        return k

    @overload
    def __mul__[T: KSPath](self, other: T) -> "KSProduct[*Ts, T]":
        ...
    
    @overload
    def __mul__[*Us](self, other: "KSProduct[*Us]") \
        -> "KSProduct[*Ts, *Us]":
        ...

    def __mul__(self, other):
        if isinstance(other, KSPath):
            return KSProduct(*self.paths, other)
        if isinstance(other, KSProduct):
            return KSProduct(*self.paths, *other.paths)
        return NotImplemented

    def _keyform_(self, *hs: int) -> KeyForm:
        match hs:
            case ():
                kf = self._paths[0]._keyform_()
                for path in self._paths[1:]:
                    kf = kf * path._keyform_()
                return kf
            case hs if len(hs) == len(self._paths):
                kf = self._paths[0]._keyform_(hs[0])
                for h, path in zip(hs[1:], self._paths[1:], strict=True):
                    kf = kf * path._keyform_(h)
                return kf
            case _:
                raise ValueError(f"Expected exactly {len(self.paths)} heights, "
                    f"got {len(hs)}")

    def _iter_(self, *hs: int) -> Iterator[Key]:
        iterables = [KeyGroup(ks, h) if isinstance(ks, KSParent) else ()
            for ks, h in zip(self.paths, hs, strict=True)]
        for keys in product(*iterables):
            k = keys[0]
            for k_i in keys[1:]:
                k = k * k_i
            yield k        
        return


class KSPath(KSProtocol):
    """Abstract base class for all elementary keyspaces."""

    _name_: str
    _h_offset_: ClassVar[int] = 0

    def __contains__(self, key: str | Key) -> bool:
        return Key(key) == ~self

    def __invert__(self) -> Key:
        return Key(self._name_)

    @overload
    def __mul__[T: KSPath](self: Self, other: T) -> KSProduct[Self, T]:
        ...
    
    @overload
    def __mul__[*Ts](self: Self, other: KSProduct[*Ts]) -> KSProduct[Self, *Ts]:
        ...

    def __mul__(self, other):
        if isinstance(other, KSPath):
            return KSProduct(self, other)
        if isinstance(other, KSProduct):
            return KSProduct(self, *other.paths)
        return NotImplemented
    
    def _keyform_(self, *hs: int) -> KeyForm:
        match hs:
            case ():
                return KeyForm(~self, (self._h_offset_,))
            case (h,):
                h_effective = h + self._h_offset_
                if 0 <= h_effective:
                    return KeyForm(~self, (h_effective,))
                if (i := (key := ~self).size + h_effective) < 0:
                    raise ValueError(f"Cannot prune {h + self._h_offset_} "
                        f"nodes off {key}")
                key, _ = key.cut(i)
                return KeyForm(key, (h + self._h_offset_,))
            case _:
                raise ValueError(f"Unexpected arguments {hs}")

    def _iter_(self, *hs: int) -> Iterator[Key]:
        return
        yield


class KSParent[M: "KSChild"](KSPath):
    """
    Base class for parent keyspaces.
    
    Do not instantiate this class directly.
    """
    _h_offset_ = 1
    _members_: dict[Key, M]
    _observers_: WeakSet["KSObserver"]
    _namer_: Iterator[str]

    def __iter__(self) -> Iterator[str]:
        for k in self._members_:
            yield k[1][0]

    def _iter_(self, *hs: int) -> Iterator[Key]:
        h, = hs
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

    def __setitem__(self, name: str | None, value: M) -> None:
        if name is None:
            try:
                name = next(self._namer_)
            except AttributeError as e:
                raise ValueError("Automatic naming not enabled.") from e
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


class KeyGroup:
    """
    A grouping of elementary keys within a keyspace. 
    
    The group is identified by the host keyspace and an integer designating the 
    height of the keyspace relative to the units of analysis.
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
            yield Key()


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


def ks_root(ks: KSPath) -> KSRoot | None:
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


def ks_crawl(ks: KSParent, path: str | Key) -> KSPath:
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


@overload
def keyform(__ks: KSPath | KSProduct) -> KeyForm:
    ...

@overload
def keyform(__ks: KSPath, __h: int) -> KeyForm:
    ...

@overload
def keyform(__ks: KSProtocol, *hs: int) -> KeyForm:
    ...

def keyform(__ks: KSProtocol, *hs: int) -> KeyForm:
    """Construct a keyform from a keyspace."""
    return __ks._keyform_(*hs)
