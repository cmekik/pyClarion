from typing import Iterator, Self, ClassVar, Any, overload, cast, get_type_hints, no_type_check
from abc import ABC, abstractmethod
from weakref import WeakSet
from itertools import product

from .exc import ValidationError
from .keys import Key, KeyForm


class KeySpace(ABC):
    """Abstract base class for keyspaces."""

    @abstractmethod
    def __contains__(self, key: str | Key) -> bool:
        """True iff key is a member of self"""
        raise NotImplementedError()

    @abstractmethod
    def __invert__(self) -> Key:
        """Return a Key instance representing self"""
        raise NotImplementedError()

    @abstractmethod
    def _keyform_(self, *hs: int) -> KeyForm:
        """Construct a keyform from self."""
        raise NotImplementedError()

    @abstractmethod
    def _iter_(self, *hs: int) -> Iterator[Key]:
        """Iterate keys in self."""
        raise NotImplementedError()


class Unit(KeySpace):
    """Abstract base class for elementary keyspaces."""
    pass


class Product[T: tuple[Unit, ...]](KeySpace):
    """A product of elementary keyspaces."""
    paths: T

    def __init__(self, paths: T) -> None:
        if not paths:
            raise ValueError("Cannot construct empty product.")
        self.paths = paths

    def __contains__(self, key: str | Key) -> bool:
        branches = Key(key).split()
        if len(branches) != len(self.paths):
            return False
        return all(k in ks for k, ks in zip(branches, self.paths, strict=True))

    def __invert__(self) -> Key:
        paths = self.paths 
        k = ~paths[0]
        for p in paths[1:]:
            k = k * ~p
        return k

    def _keyform_(self, *hs: int) -> KeyForm:
        paths = self.paths
        match hs:
            case ():
                kf = paths[0]._keyform_()
                for path in paths[1:]:
                    kf = kf * path._keyform_()
                return kf
            case hs if len(hs) == len(paths):
                kf = paths[0]._keyform_(hs[0])
                for h, path in zip(hs[1:], paths[1:], strict=True):
                    kf = kf * path._keyform_(h)
                return kf
            case _:
                raise ValueError(f"Expected exactly {len(self.paths)} heights, "
                    f"got {len(hs)}")

    def _iter_(self, *hs: int) -> Iterator[Key]:
        iterables = [KeyGroup(ks, h) if isinstance(ks, Parent) else ()
            for ks, h in zip(self.paths, hs, strict=True)]
        for keys in product(*iterables):
            k = keys[0]
            for k_i in keys[1:]:
                k = k * k_i
            yield k        
        return


class Path(Unit):
    """Abstract base class for keyspace paths."""
    _h_offset_: ClassVar[int] = 0
    
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


class Parent[M: Child](Path):
    """Abstract base class for parent keyspaces."""
    _h_offset_ = 1
    _init_required_: ClassVar[bool] = True
    _required_: "ClassVar[dict[Key, type[Child]]]"
    _members_: dict[Key, M]
    _observers_: WeakSet["KSObserver"]
    _namer_: Iterator[str]

    @classmethod
    @abstractmethod
    def _get_member_type_(cls) -> type[M]:
        raise NotImplementedError()

    @classmethod
    def __init_subclass__(cls: type[Self]) -> None:
        try:
            member_type = cls._get_member_type_()
        except NotImplementedError:
            pass
        else:
            required = dict()
            for name, annotation_type in get_type_hints(cls).items():
                if not isinstance(annotation_type, type):
                    continue
                elif name.startswith("_") and name.endswith("_"):
                    continue
                elif issubclass(annotation_type, member_type):
                    required[Key(name)] = annotation_type
                    descriptor = ChildSlot()
                    setattr(cls, name, descriptor)
                    descriptor.__set_name__(cls, name)
                else:
                    raise TypeError(
                        f"Attribute '{name}' has incompatible type "
                        f"{annotation_type}")
            cls._required_ = required

    def __init__(self, skip: "tuple[type[Child], ...]" = ()) -> None:
        super().__init__()
        self._members_ = {}
        self._observers_ = WeakSet()
        if self._init_required_:
            self._initialize_required_(skip)

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
            if not isinstance(ksp, Parent):
                return False
            try:
                ksp = ksp._members_[node]
            except KeyError:
                return False
        else:
            while k.size:
                k, branch = k.cut(0, (0,))
                if not isinstance(ksp, Parent) or branch not in ksp:
                    return False
        return True

    def __getitem__(self, name: str | Key) -> M:
        return self._members_[Key(name)]

    def __setitem__(self, name: str, value: M) -> None:
        if not name:
            try:
                name = next(self._namer_)
            except AttributeError as e:
                raise ValueError("Automatic naming not enabled.") from e
        elif any(not s.isidentifier() for s in name.split(".")):
            raise ValueError(f"Invalid keyspace name: '{name}'")
        elif Key(name) in self._members_:
            raise ValueError(f"Member '{name}' already exists")
        member_type = self._get_member_type_()
        if not isinstance(value, member_type):
            raise TypeError(f"Incompatible member type {type(value)}")
        elif hasattr(value, "_parent_"):
            raise ValueError("Value is already bound to a parent")
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
        if (key := Key(name)) in self._required_:
            raise ValueError(f"Child '{name}' is a required member of {~self}")
        child = self._members_[(key := Key(name))]
        for obs in self._observers_:
            obs.on_del(self, child)
        del self._members_[key]
        del child._parent_

    def __setattr__(self, name, value) -> None:
        if Key(name) in type(self)._required_:
            self[name] = value
        else:
            super().__setattr__(name, value)

    def _initialize_required_(self, skip: "tuple[type[Child],...]" = ()) -> None:
        for key, required_type in self._required_.items():
            if issubclass(required_type, skip):
                continue
            if key not in self._members_:
                name, _ = key[-1]
                self[name] = cast(M, required_type())

    def _check_required_members_(self) -> None:
        missing = [key for key in self._required_ if key not in self._members_]
        if missing:
            missing_members = ", ".join(key[-1][0] for key in missing)
            raise ValidationError(f"Uninitialized required members in {~self}: "
                f"{missing_members}")
        for v in self._members_.values():
            if isinstance(v, Parent):
                v._check_required_members_()


class Root[M: Child](Parent[M]):
    """Base class for keyspace roots."""
    def __invert__(self) -> Key:
        return Key()


class Child(Path):
    """Base class for child keyspaces."""
    _parent_: Parent
    _name_: str

    def __init__(self, name: str = "") -> None:
        self._name_ = name

    def __contains__(self, key: str | Key) -> bool:
        return Key(key) == ~self

    def __invert__(self) -> Key:
        p = ~self._parent_
        return p.link(Key(self._name_), p.size)

    def __mul__[T: Path](self, other: T) -> "Dyad[Self, T]":
        """Return the keyspace obtained by crossing self with other"""
        if isinstance(other, Path):
            return Dyad((self, other))
        return NotImplemented


class ChildSlot[T: Child]:
    def __set_name__(self, owner: Any, name: str) -> None:
        if issubclass(owner, Parent):
            self._name_ = name

    def __get__(self, obj: Any, objtype: Any | None) -> T:
        if not isinstance(obj, Parent):
            raise TypeError()
        return obj._members_[Key(self._name_)]

    def __set__(self, obj: Any, value: T) -> None:
        if isinstance(obj, Parent):
            raise ValidationError(f"Cannot mutate required path")


class Node[M: Child](Child, Parent[M]):
    """Base class for internal keyspace nodes."""
    def __init__(self, name: str = "", skip: "tuple[type[Child],...]" = ()):
        super(Child, self).__init__(skip)
        self._name_ = name


class Dyad[L: Unit, R: Unit](Product):
    """A dyad of elementary keyspaces."""
    def __mul__[T: Unit](self, other: T) -> "Triad[L, R, T]":
        """Return the keyspace obtained by crossing self with other"""
        if isinstance(other, Unit):
            return Triad((*self.paths, other))
        return NotImplemented


class Triad[L: Unit, C: Unit, R: Unit](Product):
    """A triad of elementary keyspaces."""
    pass


class KeyGroup:
    """
    A grouping of elementary keys within a keyspace. 
    
    The group is identified by the host keyspace and an integer designating the 
    height of the keyspace relative to the units of analysis.
    """
    __slots__ = ("ks", "h")
    ks: Unit
    h: int

    def __init__(self, ks: Unit, h: int) -> None:
        self.ks = ks
        self.h = h

    def __contains__(self, key: Key) -> bool:
        return key in self.ks and (~self.ks).size + self.h == key.size

    def __len__(self) -> int:
        if self.h <= 0:
            return 1
        elif isinstance(self.ks, Parent):
            return sum([len(KeyGroup(ks, self.h - 1)) 
                for ks in self.ks._members_.values()])
        else:
            return 0

    def __iter__(self) -> Iterator[Key]:
        if 0 < self.h and isinstance(self.ks, Parent):
            yield from self.ks._iter_(self.h)
        elif self.h == 0:
            yield Key()


class KSObserver:
    """
    Base class for keyspace observers.
    
    Do not instantiate this class directly.
    """

    def subscribe(self, parent: Parent) -> None:
        """Register self as an observer of parent keyspace."""
        parent._observers_.add(self)

    def on_add(self, parent: Parent, child: Child) -> None:
        """Called when a child keyspace is added to parent."""
        pass

    def on_del(self, parent: Parent, child: Child) -> None:
        """Called when child keyspace is deleted from parent."""
        pass


def find_root(ks: Path) -> Root | None:
    """Return the root of keyspace ks if it exists, otherwise return None."""
    ret = ks
    while isinstance(ret, Child):
        try:
            ret = ret._parent_
        except AttributeError:
            return None
    if isinstance(ret, Root):
        return ret
    return None


def find_parent(ks: Child) -> Parent:
    """Return parent of child keyspace ks."""
    return ks._parent_


def crawl(ks: Parent, star: Key) -> Path | Dyad | Triad | Product:
    """Find and return the keyspace located at path relative to ks."""
    if any(1 < deg for _, deg in star[1:]):
        raise ValueError(f"Key {star} is not a star")
    if star[0][1] == 1:
        ksp = ks
        for key in star[1:]:
            label, deg = key
            if 1 < deg:
                raise ValueError(f"{key} is not star-shaped")
            if not isinstance(ksp, Parent):
                raise ValidationError(f"{key} not a member of {ksp}")
            ksp = ksp[label]
        return ksp  
    elif star[0][1] == 2:
        l, r = star.split()
        ks_l, ks_r = crawl(ks, l), crawl(ks, r)
        assert isinstance(ks_l, Unit) and isinstance(ks_r, Unit)
        return Dyad((ks_l, ks_r))
    elif star[0][1] == 3:
        l, c, r = star.split()
        ks_l, ks_c, ks_r = crawl(ks, l), crawl(ks, c), crawl(ks, r)
        assert isinstance(ks_l, Unit) 
        assert isinstance(ks_c, Unit) 
        assert isinstance(ks_r, Unit)
        return Triad((ks_l, ks_c, ks_r))
    else:
        split = star.split()
        units = tuple(crawl(ks, branch) for branch in split)
        assert all(isinstance(unit, Unit) for unit in units)
        return Product(cast(tuple[Unit, ...], units))


@overload
def keyform(__ks: Path | Product) -> KeyForm:
    ...

@overload
def keyform(__ks: Path, __h: int) -> KeyForm:
    ...

@overload
def keyform(__ks: KeySpace, *hs: int) -> KeyForm:
    ...

def keyform(__ks: KeySpace, *hs: int) -> KeyForm:
    """Construct a keyform from a keyspace."""
    return __ks._keyform_(*hs)
