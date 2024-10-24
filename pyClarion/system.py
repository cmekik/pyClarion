from typing import Callable, Sequence, Mapping, LiteralString, Any
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from math import isnan
from datetime import timedelta
from inspect import ismethod
import warnings
import heapq

from .knowledge import Sort, Term
from .numdicts import NumDict, Key, KeySpace


PROCESS: ContextVar["Process"] = ContextVar("PROCESS")


class Update:
    def apply(self) -> None:
        ...
    def affects(self, site: NumDict) -> bool:
        ...


@dataclass(slots=True)
class UpdateSite(Update):
    site: NumDict
    data: dict[Key, float] 
    reset: bool = True

    def __bool__(self) -> bool:
        return bool(self.reset or self.data)

    def apply(self) -> None:
        with self.site.mutable() as d:
            if self.reset: d.reset()
            d.update(self.data)

    def affects(self, site: NumDict) -> bool:
        return self.site is site


@dataclass(slots=True)
class UpdateSort[C: Term](Update):
    sort: Sort[C]
    add: tuple[tuple[LiteralString, C], ...] = ()
    remove: tuple[LiteralString, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.add or self.remove)

    def apply(self) -> None:
        for name, value in self.add:
            self.sort[name] = value
        for name in self.remove:
            self.sort[name]

    def affects(self, site: NumDict) -> bool:
        return site.i.depends_on(self.sort)
    

@dataclass(slots=True)
class UpdateTerm(Update):
    term: Term
    add: tuple[tuple[LiteralString, Sort], ...] = ()
    remove: tuple[LiteralString, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.add or self.remove)

    def apply(self) -> None:
        for name, value in self.add:
            self.term[name] = value
        for name in self.remove:
            self.term[name]

    def affects(self, site: NumDict) -> bool:
        return site.i.depends_on(self.term)
    

@dataclass(slots=True)
class Event:
    time: timedelta
    source: Callable
    updates: Sequence[Update]

    def __repr__(self) -> str:

        if ismethod(self.source) and isinstance(self.source.__self__, Process):
            source = f"{self.source.__self__.name}.{self.source.__name__}"
        else:
            source = self.source.__qualname__        
        return (f"<{self.__class__.__qualname__} "
            f"source={source} time={repr(self.time)} "
            f"at {hex(id(self))}>")

    def __lt__(self, other) -> bool:
        if isinstance(other, Event):
            return self.time < other.time
        return NotImplemented
        
    def affects(self, site: NumDict):
        return any(ud.affects(site) for ud in self.updates)


@dataclass(slots=True)
class Clock:
    time: timedelta = timedelta()
    limit: timedelta = timedelta()

    def advance(self, timepoint: timedelta) -> None:
        if timedelta() < self.limit and self.limit < timepoint:
            raise StopIteration("Timepoint beyond time limit")
        if timepoint < self.time:
            raise ValueError("Timepoint precedes current time")
        self.time = timepoint
    
    def event(self, dt: timedelta, src: Callable, *uds: Update) -> Event:
        if dt < timedelta():
            raise ValueError("Cannot schedule an event in the past.")
        return Event(self.time + dt, src, uds)


class Process:

    @dataclass(slots=True)
    class System:
        root: KeySpace = field(default_factory=KeySpace)
        clock: Clock = field(default_factory=Clock)
        queue: list[Event] = field(default_factory=list)
        procs: list["Process"] = field(default_factory=list)

        @property
        def index(self) -> KeySpace:
            warnings.warn(
                "System.index is deprecated; use system.root instead", 
                DeprecationWarning)
            return self.root

        def user_update(
                self, *updates: Update, dt: timedelta = timedelta()
            ) -> None:
            self.schedule(self.user_update, *updates, dt=dt)

        def schedule(
            self, src: Callable, *uds: Update, dt: timedelta = timedelta()
        ) -> None:
            heapq.heappush(self.queue, self.clock.event(dt, src, *uds))

        def advance(self) -> Event:
            event = heapq.heappop(self.queue)
            for update in event.updates:
                try:
                    update.apply()
                except Exception as e:
                    raise RuntimeError(
                        f"Update scheduled by {event.source.__qualname__} at "
                        f"{event.time} failed") from e
            self.clock.advance(event.time)
            for proc in self.procs:
                proc.resolve(event)
            return event

    name: str
    system: System
    __tokens: list[Token]

    def __init__(self, name: str) -> None:
        if not name.isidentifier():
            raise ValueError("Process name must be a valid Python identifier")
        try:
            sup = PROCESS.get()
        except LookupError:
            self.name = name
            self.system = self.System()
        else:
            if any(proc.name == name for proc in sup.system.procs):
                raise ValueError(f"Duplicate process name '{name}'")
            self.name = name
            self.system = sup.system
        self.__tokens = []
        self.system.procs.append(self)

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            old = getattr(self, name)
        except AttributeError:
            pass
        else:
            if not isinstance(old, NumDict):
                pass
            if not isinstance(value, NumDict):
                raise TypeError()
            if old.i != value.i:
                raise ValueError()
            if not (isnan(old.c) and isnan(value.c) or old.c == value.c):
                raise ValueError()
        super().__setattr__(name, value)

    def __enter__(self):
        self.__tokens.append(PROCESS.set(self))
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        PROCESS.reset(self.__tokens.pop())

    def resolve(self, event: Event) -> None:
        pass

    def breakpoint(self, dt: timedelta) -> None:
        self.system.schedule(self.breakpoint, dt=dt)
