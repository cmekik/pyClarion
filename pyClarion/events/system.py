from typing import Callable, Protocol, Hashable, cast
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import timedelta
from inspect import ismethod
from itertools import count
from enum import IntEnum
import logging
import heapq

from ..numdicts import Key, KeyForm, Index, ks_root
from ..numdicts.keyspaces import KSRoot, KSPath


PROCESS: ContextVar["Process"] = ContextVar("PROCESS")


class Update[I: Hashable](Protocol):
    """A future update to the simulation state."""
    def apply(self) -> None:
        ...
    @property
    def target(self) -> I:
        ...


@dataclass(slots=True)
class Event:
    """
    A simulation event.
    
    Events are ordered first by time, then by priority, then by number.

    Do not mutate the updates list as this will corrupt the event.
    """
    source: Callable
    updates: list[Update]
    time: timedelta = timedelta()
    priority: int = 0
    number: int = 0
    _index: dict[type[Update], dict[Hashable, list[Update]]] = field(init=False)

    def __post_init__(self) -> None:
        self._index = {}
        for ud in self.updates:
            (self._index
                .setdefault(type(ud), dict())
                .setdefault(ud.target, [])
                .append(ud))

    def __repr__(self) -> str:
        if ismethod(self.source) and isinstance(self.source.__self__, Process):
            source = f"{self.source.__self__.name}.{self.source.__name__}"
        else:
            source = self.source.__qualname__        
        return (f"<{self.__class__.__qualname__} "
            f"source={source} time={repr(self.time)} "
            f"at {hex(id(self))}>")
    
    def append(self, *updates) -> None:
        self.updates.extend(updates)
        for ud in updates:
            (self._index
                .setdefault(type(ud), dict())
                .setdefault(ud.target, [])
                .append(ud))
    
    def describe(self) -> str:
        if ismethod(self.source) and isinstance(self.source.__self__, Process):
            source = f"{self.source.__self__.name}.{self.source.__name__}"
        else:
            source = self.source.__qualname__
        days = self.time.days
        hours = (self.time.seconds // 86400) % 24
        minutes = (self.time.seconds // 3600) % 60 
        seconds = self.time.seconds % 60
        centiseconds = int(self.time.microseconds / 10000)
        time = (f"{days:#06x} {hours:02d}:{minutes:02d}:"
            f"{seconds:02d}.{centiseconds:02d}")
        return f"event {time} {self.priority:03d} {self.number} {source}"
    
    def index[U: Update](self, update_type: type[U]) \
        -> dict[Hashable, list[U]]:
        return cast(dict[Hashable, list[U]], self._index.get(update_type, {}))

    def __lt__(self, other) -> bool:
        if isinstance(other, Event):
            if self.time == other.time:
                if self.priority == other.priority:
                    return self.number < other.number
                return self.priority > other.priority
            return self.time < other.time
        return NotImplemented


@dataclass(slots=True)
class Clock:
    """
    A simulation clock.
    
    Tracks simulation time using datetime.timedelta() objects.
    """
    time: timedelta = timedelta()
    limit: timedelta = timedelta()
    counter: count = field(default_factory=count)

    @property
    def has_time(self) -> bool:
        """
        Return True iff clock time limit has not yet been reached.
        
        If self.limit == timedelta(), always returns True.
        """
        return not self.limit or self.time <= self.limit

    def advance(self, timepoint: timedelta) -> None:
        """
        Advance clock to given timepoint.
        
        Raises StopIteration if a time limit is set and timepoint is beyond it 
        and ValueError if timepoint precedes clock time.
        """
        if timedelta() < self.limit and self.limit < timepoint:
            raise StopIteration("Timepoint beyond time limit")
        if timepoint < self.time:
            raise ValueError("Timepoint precedes current time")
        self.time = timepoint


class Process[R: KSRoot]:
    """
    A simulated process.
    
    Owns data sites and schedules simulation events. Maintains a handle to 
    global simulation state.

    Process instances may be used in with statements for compositional model 
    construction. 

    >>> with Process("p1") as p1:
    ...     p2 = Process("p2")
    ...     assert p1.system is p2.system
    
    """

    @dataclass(slots=True)
    class System[R_: KSRoot]:
        """
        A simulated system.

        Maintains global simulation data.
        """

        root: R_
        clock: Clock = field(default_factory=Clock)
        queue: list[Event] = field(default_factory=list)
        procs: list["Process"] = field(default_factory=list)
        logger: logging.Logger = logging.getLogger(__name__)

        def check_root(self, *keyspaces: KSPath) -> None:
            for keyspace in keyspaces:
                if self.root == ks_root(keyspace):
                    continue
                raise ValueError(f"Root of {keyspace} does not match system")

        def get_index(self, form: KeyForm | Key | str) -> Index:
            return Index(self.root, form)

        def schedule(self, event: Event) -> None:
            if event.time < timedelta():
                raise ValueError("Cannot schedule an event in the past.")
            event.time += self.clock.time
            event.number = next(self.clock.counter)
            heapq.heappush(self.queue, event)

        def advance(self) -> Event:
            """Process the next event in the queue."""
            event = heapq.heappop(self.queue)
            for update in event.updates:
                try:
                    update.apply()
                except Exception as e:
                    raise RuntimeError(
                        f"Update scheduled by {event.source.__qualname__} at "
                        f"{event.time} failed") from e
            self.clock.advance(event.time)
            if self.logger.isEnabledFor(logging.INFO):
                msg = event.describe()
                self.logger.info(msg)
            for proc in self.procs:
                proc.resolve(event)
            return event
        
        def run_all(self) -> None:
            """
            Process all events.
            
            Will stop when scheduled events are exhausted or a time limit is 
            reached, whichever comes first. For information on time limits, see 
            Clock.
            """
            while self.queue and self.clock.has_time:
                self.advance()

    name: str
    system: System
    __tokens: list[Token]

    def __init__(self, name: str, root: R | None = None) -> None:
        if not all(s.isidentifier() for s in name.split(".")):
            raise ValueError("Process name must be a valid Python identifier")
        try:
            sup = PROCESS.get()
        except LookupError:
            if root is None:
                raise ValueError("Root must not be None.")
            self.name = name
            self.system = self.System(root)
        else:
            if any(proc.name == name for proc in sup.system.procs):
                raise ValueError(f"Duplicate process name '{name}'")
            self.name = name
            self.system = sup.system
        self.__tokens = []
        self.system.procs.append(self)

    def __repr__(self) -> str:
        return f"<{type(self).__qualname__} '{self.name}' at {hex(id(self))}>"        

    def __enter__(self):
        self.__tokens.append(PROCESS.set(self))
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        PROCESS.reset(self.__tokens.pop())

    def resolve(self, event: Event) -> None:
        """
        Analyze an event and schedule new events as needed.
        
        Typically, dispatches calls to various event scheduling methods.
        """
        pass

    def breakpoint(self, dt: timedelta, priority: int = 0) \
        -> Event:
        """Schedule a dummy event at specified time."""
        return Event(self.breakpoint, [], dt, priority, 0)
