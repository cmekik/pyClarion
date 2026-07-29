from typing import Protocol, Hashable, ClassVar
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import count
from enum import Enum
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


class Action:
    __slots__ = ("source", "updates", "duration", "priority")

    source: "Process"
    updates: list[Update]
    duration: timedelta
    priority: int

    def __init__(self, source: "Process") -> None:
        self.source = source
        self.updates = []
        self.duration = timedelta()
        self.priority = 0


@dataclass(slots=True)
class Event:
    """
    A simulation event.
    
    Events are ordered first by time, then by priority, then by number.

    Do not directly mutate the updates list as this will corrupt the event, use 
    event.append() instead.

    Do not mutate any event attribute if the event is marked as scheduled.
    """

    action: Action
    time: tuple[timedelta, int, int]

    def __lt__(self, other) -> bool:
        return self.time < other.time
    
    def describe(self) -> str:
        action = self.action.__class__.__name__
        source = self.action.source.name
        t, p, n = self.time
        days = t.days
        hours = (t.seconds // 86400) % 24
        minutes = (t.seconds // 3600) % 60 
        seconds = t.seconds % 60
        centiseconds = int(t.microseconds / 10000)
        time = (f"{days:#06x} {hours:02d}:{minutes:02d}:"
            f"{seconds:02d}.{centiseconds:02d}")
        return f"{action} {source} {time} {p:03d} {n}"


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
        main: "Process"
        depth: int = 1
        clock: Clock = field(default_factory=Clock)
        queue: list[Event] = field(default_factory=list)
        procs: list["Process"] = field(default_factory=list)
        active: set["Process"] = field(default_factory=set)
        states: dict["Process", Enum] = field(default_factory=dict)
        logger: logging.Logger = logging.getLogger(__name__)

        def check_root(self, *keyspaces: KSPath) -> None:
            for keyspace in keyspaces:
                if self.root == ks_root(keyspace):
                    continue
                raise ValueError(f"Root of {keyspace} does not match system")

        def get_index(self, form: KeyForm | Key | str) -> Index:
            return Index(self.root, form)

        def schedule(self, action: Action) -> None:
            if action.source.is_locked:
                raise ValueError("Cannot generate event from a locked process.")
            if action.duration < timedelta():
                raise ValueError("Action has negative duration")
            t = self.clock.time + action.duration
            p = action.priority
            n = next(self.clock.counter)
            heapq.heappush(self.queue, Event(action, (t, p, n)))
            if action.source.is_blocking:
                self.active.add(action.source)

        def advance(self) -> Event:
            """Process the next event in the queue."""
            event = heapq.heappop(self.queue)
            self.clock.advance(event.time[0])
            for update in event.action.updates:
                try:
                    update.apply()
                except Exception as e:
                    raise RuntimeError(
                        f"Update scheduled by {event.action.source.name} at "
                        f"{event.time} failed") from e
            if event.action.source.is_blocking:
                self.active.remove(event.action.source)
            if self.logger.isEnabledFor(logging.INFO):
                msg = event.describe()
                self.logger.info(msg)
            self.main.resolve(event)
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

    is_blocking: ClassVar[bool] = True

    name: str
    system: System
    parent: "Process | None"
    children: "list[Process]"
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
            self.system = self.System(root, self)
            self.parent = None
            self.children = []
        else:
            if any(proc.name == name for proc in sup.system.procs):
                raise ValueError(f"Duplicate process name '{name}'")
            if sup.is_blocking and not self.is_blocking:
                raise ValueError("Cannot subordinate non-blocking process "
                    "to a blocking process.")
            self.name = name
            self.system = sup.system
            self.parent = sup
            self.parent.children.append(self)
        self.__tokens = []
        self.system.procs.append(self)

    @property
    def is_locked(self) -> bool:
        if not self.is_blocking:
            return False
        elif self.parent is None or not self.parent.is_blocking:
            return any(p in self.system.active for p in self.descendants())
        else:
            cluster = self.cluster()
            return cluster.is_locked 

    def __repr__(self) -> str:
        return f"<{type(self).__qualname__} '{self.name}' at {hex(id(self))}>"        

    def __enter__(self):
        self.__tokens.append(PROCESS.set(self))
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        PROCESS.reset(self.__tokens.pop())

    def cluster(self) -> "Process":
        current = self
        while current.parent is not None and current.parent.is_blocking:
            current = current.parent
        return current

    def descendants(self) -> "list[Process]":
        return [self, *(p for c in self.children for p in c.descendants())]
        
    def resolve(self, event: Event) -> None:
        """Analyze current event and schedule new events as needed."""
        pass
