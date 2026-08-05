__all__ = ["Update", "Fluent", "Status", "Action", "Start", "Stop", 
    "Transition", "Event", "Clock", "Process"]


from typing import Protocol, Hashable, ClassVar, Iterator, Any, final
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import count
from enum import Enum, auto
import random
import logging
import heapq
import time

from ..numdicts.keys import Key, KeyForm
from ..numdicts.indices import Index
from ..numdicts.keyspaces import Root, Child, Product, find_root


PROCESS: ContextVar["Process"] = ContextVar("PROCESS")


class Update[I: Hashable](Protocol):
    """A future update to the simulation state."""
    def apply(self) -> None:
        ...
    @property
    def target(self) -> I:
        ...


class Fluent(Enum):
    """Base class for state fluents."""
    pass


class Status(Fluent):
    """Basic free/busy status fluents."""
    FREE = auto()
    BUSY = auto()


class Action[P: Process]:
    preconditions: ClassVar[tuple[Fluent, ...]] = (Status.FREE,)

    __slots__ = ("source", "fluent", "updates", "duration", "priority")
    __match_args__ = ("source",)

    source: P
    fluent: Fluent
    updates: list[Update]
    duration: timedelta
    priority: int

    def __init__(self, source: P) -> None:
        if self.preconditions and source.status not in self.preconditions:
            raise ValueError(f"Process '{source.name}' does not satify "
                f"preconditions for action of type {type(self).__name__}")
        latency = random.expovariate(1 / source.system.latency)
        self.source = source
        self.fluent = source.status
        self.updates = []
        self.duration = timedelta(seconds=latency)
        self.priority = 0

    def describe(self) -> str:
        return ""


@final
class Start[P: Process](Action[P]):
    __slots__ = ()
    def __init__(self, source: P, fluent: Fluent | None = None) -> None:
        super().__init__(source)
        if fluent is not None:
            self.fluent = fluent


@final
class Stop[P: Process](Action[P]):
    __slots__ = ()
    def __init__(self, source: P, fluent: Fluent | None = None) -> None:
        super().__init__(source)
        if fluent is not None:
            self.fluent = fluent

@final
class Transition[P: Process](Action[P]):
    __slots__ = ()
    __match_args__ = ("source", "fluent")
    preconditions = ()
    def __init__(self, source: P, fluent: Fluent) -> None:
        super().__init__(source)
        self.fluent = fluent


@dataclass(slots=True)
class Event[A: Action]:
    """
    A simulation event.
    
    Events are ordered first by time, then by priority, then by number.

    Do not directly mutate the updates list as this will corrupt the event, use 
    event.append() instead.

    Do not mutate any event attribute if the event is marked as scheduled.
    """

    action: A
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
    time: timedelta = field(default_factory=timedelta)
    start: float = float("-inf")
    finish: float = float("-inf")
    counter: count = field(default_factory=count)

    @property
    def wall_time(self) -> timedelta:
        """
        Current simulation duration in seconds.
        
        If simulation is finished, reports total simulation duration in wall 
        time, otherwise reports running simulation duration up to current time.
        """
        bound = self.finish
        if bound < 0.0:
            bound = time.perf_counter()
        dt = bound - self.start
        return timedelta(seconds=dt)

    def punch_in(self) -> None:
        self.start = time.perf_counter()
        self.finish = float("-inf")

    def punch_out(self) -> None:
        self.finish = time.perf_counter()

    def advance(self, timepoint: timedelta) -> None:
        """
        Advance clock to given timepoint.
        
        Raises StopIteration if a time limit is set and timepoint is beyond it 
        and ValueError if timepoint precedes clock time.
        """
        if timepoint < self.time:
            raise ValueError("Timepoint precedes current time")
        self.time = timepoint


class System[R: Root[Child]]:
    """
    A simulated system.

    Maintains global simulation data.
    """
        
    def __init__(self, 
        root: R, 
        main: "Process", 
        depth: int = 1, 
        latency: float = 5e-3
    ) -> None:
        if latency <= 0:
            raise ValueError("Non-positive latency value")
        self._root = root
        self._main = main
        self._depth = depth
        self._clock = Clock()
        self._queue: list[Event] = []
        self._procs: list["Process"] = []
        self._status: dict["Process", Fluent] = {}
        self._latency = latency
        self._root._check_required_members_()

    def __iter__(self) -> Iterator["Process"]:
        yield from self._procs

    def __contains__(self, proc: "Process") -> bool:
        for p in self._procs:
            if proc is p:
                return True
        return False

    @property
    def root(self) -> R:
        return self._root

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def status(self) -> dict["Process", Fluent]:
        return self._status

    @property
    def latency(self) -> float:
        return self._latency

    def register(self, proc: "Process") -> None:
        self._procs.append(proc)
        self._status[proc] = Status.FREE

    def check_root(self, *ksps: Child | Product[tuple[Child, ...]]) -> None:
        for ks in ksps:
            match ks:
                case Child():
                    if self.root == find_root(ks):
                        continue
                    raise ValueError(f"Root of {ks} does not match system")
                case Product():
                    self.check_root(*ks.units)
                case _:
                    raise TypeError(f"Unexpected object of type {type(ks)}")

    def get_index(self, form: KeyForm | Key | str) -> Index:
        return Index(self.root, form)

    def schedule(self, action: Action) -> None:
        if action.source.is_locked:
            raise ValueError("Cannot generate event from a locked process.")
        if action.duration < timedelta():
            raise ValueError("Action has negative duration")
        t = self._clock.time + action.duration
        p = action.priority
        n = next(self._clock.counter)
        heapq.heappush(self._queue, Event(action, (t, p, n)))
        if action.source.is_blocking:
            self._status[action.source] = Status.BUSY
    
    def run(self, sync: float = float("-inf")) -> None:
        """
        Run system.

        Processes queued events in temporal order. If 0.0 < sync, events 
        are synchronized with real time at the given rate.
        """
        self.schedule(Start(self._main))
        self._clock.punch_in()
        while self._queue:
            if 0.0 < sync:
                t_next = self._queue[0].time[0] / sync
                t_wall = self._clock.wall_time
                dt = (t_next - t_wall).total_seconds()
                if 0.0 < dt:
                    time.sleep(dt)
            event = heapq.heappop(self._queue)
            self._clock.advance(event.time[0])
            for update in event.action.updates:
                try:
                    update.apply()
                except Exception as e:
                    n, t = event.action.source.name, event.time
                    raise RuntimeError(f"Failed '{repr(update)}' to "
                        f"process '{n}' at {t}") from e
            self._status[event.action.source] = event.action.fluent
            self._main.resolve(event)
            if type(event.action) is Stop: 
                if event.action.source is not self._main:
                    raise ValueError("Stop signaled from child process")
                break
        self._clock.punch_out()


class Process[R: Root[Child]]:
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
    is_blocking: ClassVar[bool] = True

    def __init__(self, name: str, root: R | None = None) -> None:
        if not all(s.isidentifier() for s in name.split(".")):
            raise ValueError("Process name must be a valid Python identifier")
        try:
            sup = PROCESS.get()
        except LookupError:
            if root is None:
                raise ValueError("Root must not be None.")
            self._name = name
            self._system: System[R] = System(root, self)
            self._parent = None
        else:
            if any(proc.name == name for proc in sup.system):
                raise ValueError(f"Duplicate process name '{name}'")
            if root is not None and root is not sup.system.root:
                raise ValueError("New root passed to subordinate process")
            if sup.is_blocking and not self.is_blocking:
                raise ValueError("Cannot subordinate non-blocking process "
                    "to a blocking process.")
            self._name = name
            self._system = sup.system
            self._parent = sup
            self._parent._children.append(self)
        self._children: list["Process"] = []
        self.__tokens: list[Token] = []
        self.system.register(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def system(self) -> System[R]:
        return self._system

    @property
    def parent(self) -> "Process | None":
        return self._parent

    @property
    def is_locked(self) -> bool:
        if not self.is_blocking:
            return False
        elif self.parent is None or not self.parent.is_blocking:
            status = self.system.status
            return any(status[p] is Status.BUSY for p in self)
        else:
            cluster = self.cluster()
            return cluster.is_locked 

    @property
    def is_active(self) -> bool:
        system = self.system
        return system.status[self] is not Status.FREE

    @property
    def status(self) -> Fluent:
        return self.system.status[self]

    def __iter__(self) -> Iterator["Process"]:
        yield from self._children
        for p in self._children:
            yield from p

    def __contains__(self, other: "Process") -> bool:
        return any(other is p for p in self)

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
        
    def resolve(self, event: Event) -> None:
        """Analyze current event and schedule new events as needed."""
        raise ValueError(f"Unexpected event {event}")
