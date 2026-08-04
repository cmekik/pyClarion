__all__ = ["Update", "Fluent", "Status", "Action", "Start", "Stop", 
    "Transition", "Event", "Clock", "Process"]


from typing import Protocol, Hashable, ClassVar, final
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import count
from enum import Enum, auto
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
        self.source = source
        self.fluent = source.status
        self.updates = []
        self.duration = timedelta()
        self.priority = 0

    def describe(self) -> str:
        return ""


@final
class Start[P: Process](Action[P]):
    __slots__ = ()
    def __init__(self, source: P) -> None:
        super().__init__(source)


@final
class Stop[P: Process](Action[P]):
    __slots__ = ()
    def __init__(self, source: P) -> None:
        super().__init__(source)


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


class Process[R: Root]:
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
    class System[R_: Root]:
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
        status: dict["Process", Fluent] = field(default_factory=dict)
        logger: logging.Logger = logging.getLogger(__name__)

        def __post_init__(self):
            self.root._check_required_members_()

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
            t = self.clock.time + action.duration
            p = action.priority
            n = next(self.clock.counter)
            heapq.heappush(self.queue, Event(action, (t, p, n)))
            if action.source.is_blocking:
                self.status[action.source] = Status.BUSY
        
        def run(self, sync: float = float("-inf")) -> None:
            """
            Run system.

            Processes queued events in temporal order. If 0.0 < sync, events 
            are synchronized with real time at the given rate.
            """
            self.schedule(Start(self.main))
            self.clock.punch_in()
            while self.queue:
                if 0.0 < sync:
                    t_next = self.queue[0].time[0] / sync
                    t_wall = self.clock.wall_time
                    dt = (t_next - t_wall).total_seconds()
                    if 0.0 < dt:
                        time.sleep(dt)
                event = heapq.heappop(self.queue)
                self.clock.advance(event.time[0])
                for update in event.action.updates:
                    try:
                        update.apply()
                    except Exception as e:
                        n, t = event.action.source.name, event.time
                        raise RuntimeError(f"Failed '{repr(update)}' to "
                            f"process '{n}' at {t}") from e
                self.status[event.action.source] = event.action.fluent
                self.main.resolve(event)
                if type(event.action) is Stop: 
                    if event.action.source is not self.main:
                        raise ValueError("Stop signaled from child process")
                    break
            self.clock.punch_out()

    is_blocking: ClassVar[bool] = True

    name: str
    system: System[R]
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
        else:
            if any(proc.name == name for proc in sup.system.procs):
                raise ValueError(f"Duplicate process name '{name}'")
            if root is not None:
                raise ValueError("Root passed to subordinate process")
            if sup.is_blocking and not self.is_blocking:
                raise ValueError("Cannot subordinate non-blocking process "
                    "to a blocking process.")
            self.name = name
            self.system = sup.system
            self.parent = sup
            self.parent.children.append(self)
        self.children = []
        self.__tokens = []
        self.system.procs.append(self)
        self.system.status[self] = Status.FREE

    @property
    def is_locked(self) -> bool:
        if not self.is_blocking:
            return False
        elif self.parent is None or not self.parent.is_blocking:
            status = self.system.status
            return any(status[p] is Status.BUSY for p in self.descendants())
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

    def __contains__(self, other: "Process") -> bool:
        if other in self.children:
            return True
        else:
            return any(other in child for child in self.children)

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
        raise ValueError(f"Unexpected event {event}")
