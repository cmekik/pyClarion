from typing import Callable, Sequence, ClassVar, Any, Iterator
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from math import isnan
from datetime import timedelta
from inspect import ismethod
from itertools import count
from enum import IntEnum
from collections import deque
from weakref import WeakSet
import logging
import heapq

from .knowledge import Root, Sort, Term
from .numdicts import NumDict, Key, KeyForm, Index, numdict, ks_root
from .numdicts.keyspaces import KSPath


PROCESS: ContextVar["Process"] = ContextVar("PROCESS")



class Update:
    """A future update to the simulation state."""
    def apply(self) -> None:
        ...


@dataclass(slots=True)
class UpdateSort[C: Term](Update):
    """A future update to a sort within a simulation keyspace."""
    sort: Sort[C]
    add: tuple[C, ...] = ()
    remove: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.add or self.remove)

    def apply(self) -> None:
        for term in self.add:
            try:
                self.sort[term._name_] = term
            except AttributeError:
                self.sort[f"_{next(self.sort._counter_)}"] = term
        for name in self.remove:
            self.sort[name]

    def affects(self, site: NumDict) -> bool:
        return site.i.depends_on(self.sort)
    

class Priority(IntEnum):
    """
    An event priority enum.
    
    Used to indicate event priority in case two events are scheduled at the 
    same time within a simulation. 
    """
    MAX = 128
    PARAM = 120
    CHOICE = 112
    LEARNING = 96
    PROPAGATION = 64
    DEFERRED = 32
    MIN = 0


@dataclass(slots=True)
class Event:
    """
    A simulation event.
    
    Events are ordered first by time, then by priority, then by number.
    """
    source: Callable
    updates: Sequence[Update]
    time: timedelta = timedelta()
    priority: int = Priority.PROPAGATION
    number: int = 0

    def __repr__(self) -> str:
        if ismethod(self.source) and isinstance(self.source.__self__, Process):
            source = f"{self.source.__self__.name}.{self.source.__name__}"
        else:
            source = self.source.__qualname__        
        return (f"<{self.__class__.__qualname__} "
            f"source={source} time={repr(self.time)} "
            f"at {hex(id(self))}>")
    
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
        return self.time <= self.limit

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


class Process:
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
    class System:
        """
        A simulated system.

        Maintains global simulation data.
        """

        root: Root = field(default_factory=Root)
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

        def schedule[**P](self, 
            src: Callable[P, Event],  
            *args: P.args,
            **kwargs: P.kwargs
        ) -> None:
            event = src(*args, **kwargs)
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


    lax: ClassVar[tuple[str, ...]] = ()
    name: str
    system: System
    __tokens: list[Token]

    def __init__(self, name: str) -> None:
        if not all(s.isidentifier() for s in name.split(".")):
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

    def __repr__(self) -> str:
        return f"<{type(self).__qualname__} '{self.name}' at {hex(id(self))}>"

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            old = getattr(self, name)
        except AttributeError:
            super().__setattr__(name, value)
            return
        if not isinstance(old, Site):
            super().__setattr__(name, value)
            return
        if not isinstance(value, Site):
            raise TypeError("Process site assigned object of wrong type")
        if any(d.d for d in old.data):
            raise ValueError(f"Site '{name}' of process {self.name} "
                "contains data")
        if 1 < len(old.procs):
            raise ValueError(f"Site connects processes {old.procs} and cannot "
                "be replaced")
        if old.index != value.index and name not in self.lax \
            or old.index < value.index:
            raise ValueError("Incompatible index in site assignment")
        if not (isnan(old.const) and isnan(value.const) \
            or old.const == value.const):
            raise ValueError("Incompatible default value in site assignment")
        if len(old.data) != len(value.data) and name not in self.lax:
            raise ValueError("Incompatible lag values")
        value.procs.add(self)
        super().__setattr__(name, value)        

    def __enter__(self):
        self.__tokens.append(PROCESS.set(self))
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        PROCESS.reset(self.__tokens.pop())

    def resolve(self, event: Event) -> None:
        """
        Analyze event and schedule new events as needed.
        
        Typically, dispatches calls to various event scheduling methods.
        """
        pass

    def breakpoint(self, dt: timedelta, priority: Priority = Priority.MAX) \
        -> Event:
        """Schedule a dummy event at specified time."""
        return Event(self.breakpoint, (), dt, priority, 0)


class Site:
    """A simulation data site."""

    @dataclass(slots=True)
    class Update(Update):
        """A future update to a data site."""
        site: "Site"
        data: NumDict | dict[Key, float]
        method: Callable[["Site", NumDict, int, bool], None]
        index: int
        grad: bool
        
        def __init__(self, 
            site: "Site", 
            data: NumDict | dict[Key, float], 
            method: Callable[["Site", NumDict, int, bool], None],
            index: int,
            grad: bool
        ) -> None:
            self.site = site
            self.data = data
            self.method = method
            self.index = index
            self.grad = grad
        
        def apply(self) -> None:
            data = self.data
            if not isinstance(data, NumDict):
                data = self.site.new(data)
            self.method(self.site, data, self.index, self.grad)

    procs: WeakSet[Process]
    index: Index
    const: float
    data: deque[NumDict]
    grad: deque[NumDict]

    def __init__(self, i: Index, d: dict, c: float, l: int = 1) -> None:
        l = 1 if l < 1 else l
        self.procs = WeakSet()
        self.index = i
        self.const = c
        self.data = deque(
            [numdict(i, d, c) for _ in range((l + 1))], 
            maxlen=l)
        self.grad = deque(
            [numdict(i, {}, 0.0) for _ in range((l + 1))], 
            maxlen=l)

    def __iter__(self) -> Iterator[NumDict]:
        yield from self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> NumDict:
        return self.data[i]
    
    def new(self, d: dict) -> NumDict:
        return numdict(self.index, d, self.const)

    def push(self, data: NumDict, index: int = 0, grad: bool = False) -> None:
        if index != 0:
            raise ValueError("Site.push() only operates on index 0")
        d = self.grad if grad else self.data
        d.appendleft(data)

    def add_inplace(self, 
        data: NumDict, 
        index: int = 0, 
        grad: bool = False
    ) -> None:
        d = self.grad if grad else self.data
        d[index] = d[index].sum(data)

    def write_inplace(self, 
        data: NumDict, 
        index: int = 0, 
        grad: bool = False
    ) -> None:
        d = self.grad if grad else self.data
        with d[index].mutable():
            d[index].update(data.d)

    def update(self, 
        data: NumDict | dict[Key, float], 
        method: Callable[["Site", NumDict, int, bool], None] = push,
        index: int = 0,
        grad: bool = False
    ) -> Update:
        if isinstance(data, NumDict) and data.i != self.index:
            raise ValueError("Index of data numdict does not match site")
        if isinstance(data, NumDict) and \
            not (isnan(data.c) and isnan(self.const) or data.c == self.const):
            raise ValueError(f"Default constant {data.c} of data does not "
                f"match site {self.const}")
        return Site.Update(self, data, method, index, grad)
    
    def affected_by(self, *updates, grad: bool = False) -> bool:
        for ud in updates:
            if isinstance(ud, Site.Update) and self is ud.site \
                and ud.grad == grad:
                return True
            if isinstance(ud, UpdateSort) and self.index.depends_on(ud.sort):
                return True
        return False
