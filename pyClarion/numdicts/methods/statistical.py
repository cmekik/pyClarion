from typing import TypeVar
import math
import statistics as stats

from ..keys import Key, KeyForm
from ..indices import Index
from .. import numdicts as nd


D = TypeVar("D", bound="nd.NumDict")


def mean(self: D, by: KeyForm | None = None) -> D:
    match by:
        case None:
            by = self._i.kf.agg
            mode = "self" if math.isnan(self._c) else "full"
            it = ()
            i = Index(self._i.root, by)
            c = stats.fmean(self.group(by, mode=mode).get(Key(), (self._c,)))
        case by:
            if not by < self._i.kf:
                raise ValueError(f"Keyform {by.as_key()} cannot reduce "
                    f"{self._i.kf.as_key()}")
            mode = "self" if math.isnan(self._c) else "full"
            it = self.group(by, mode=mode).items()
            i = Index(self._i.root, by)
            c = self._c if mode == "self" else float("nan")
    d = {k: stats.fmean(vs) for k, vs in it}
    return type(self)(i, d, c, False)


def stdev(self: D, by: KeyForm | None = None) -> D:
    match by:
        case None:
            by = self._i.kf.agg
            mode = "self" if math.isnan(self._c) else "full"
            it = ()
            i = Index(self._i.root, by)
            c = stats.stdev(self.group(by, mode=mode).get(Key(), (self._c,)))
        case by:
            if not by < self._i.kf:
                raise ValueError(f"Keyform {by.as_key()} cannot reduce "
                    f"{self._i.kf.as_key()}")
            mode = "self" if math.isnan(self._c) else "full"
            it = self.group(by, mode=mode).items()
            i = Index(self._i.root, by)
            c = self._c if mode == "self" else float("nan")
    d = {k: stats.stdev(vs) for k, vs in it}
    return type(self)(i, d, c, False)


def variance(self: D, by: KeyForm | None = None) -> D:
    match by:
        case None:
            by = self._i.kf.agg
            mode = "self" if math.isnan(self._c) else "full"
            it = ()
            i = Index(self._i.root, by)
            c = stats.variance(self.group(by, mode=mode).get(Key(), (self._c,)))
        case by:
            if not by < self._i.kf:
                raise ValueError(f"Keyform {by.as_key()} cannot reduce "
                    f"{self._i.kf.as_key()}")
            mode = "self" if math.isnan(self._c) else "full"
            it = self.group(by, mode=mode).items()
            i = Index(self._i.root, by)
            c = self._c if mode == "self" else float("nan")
    d = {k: stats.variance(vs) for k, vs in it}
    return type(self)(i, d, c, False)


def pstdev(self: D, by: KeyForm | None = None) -> D:
    match by:
        case None:
            by = self._i.kf.agg
            mode = "self" if math.isnan(self._c) else "full"
            it = ()
            i = Index(self._i.root, by)
            c = stats.pstdev(self.group(by, mode=mode).get(Key(), (self._c,)))
        case by:
            if not by < self._i.kf:
                raise ValueError(f"Keyform {by.as_key()} cannot reduce "
                    f"{self._i.kf.as_key()}")
            mode = "self" if math.isnan(self._c) else "full"
            it = self.group(by, mode=mode).items()
            i = Index(self._i.root, by)
            c = self._c if mode == "self" else float("nan")
    d = {k: stats.pstdev(vs) for k, vs in it}
    return type(self)(i, d, c, False)


def pvariance(self: D, by: KeyForm | None = None) -> D:
    match by:
        case None:
            by = self._i.kf.agg
            mode = "self" if math.isnan(self._c) else "full"
            it = ()
            i = Index(self._i.root, by)
            c = stats.pvariance(self.group(by, mode=mode).get(Key(), (self._c,)))
        case by:
            if not by < self._i.kf:
                raise ValueError(f"Keyform {by.as_key()} cannot reduce "
                    f"{self._i.kf.as_key()}")
            mode = "self" if math.isnan(self._c) else "full"
            it = self.group(by, mode=mode).items()
            i = Index(self._i.root, by)
            c = self._c if mode == "self" else float("nan")
    d = {k: stats.pvariance(vs) for k, vs in it}
    return type(self)(i, d, c, False)