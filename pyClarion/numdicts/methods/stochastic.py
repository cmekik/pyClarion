from typing import TypeVar
import random
import math

from .. import numdicts as nd


D = TypeVar("D", bound="nd.NumDict")


def stduniformvariate(self: D) -> D:
    it = self._d if math.isnan(self._c) else self._i
    d = {k: random.random() for k in it} 
    c = math.nan
    return type(self)(self._i, d, c, False)


def normalvariate(self: D, sigma: D) -> D:
    mode = "self" if math.isnan(self._c) else "full"
    d = {k: random.normalvariate(m, s) 
        for k, (m, s) in self.collect(sigma, mode=mode)} 
    c = math.nan
    return type(self)(self._i, d, c, False)


def lognormvariate(self: D, sigma: D) -> D:
    mode = "self" if math.isnan(self._c) else "full"
    d = {k: random.lognormvariate(m, s) 
        for k, (m, s) in self.collect(sigma, mode=mode)} 
    c = math.nan
    return type(self)(self._i, d, c, False)


def vonmisesvariate(self: D, kappa: D) -> D:
    mode = "self" if math.isnan(self._c) else "full"
    d = {k: random.vonmisesvariate(m, v) 
        for k, (m, v) in self.collect(kappa, mode=mode)} 
    c = math.nan
    return type(self)(self._i, d, c, False)


def expovariate(self: D) -> D:
    it = self._d if math.isnan(self._c) else self._i
    d = {k: random.expovariate(self[k]) for k in it} 
    c = math.nan
    return type(self)(self._i, d, c, False)


def gammavariate(self: D, sigma: D) -> D:
    mode = "self" if math.isnan(self._c) else "full"
    d = {k: random.gammavariate(m, s) 
        for k, (m, s) in self.collect(sigma, mode=mode)} 
    c = math.nan
    return type(self)(self._i, d, c, False)


def paretovariate(self: D) -> D:
    it = self._d if math.isnan(self._c) else self._i
    d = {k: random.paretovariate(self[k]) for k in it} 
    c = math.nan
    return type(self)(self._i, d, c, False)


def logisticvariate(self: D, scale: D, *, b: int | None = None) -> D:
    mode = "self" if math.isnan(self._c) else "full"
    d = {k: m + s * (math.log(u) - math.log1p(-u))
        for k, (m, s), u in ((*tup, random.random()) 
            for tup in self.collect(scale, branches=(b,), mode=mode))} 
    c = math.nan
    return type(self)(self._i, d, c, False)


def gumbelvariate(self: D, beta: D) -> D:
    mode = "self" if math.isnan(self._c) else "full"
    d = {k: m - b * math.log(-math.log(random.random())) 
        for k, (m, b) in self.collect(beta, mode=mode)} 
    c = math.nan
    return type(self)(self._i, d, c, False)
