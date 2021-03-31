
__all__ = ["MatchStatistics", "RuleStatDB", "RERupdater"]

from ..base.symbols import ConstructType, Symbol, rule, chunk, flow_tt
from ..base.components import Process
from .chunks_ import Chunks
from .. import numdicts as nd
from ..rules import Rules
from math import log2

from typing import (
    Mapping, MutableMapping, TypeVar, Generic, Type, Dict, FrozenSet, Set,
    Tuple, overload, cast, Callable, Hashable, Tuple, List, Collection, Any
)
from contextlib import contextmanager
from types import MappingProxyType

class MatchStatistics(object):
    """ updates the rule stats"""

    def __init__(
        self
    ):
        self.PM: int = 0
        self.NM: int = 0

    def update_MC(self, criterion: bool, matched: bool, fired: bool):
        """give an input of boolean whether match count was positive
        or negative, if positive (true), update Posotive Match Count(PM), 
        if negative(false), update negative Match Count(PM) """
        if matched and fired and criterion: 
            self.PM +=1
        elif matched and fired and not criterion:
            self.NM +=1
        else:
            pass

class RuleStatDB(Mapping):
    def __init__(
        self
    ):
        self._dict: dict = {}
        self._invoked: set = set()
        self._reset: set = set()
        self._del: set = set()
        self._add: set = set()

    def __repr__(self):
    
        return "<RuleStatDB {}>".format(self._dict)

    def __len__(self):

        return len(self._dict)

    def __iter__(self):

        yield from self._dict

    def __getitem__(self, key):

        return self._dict[key]

    def __delitem__(self, key):

        del self._dict[key]

    def add(self, key: rule):
        """Add item to ruleStatDB database, they key must be a rule."""

        self._dict[key] = MatchStatistics()

    def step(self):
        """
        Update BLA database according to promises.

        Steps every existing BLA, adds invocations as promised. Also adds 
        and removes entries according to promises made.
        """

        for key in self._del:
            del self[key]
        self._del.clear()
 
        for key, match_stat in self.items():
            if key in self._invoked:
                match_stat.step(invoked=True)
            else:
                match_stat.step(invoked=False)
        self._invoked.clear()

        for key in self._add:
            self.add(key)
        self._add.clear()

    def register_match(self, key, add_new=False):
        """
        Promise key will be treated as invoked on next update.
        
        If key does not already exist in self, add the key if add_new is True, 
        otherwise throw KeyError. 
        """

        if key in self._add or key in self._del or key in self._invoked:
            msg = "Key {} already registered for a promised update."
            raise ValueError(msg.format(key))
        else:
            if key in self:
                self._invoked.add(key)
            elif add_new:
                self._add.add(key)
            else: 
                raise KeyError("Key not in RuleStat database.")

    def request_add(self, key):
        """Promise key will be added to database on next update."""

        if key in self._add or key in self._del or key in self._invoked:
            msg = "Key {} already registered for a promised update."
            raise ValueError(msg.format(key))
        else:
            self._add.add(key)

    def request_del(self, key):
        """Promise key will be deleted from database on next update."""

        if key in self._add or key in self._del or key in self._invoked:
            msg = "Key {} already registered for a promised update."
            raise ValueError(msg.format(key))
        else:
            self._del.add(key)


class RERUpdater(Process):
    """ initialize: ruleDB and RuleStatDB, flow_tt, """
    _serves = ConstructType.updater
   
    def __init__(
        self, 
        x_source: Symbol,
        y_source: Symbol,
        r_source: Symbol,
        a_source: Symbol,
        r_symb: Symbol, 
        x_th: float,
        a_th: float,
        r_th: float,
        rdb: Rules,
        rsdb: RuleStatDB, 
        ccdb: Chunks,  # condition_chunkDB
        acdb: Chunks,  # action_chunkDB
        
        c1: float = 1.0,
        c2: float = 2.0
    ):
        super().__init__(expected=(x_source, y_source, r_source, a_source))
        self.rdb = rdb
        self.rsdb = rsdb
        self.ccdb = ccdb
        self.acdb = acdb
        self.c1 = c1
        self.c2 = c2
        self.a_th = a_th
        self.x_th = x_th
        self.r_th = r_th
        self.r_symb = r_symb
        
    def IG(self, A: rule, B: rule) -> float:
        """ returns a float """
        pm_a = self.rsdb[A].PM
        nm_a = self.rsdb[A].NM
        pm_b = self.rsdb[B].PM
        nm_b = self.rsdb[B].NM

        ig = log2((pm_a+self.c1)/(pm_a+nm_a+self.c2)) - \
            log2((pm_b+self.c1)/(pm_b+nm_b+self.c2))

        return ig   
    
    def call(self, inputs: Mapping[Any, nd.NumDict]):# -> nd.NumDict:
        x, y, r, a = self.extract_inputs(inputs)
        data = self.extract_inputs(inputs)
        crit: bool = r[self.r_symb] > self.r_th

        x_active = nd.threshold(x, th=self.x_th)
        x_match = self.ccdb.match(x_active)
        a_active = nd.threshold(a, th=self.a_th)
        a_match = self.acdb.match(a_active)

        # do extraction code here

        #determine if outcome is positive/neg/neither
        if crit and not x_match and not a_match:
            # next check if there is a rule that match at the top level
            # if not add a new rule
            
            # maybe match() should be checked against chunk??? 
            
            
            
            





        return super().call(inputs)
        
        
    def rule_extraction_criterion(self):
        """return success(true) or not(false)"""
        pass

    def rule_specialization_criterion(self):
        """return success(true) or not(false)"""
        pass

    def rule_generalization_criterion(self):
        """return success(true) or not(false)"""
        pass



    




