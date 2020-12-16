"""Definitions for setting up Q-Nets."""


__all__ = ["SimpleQNet", "ReinforcementMap"]


from .. import numdicts as nd
from ..base.symbols import ConstructType, Symbol, feature, features, buffer  
from ..base.components import FeatureDomain, FeatureInterface, Propagator

from itertools import product
from typing import List, Mapping, Tuple, Hashable
from dataclasses import dataclass
import random
import math
import warnings


def glorot_normal(fan_in, fan_out):
    """Glorot normal weight initialization."""

    var = 2.0 / (fan_in + fan_out)
    sd = var ** 0.5

    return random.gauss(0.0, sd)


class NetConfigWarning(UserWarning):
    """Signals a non-fatal error during neural network configuration."""
    pass


@dataclass
class ReinforcementMap(FeatureDomain):
    """
    A feature domain for specifying reinforcement signals.

    :param mapping: A one-to-one mapping from features to dimensions. Each 
        feature is assumed to represent a reinforcement signal for the 
        corresponding dimension. Will raise a ValueError if the mapping is 
        found not to be one-to-one.
    """

    mapping: Mapping[feature, Tuple[Hashable, int]]

    def _validate_data(self):

        if len(set(self.mapping.keys())) != len(set(self.mapping.values())):
            raise ValueError("Mapping must be one-to-one.")

    def _set_interface_properties(self):

        self._features = set(self.mapping)


class SimpleQNet(Propagator):
    """
    A simple q-network.

    Supports multiple action dimensions, but does not support action parameters
    (i.e., assumes action space is discrete). Will issue a warning if configured
    with an interface defining action parameters.
    
    Structured as a feed-forward multilayer perceptron and trained using 
    vanilla gradient descent with backpropagation. Weight updates are applied
    on every step.
    """

    _serves = ConstructType.flow_bb
    
    def __init__(
        self, 
        x_source: Symbol,
        r_source: Symbol, 
        a_source: Symbol,
        domain: FeatureDomain,
        interface: FeatureInterface,
        r_map: ReinforcementMap,
        layers: List[int], 
        gamma: float,
        lr: float
    ) -> None:

        if len(interface.params) != 0:
            msg = (
                "Received interface defining action params; "
                "these are not supported by {} and will be ignored."
            )
            tname = type(self).__name__
            warnings.warn(msg.format(tname), category=NetConfigWarning)

        self.x_source = x_source
        self.r_source = r_source
        self.a_source = a_source
        self.gamma = gamma
        self.lr = lr

        self.domain = domain
        self.interface = interface
        self.r_map = r_map

        self._layers = tuple(layers)

        self._build_variables()
        self._build_network()

    @property
    def expected(self):

        return frozenset((self.x_source, self.r_source, self.a_source))

    @property
    def layers(self):

        return self._layers

    def _build_variables(self):

        layers = self.layers
        layer_in = self.domain.features
        layer_out = self.interface.cmds
        defaults = self.interface.defaults
        rdims = self.r_map.mapping.values()

        inputs = nd.MutableNumDict(default=0)
        rs = nd.MutableNumDict({k: 0 for k in rdims}, default=0)
        qs_next = nd.MutableNumDict({k: 0 for k in layer_out}, default=0)
        actions = nd.MutableNumDict({k: 1 for k in defaults}, default=0)
        inputs_lag1 = nd.MutableNumDict(default=0)

        weights, biases = [], []
        hiddens = [{(l, i) for i in range(n)} for l, n in enumerate(layers)]
        w_keys = list(zip([layer_in] + hiddens, hiddens + [layer_out]))

        for layer_in, layer_out in w_keys:

            m, n = len(layer_in), len(layer_out)
            w_keys = product(layer_out, layer_in)
            
            b = nd.MutableNumDict({j: 0 for j in layer_out})
            w = nd.MutableNumDict({k: glorot_normal(m, n) for k in w_keys})
            
            biases.append(b)
            weights.append(w)

        self._inputs = inputs
        self._rs = rs
        self._qs_next = qs_next
        self._actions = actions
        self._inputs_lag1 = inputs_lag1

        self.weights = weights
        self.biases = biases

    def _build_network(self):

        inputs = self._inputs
        rs = self._rs
        qs_next = self._qs_next
        actions = self._actions
        gamma = self.gamma

        weights = self.weights
        biases = self.biases

        get_dim = feature.dim.fget
        with nd.GradientTape(persistent=True) as tape:
            qs = inputs
            for i, (w, b) in enumerate(zip(weights, biases)):
                qs = nd.set_by(w, qs, keyfunc=lambda k: k[1])
                qs = qs * w
                qs = nd.sum_by(qs, keyfunc=lambda k: k[0])
                qs = qs + b
                if i != len(weights) - 1: # don't squash final layer output
                    qs = nd.sigmoid(qs)
            # Assuming exactly one action/dim is mapped to a value of 1.
            q_action = nd.sum_by((qs * actions), keyfunc=get_dim)
            max_qs_next = nd.max_by(qs_next, keyfunc=get_dim)
            errors = (rs + (gamma * max_qs_next)) - q_action
            loss_dict = (errors ** 2) / 2
            loss = nd.sum_by(loss_dict, keyfunc=lambda k: "loss")

        self._tape = tape
        self._qs = qs
        self.loss = loss
        self.loss_dict = loss_dict

    def call(self, inputs):

        x_strengths = inputs[self.x_source]
        input_features = self.domain.features

        tape = self._tape
        qs = self._qs
        loss = self.loss

        _inputs = nd.keep(x_strengths, keys=input_features)
        self._inputs.clearupdate(_inputs)

        loss, qs = tape.evaluate(loss, qs)
        self.loss = loss
        self._qs = qs

        # squash the q values to lie in (0, 1)
        d = nd.sigmoid(qs)

        return d

    def update(self, inputs, output) -> None:
        
        x_strengths = inputs[self.x_source]
        r_strengths = inputs[self.r_source]
        a_strengths = inputs[self.a_source]

        input_features = self.domain.features
        r_mapping = self.r_map.mapping

        tape = self._tape
        inputs_lag1 = self._inputs_lag1
        qs = self._qs
        loss = self.loss
        weights = self.weights
        biases = self.biases
        variables = tuple(weights + biases)
        lr = self.lr

        rs = nd.keep(r_strengths, keys=r_mapping)
        rs = nd.NumDict({r_mapping[k]: v for k, v in r_strengths.items()}, 0)
        
        self._inputs.clearupdate(inputs_lag1)
        self._actions.clearupdate(a_strengths)
        self._qs_next.clearupdate(qs)
        self._rs.clearupdate(rs)

        loss, qs = tape.evaluate(loss, qs)
        self._qs = qs

        loss, grads = tape.gradients(loss, variables, forward=False)
        for var, grad in zip(variables, grads):
            var -= lr * grad
        
        self.loss = loss

        _inputs_lag1 = nd.keep(x_strengths, keys=input_features)
        self._inputs_lag1.clearupdate(_inputs_lag1)
        