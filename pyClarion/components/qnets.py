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


def glorot_normal(fan_in, fan_out):
    """Glorot normal weight initialization."""

    var = 2.0 / (fan_in + fan_out)
    sd = var ** 0.5

    return random.gauss(0.0, sd)


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
    
    Structured as a feed-forward multilayer perceptron and trained using 
    vanilla gradient descent with backpropagation. Weight updates are applied 
    on every step..
    """

    _serves = ConstructType.flow_bb
    
    def __init__(
        self, 
        source: Symbol,
        r_source: Symbol, 
        a_source: Symbol,
        input_domain: FeatureDomain,
        output_interface: FeatureInterface,
        reinforcement_map: ReinforcementMap,
        layers: List[int], 
        gamma: float,
        lr: float
    ):

        self.source = source
        self.r_source = r_source
        self.a_source = a_source
        self.gamma = gamma
        self.lr = lr

        self.input_domain = input_domain
        self.reinforcement_map = reinforcement_map
        self.output_interface = output_interface

        self._layers = tuple(layers)

        self._build_variables()
        self._build_network()

    @property
    def expected(self):

        return frozenset((self.source, self.r_source, self.a_source))

    @property
    def layers(self):

        return self._layers

    def _build_variables(self):

        layer_in = self.input_domain.features
        layer_out = self.output_interface.cmds
        layers = self.layers
        defaults = self.output_interface.defaults
        rdims = self.reinforcement_map.mapping.values()
        eps = nd.epsilon()

        inputs = nd.MutableNumDict(default=0)
        rs = nd.MutableNumDict({k: 0 for k in rdims}, default=0)
        qs = nd.MutableNumDict({k: eps for k in layer_out}, default=eps)
        inputs_lag1 = nd.MutableNumDict(default=0)
        actions_lag1 = nd.MutableNumDict({k: 1 for k in defaults}, default=0)
        qs_lag1 = nd.MutableNumDict({k: eps for k in layer_out}, default=eps)

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
        self._qs = qs
        self._inputs_lag1 = inputs_lag1
        self._actions_lag1 = actions_lag1
        self._qs_lag1 = qs_lag1
        self.weights = weights
        self.biases = biases

    def _build_network(self):

        inputs = self._inputs
        rs = self._rs
        qs_next = self._qs
        actions = self._actions_lag1
        gamma = self.gamma

        weights = self.weights
        biases = self.biases

        keyfunc = lambda k: k.dim
        with nd.GradientTape(persistent=True) as tape:
            output = inputs
            for i, (w, b) in enumerate(zip(weights, biases)):
                output = nd.set_by(w, output, keyfunc=lambda k: k[1])
                output = output * w
                output = nd.sum_by(output, keyfunc=lambda k: k[0])
                output = output + b
                if i == len(weights) - 1:
                    qs = output
                output = nd.sigmoid(output)
            # Assuming exactly one action is mapped to a value of 1.
            q_action = nd.sum_by((qs * actions), keyfunc=keyfunc)
            # Assuming qs_next is unsquashed.
            max_qs_next = nd.max_by(qs_next, keyfunc=keyfunc)
            errors = (rs + (gamma * max_qs_next)) - q_action
            loss_dict = (errors ** 2) / 2
            loss = nd.sum_by(loss_dict, keyfunc=lambda k: "loss")

        self._tape = tape
        self._output = output
        self.loss = loss
        self.loss_dict = loss_dict

    def call(self, inputs):

        strengths = inputs[self.source]
        input_features = self.input_domain.features

        tape = self._tape
        output = self._output
        loss = self.loss

        _inputs = nd.keep(strengths, keys=input_features)
        self._inputs.clear()
        self._inputs.update(_inputs)

        loss, output = tape.evaluate(loss, output)
        self.loss = loss
        self._output = output

        return output

    def update(self, inputs, output):
        
        strengths = inputs[self.source]
        r_strengths = inputs[self.r_source]
        a_strengths = inputs[self.a_source]

        input_features = self.input_domain.features
        r_map = self.reinforcement_map.mapping

        tape = self._tape
        inputs_lag1 = self._inputs_lag1
        output = self._output
        loss = self.loss
        weights = self.weights
        biases = self.biases
        variables = tuple(weights + biases)
        lr = self.lr

        rs = nd.keep(r_strengths, keys=r_map)
        rs = nd.NumDict({r_map[k]: v for k, v in r_strengths.items()}, 0)

        self._inputs.clear()
        self._inputs.update(inputs_lag1)

        self._actions_lag1.clear()
        self._actions_lag1.update(a_strengths)

        self._qs.clear()
        self._qs.update(nd.log(output / (1 - output)))

        self._rs.clear()
        self._rs.update(rs)

        loss, output = tape.evaluate(loss, output)
        self._output = output

        loss, grads = tape.gradients(loss, variables, forward=False)
        for var, grad in zip(variables, grads):
            var -= lr * grad
        
        self.loss = loss

        _inputs = nd.keep(strengths, keys=input_features)
        self._inputs_lag1.clear()
        self._inputs_lag1.update(_inputs)
        