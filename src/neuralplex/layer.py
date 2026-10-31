from math import isfinite
from typing import Self
from random import Random

from neuralplex.neuron import Neuron


class Layer:
    def __init__(self, neurons: list[Neuron], step: float | None = None):
        if step is not None and (not isfinite(step) or step <= 0):
            raise ValueError("step must be finite and positive")
        self.neurons = neurons
        self.step = step
        for neuron in neurons:
            if neuron.step is None:
                neuron.step = step

    def connect(self, layer: Self, rng: Random | None = None) -> None:
        for receiver in layer.neurons:
            for source in self.neurons:
                if source not in receiver.weights:
                    weight = rng.uniform(-0.1, 0.1) if rng is not None else None
                    source.connectRHS(receiver, weight)
