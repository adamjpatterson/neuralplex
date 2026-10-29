from math import tanh
from random import uniform
from typing import Self


class Neuron:
    """One weight per incoming connection, one bias, and an activation."""

    def __init__(self, bias: float = 0, step: float | None = None,
                 name=None, activation: str = "linear"):
        if activation not in ("linear", "tanh"):
            raise ValueError("activation must be 'linear' or 'tanh'")
        if step is not None and step <= 0:
            raise ValueError("step must be positive")
        self.bias = bias
        self.step = step
        self.name = name
        self.activation = activation
        self.weights: dict[Self, float] = {}
        self.neuronsRHS: list[Self] = []
        self.value = 0.0
        self.gradient = 0.0  # d(loss) / d(output)
        self.delta = 0.0  # d(loss) / d(weighted sum)

    @property
    def neuronsLHS(self) -> list[Self]:
        return list(self.weights)

    def connectRHS(self, neuronRHS: Self, weight: float | None = None) -> None:
        if neuronRHS is self:
            raise ValueError("A neuron cannot connect to itself")
        if neuronRHS not in self.neuronsRHS:
            self.neuronsRHS.append(neuronRHS)
            neuronRHS.weights[self] = uniform(-0.1, 0.1) if weight is None else weight

    def connectLHS(self, neuronLHS: Self, weight: float | None = None) -> None:
        neuronLHS.connectRHS(self, weight)

    def disconnectRHS(self, neuronRHS: Self) -> None:
        """Remove just this edge and its weight; save the weight before reconnecting."""
        if neuronRHS in self.neuronsRHS:
            self.neuronsRHS.remove(neuronRHS)
            del neuronRHS.weights[self]

    def disconnectLHS(self, neuronLHS: Self) -> None:
        neuronLHS.disconnectRHS(self)

    def activate(self) -> None:
        total = self.bias
        for source, weight in self.weights.items():
            total += source.value * weight
        self.value = tanh(total) if self.activation == "tanh" else total

    def propagate(self) -> None:
        """Pass gradients backward without changing any parameters."""
        derivative = 1 - self.value ** 2 if self.activation == "tanh" else 1.0
        self.delta = self.gradient * derivative
        for source, weight in self.weights.items():
            source.gradient += self.delta * weight
