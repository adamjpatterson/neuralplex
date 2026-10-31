from math import isfinite
from random import Random

from neuralplex.layer import Layer
from neuralplex.neuron import Neuron


class Network:
    def __init__(self, layers: list[Layer], seed: int | None = None):
        if len(layers) < 2 or any(not layer.neurons for layer in layers):
            raise ValueError("Provide at least two nonempty layers")
        neurons = [neuron for layer in layers for neuron in layer.neurons]
        if len(set(neurons)) != len(neurons):
            raise ValueError("Each neuron must appear exactly once")
        self.layers = layers
        self.input_layer = layers[0]
        self.output_layer = layers[-1]
        rng = Random(seed)
        for source, receiver in zip(layers, layers[1:]):
            source.connect(receiver, rng)

    def predict(self, X: list[float]) -> list[float]:
        if len(X) != len(self.input_layer.neurons):
            raise ValueError("Supply one input value per input neuron")

        # Check manual connection edits before using the layer traversal order.
        positions = {neuron: index for index, layer in enumerate(self.layers)
                     for neuron in layer.neurons}
        if len(positions) != sum(len(layer.neurons) for layer in self.layers):
            raise ValueError("Each neuron must appear exactly once")
        for receiver, index in positions.items():
            for source in receiver.neuronsLHS:
                if source not in positions or positions[source] >= index:
                    raise ValueError("Connections must run from earlier to later layers")
            if any(target not in positions for target in receiver.neuronsRHS):
                raise ValueError("All connected neurons must belong to this network")

        # Input neurons hold data directly; they have no trainable parameters.
        for neuron, value in zip(self.input_layer.neurons, X):
            neuron.value = value
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                neuron.activate()
        return [neuron.value for neuron in self.output_layer.neurons]

    def train(self, X_train: list[float], y_train: list[float]) -> None:
        """One SGD step for loss = sum((prediction - target)**2) / 2."""
        if len(y_train) != len(self.output_layer.neurons):
            raise ValueError("Supply one target value per output neuron")
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                if neuron.step is None or not isfinite(neuron.step) or neuron.step <= 0:
                    raise ValueError("Every computational neuron needs a finite, positive step")
        self.predict(X_train)

        # Clear all gradients, including those accumulated at input neurons.
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.gradient = 0.0
                neuron.delta = 0.0
        for neuron, target in zip(self.output_layer.neurons, y_train):
            neuron.gradient = neuron.value - target

        # Each receiver adds its contribution before its sources propagate.
        for layer in reversed(self.layers[1:]):
            for neuron in layer.neurons:
                neuron.propagate()

        # Only now update parameters: the backward pass used the old weights.
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                for source in neuron.weights:
                    neuron.weights[source] -= neuron.step * neuron.delta * source.value
                neuron.bias -= neuron.step * neuron.delta

    def detach(self, neuron: Neuron) -> None:
        """Remove a hidden neuron and its edges, between training steps."""
        if neuron in self.input_layer.neurons or neuron in self.output_layer.neurons:
            raise ValueError("Input and output neurons cannot be detached")
        for layer in self.layers[1:-1]:
            if neuron in layer.neurons:
                for source in list(neuron.neuronsLHS):
                    neuron.disconnectLHS(source)
                for receiver in list(neuron.neuronsRHS):
                    neuron.disconnectRHS(receiver)
                layer.neurons.remove(neuron)
                return
        raise ValueError("Neuron is not a hidden member of this network")
