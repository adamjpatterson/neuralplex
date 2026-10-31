"""Rendering helpers for the nibble notebook; no training happens here."""

from copy import deepcopy
from math import sqrt

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def visualize_network(before, after, inputs):
    """Return an inline before/after figure without writing files.

    Models are copied so exploring a new input does not alter their cached values.
    The two networks must have the same three-layer topology and unique names.
    """
    snapshots = [deepcopy(before), deepcopy(after)]
    network = snapshots[1]
    names = [[n.name for layer in model.layers for n in layer.neurons] for model in snapshots]
    if names[0] != names[1] or len(set(names[0])) != len(names[0]) or None in names[0]:
        raise ValueError('Use matching networks with unique, nonempty node names')
    if any(len(model.layers) != 3 for model in snapshots):
        raise ValueError('This example visualizer expects three layers')
    nibble = list(inputs)
    input_label = ''.join(str(bit) for bit in nibble)
    for model in snapshots:
        model.predict(nibble)
    maximum_weight = max((abs(weight) for model in snapshots
                          for layer in model.layers for neuron in layer.neurons
                          for weight in neuron.weights.values()), default=0) or 1
    maximum_activation = max(abs(n.value) for model in snapshots for layer in model.layers for n in layer.neurons) or 1
    colors = ['#32965a', '#ce4848', '#e2b52e']
    positions = {n.name: (140 + layer_index * 300, 65 + (index + 0.5) * 480 / len(layer.neurons))
                 for layer_index, layer in enumerate(network.layers) for index, n in enumerate(layer.neurons)}
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    for stage, model, axis in zip(['before', 'after'], snapshots, axes):
        title = f'{stage.title()} training — {input_label} → {model.output_layer.neurons[0].value:.3f}'
        for layer_index, layer in enumerate(model.layers):
            for neuron in layer.neurons:
                # Area above a small visibility floor is proportional to |activation|.
                radius = sqrt(36 + 540 * abs(neuron.value) / maximum_activation)
                x, y = positions[neuron.name]
                axis.scatter(x, y, s=radius**2 * 2, color=colors[layer_index], edgecolor='white', zorder=3)
                axis.text(x + (40 if layer_index == 2 else 25), y, f'{neuron.name}\n{neuron.value:.2f}', fontsize=8, va='center', zorder=4)
        for layer in model.layers:
            for receiver in layer.neurons:
                for source, weight in receiver.weights.items():
                    color = '#377eb8' if weight >= 0 else '#e68126'
                    width = 5 * abs(weight) / maximum_weight
                    x1, y1 = positions[source.name]
                    x2, y2 = positions[receiver.name]
                    axis.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.55, zorder=1)
        axis.set(xlim=(50, 870), ylim=(610, 10), title=title)
        axis.axis('off')
    figure.legend(handles=[Line2D([0], [0], color='#377eb8', label='Positive weight'),
                           Line2D([0], [0], color='#e68126', label='Negative weight')],
                  loc='lower center', ncol=2)
    figure.suptitle('Nibble Challenge • edge thickness = |weight| • node area = |activation| + visibility floor')
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    return figure
