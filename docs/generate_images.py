"""Generate the root README images."""

from copy import deepcopy
from math import sqrt
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from neuralplex import Network, Layer, Neuron

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


def plot_predictions(images, labels, predictions=None, title='Sample digits'):
    """Return a figure of the first twelve supplied images, without selecting successes."""
    count = min(12, len(images))
    if count == 0:
        raise ValueError('Supply at least one image')
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis('off')
        if index >= len(images):
            continue
        axis.imshow(images[index], cmap='gray', vmin=0, vmax=16)
        label = f'True: {labels[index]}'
        if predictions is not None:
            label += f'   Predicted: {predictions[index]}'
        axis.set_title(label, fontsize=10)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_confusion_matrix(matrix):
    """Return a figure; rows are true digits and columns are predicted digits."""
    figure, axis = plt.subplots(figsize=(7, 6))
    heatmap = axis.imshow(matrix, cmap='Blues')
    axis.set(xticks=range(10), yticks=range(10), xlabel='Predicted digit',
             ylabel='True digit', title='Neural-pleX: held-out confusion matrix')
    for row in range(10):
        for column in range(10):
            axis.text(column, row, str(matrix[row, column]), ha='center', va='center',
                      color='white' if matrix[row, column] > matrix.max() / 2 else 'black')
    figure.colorbar(heatmap, ax=axis)
    figure.tight_layout()
    return figure


def main():
    seed = 42
    output = Path(__file__).resolve().parents[1] / 'images'
    output.mkdir(exist_ok=True)

    # Fixed documentation scenario: four bits, 10,000 updates, linear activations.
    nibble = Network([
        Layer([Neuron(name=f'i{i}') for i in range(4)]),
        Layer([Neuron(name=f'h{i}') for i in range(8)], step=0.01),
        Layer([Neuron(name='output')], step=0.01),
    ], seed=seed)
    before = deepcopy(nibble)
    rng = Random(seed)
    for _ in range(10000):
        number = rng.randrange(16)
        nibble.train([int(bit) for bit in f'{number:04b}'], [number])
    figure = visualize_network(before, nibble, [1, 0, 1, 0])
    figure.savefig(output / 'nibble_multiweight.png', dpi=150)
    plt.close(figure)
    predictions = [nibble.predict([int(bit) for bit in f'{number:04b}'])[0] for number in range(16)]
    squared_error = sum((prediction-number)**2 for number, prediction in enumerate(predictions))
    print(f'Nibble R²: {1 - squared_error / sum((n-7.5)**2 for n in range(16)):.9f}')

    # Fixed documentation scenario: 20 epochs on a 75%/25% stratified digits split.
    digits = load_digits()
    pixels = (digits.data / 16.0).tolist()
    labels = digits.target.tolist()
    train, test = train_test_split(list(range(len(labels))), test_size=0.25,
                                   random_state=seed, stratify=labels)
    network = Network([
        Layer([Neuron() for _ in range(64)]),
        Layer([Neuron(activation='tanh') for _ in range(16)], step=0.001),
        Layer([Neuron() for _ in range(10)], step=0.001),
    ], seed=seed)
    rng = Random(seed)
    for _ in range(20):
        rng.shuffle(train)
        for index in train:
            network.train(pixels[index], [float(d == labels[index]) for d in range(10)])
    predictions = []
    for index in test:
        scores = network.predict(pixels[index])
        predictions.append(max(range(10), key=scores.__getitem__))
    truth = [labels[index] for index in test]
    accuracy = accuracy_score(truth, predictions)
    matrix = confusion_matrix(truth, predictions, labels=list(range(10)))
    print(f'Digits accuracy: {accuracy:.2%} ({len(train)} train, {len(test)} test)')
    figure = plot_predictions(digits.images[test[:12]], truth[:12], predictions[:12],
                              title=f'Neural-pleX: held-out digits (overall accuracy {accuracy:.1%})')
    figure.savefig(output / 'digits_predictions.png', dpi=150)
    plt.close(figure)
    figure = plot_confusion_matrix(matrix)
    figure.savefig(output / 'digits_confusion_matrix.png', dpi=150)
    plt.close(figure)
    print(f'Wrote three README images to {output}')


if __name__ == '__main__':
    main()
