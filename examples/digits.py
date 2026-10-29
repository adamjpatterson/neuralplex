"""Run with: python -m examples.digits (requires the digits extra)."""

from pathlib import Path
from random import Random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from neuralplex import Layer, Network, Neuron


def main():
    seed = 13
    epochs = 20
    step = 0.001
    rng = Random(seed)

    digits = load_digits()
    # Pixel intensities range from 0 to 16. This fixed scaling uses no test data.
    pixels = (digits.data / 16.0).tolist()
    labels = digits.target.tolist()
    train_indices, test_indices = train_test_split(
        list(range(len(labels))), test_size=0.25, random_state=seed, stratify=labels
    )
    network = Network([
        Layer([Neuron() for _ in range(64)]),
        Layer([Neuron(activation='tanh') for _ in range(16)], step=step),
        Layer([Neuron() for _ in range(10)], step=step),
    ], seed=seed)

    print(f'Digits: {len(train_indices)} training images, {len(test_indices)} test images')
    print(f'Network: 64 → 16 (tanh) → 10 (linear); seed={seed}; step={step}')
    for epoch in range(epochs):
        rng.shuffle(train_indices)
        loss = 0.0
        for index in train_indices:
            target = [float(digit == labels[index]) for digit in range(10)]
            network.train(pixels[index], target)
            # train leaves the forward-pass values available from before its update.
            loss += sum((neuron.value - expected)**2 / 2
                        for neuron, expected in zip(network.output_layer.neurons, target))
        print(f'Epoch {epoch + 1:2d}/{epochs}: training loss = {loss / len(train_indices):.6f}')

    # Evaluate only after training; these images never influence parameter updates.
    predictions = []
    for index in test_indices:
        scores = network.predict(pixels[index])
        predictions.append(max(range(10), key=lambda digit: scores[digit]))
    truth = [labels[index] for index in test_indices]
    matrix = confusion_matrix(truth, predictions, labels=list(range(10)))
    accuracy = accuracy_score(truth, predictions)
    majority = max(range(10), key=lambda digit: sum(labels[i] == digit for i in train_indices))
    baseline = sum(label == majority for label in truth) / len(truth)
    report = (
        f'Seed: {seed}; epochs: {epochs}; step: {step}\n'
        f'Train: {len(train_indices)}; test: {len(test_indices)}\n'
        f'Test accuracy: {accuracy:.2%}\n'
        f'Training-majority baseline: {baseline:.2%}\n'
        f'Confusion matrix (rows=true, columns=predicted; digits 0–9):\n{matrix}\n'
    )
    print(report)

    output = Path('results/digits')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.txt').write_text(report)
    with (output / 'predictions.csv').open('w') as file:
        file.write('dataset_index,truth,prediction\n')
        for index, expected, predicted in zip(test_indices, truth, predictions):
            file.write(f'{index},{expected},{predicted}\n')

    figure, axes = plt.subplots(3, 4, figsize=(8, 6))
    # First 12 held-out examples in split order, without selecting successes.
    for axis, index, predicted in zip(axes.flat, test_indices[:12], predictions[:12]):
        axis.imshow(digits.images[index], cmap='gray', vmin=0, vmax=16)
        axis.set_title(f'True: {labels[index]}   Predicted: {predicted}', fontsize=10)
        axis.axis('off')
    figure.suptitle(f'Neural-pleX: held-out digits (overall accuracy {accuracy:.1%})')
    figure.tight_layout()
    figure.savefig(output / 'predictions.png', dpi=150)
    Path('images').mkdir(exist_ok=True)
    figure.savefig('images/digits_predictions.png', dpi=150)
    plt.close(figure)

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
    figure.savefig(output / 'confusion_matrix.png', dpi=150)
    figure.savefig('images/digits_confusion_matrix.png', dpi=150)
    plt.close(figure)
    print(f'Report, predictions, and plots saved to {output}/')


if __name__ == '__main__':
    main()
