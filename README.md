# Neural-pleX

[![PyPI version](https://img.shields.io/pypi/v/neuralplex.svg)](https://pypi.org/project/neuralplex/)
[![Python versions](https://img.shields.io/pypi/pyversions/neuralplex.svg)](https://pypi.org/project/neuralplex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/adamjpatterson/neuralplex/blob/main/LICENSE)

An object oriented educational/experimental neural network implementation.

## Introduction

**Neural-pleX** is an intuitive object oriented neural network implementation. The Neural-pleX API consists of Network, Layer, and Neuron constructors. The networks can be easily [visualized](#visualizations-of-the-network-before-and-after-training) using a visualization library.

## Table of contents

- [Installation](#installation)
- [How it works](#how-it-works)
- [Examples](#examples)
- [The nibble challenge](#the-nibble-challenge)
- [The digits experiment](#the-digits-experiment)
- [Tests](#tests)
- [Documentation maintenance](#documentation-maintenance)

## Installation

To use the published package:

```bash
python -m pip install neuralplex
```

To explore the notebooks or modify the package, clone the repository and install
the checkout:

```bash
git clone https://github.com/adamjpatterson/neuralplex.git
cd neuralplex
python -m pip install -e .
```

Follow each example’s setup instructions below to install its notebook and
visualization dependencies.

## How it works

This implementation demonstrates each component of the API. A 3 layer neural network is constructed that has a 4 neuron input Layer and a 1 neuron output layer. The hidden layer has 8 neurons.

### Implement a 3 layer neural network

#### Import the Network, Layer, and Neuron classes.

```python
from neuralplex import Network, Layer, Neuron
```

#### Set a step.

`step` is the learning rate: it controls the size of weight and bias updates.
Each update subtracts `step` times the corresponding gradient.
`Layer(step=...)` supplies an initial default only to neurons whose `step` is
`None`. Changing `layer.step` later does not change the neurons; update each
`neuron.step` directly to change its learning rate.

```python
STEP = 0.01
```

#### Construct a neural network by specifying the Neurons for each Layer and adding the Layers to a Network.

The resulting neural network will have 4 input neurons, 1 output neuron, and 8 neurons in the hidden layer.

```python
l1 = Layer(neurons=[Neuron() for i in range(0, 4)], step=STEP)
l2 = Layer(neurons=[Neuron() for i in range(0, 8)], step=STEP)
l3 = Layer(neurons=[Neuron()], step=STEP)
n1 = Network([l1, l2, l3], seed=42)
```

#### With the Network defined, you can train the network.

Here the network is trained to recognize the nibble 1111 as the decimal number 15.

```python
n1.train([1,1,1,1], [15])
```

You can generate and print a prediction using the `predict` method. Because the network underwent just one iteration of training, the estimate will likely be inaccurate. The accuracy of the prediction can be improved by iteratively training the network. Please see [the nibble challenge](#the-nibble-challenge) for an example of how to iteratively train the network.

```python
prediction = n1.predict([1,1,1,1])
print(prediction)
```

### How a neuron learns

Each computational neuron calculates
`activation(sum(source.value * weight) + bias)`. Inputs hold raw values.
`Neuron()` uses linear activation; `Neuron(activation="tanh")` is available for
hidden layers. Linear layers are sufficient for the nibble conversion task.

`train` minimizes half the sum of squared output errors. It runs the forward
pass, accumulates gradients backward through every connection, and only then
updates weights and biases. `delta` is the derivative of the loss with respect
to the neuron's weighted sum plus bias, before activation.
A weight's gradient is `delta * source.value`;
the bias gradient is `delta`. The code uses explicit loops, not automatic
differentiation. Every non-input neuron needs a finite, positive `step` for training.

`Neuron(bias=...)` sets the bias. Each incoming connection has its own weight,
stored in `receiver.weights[source]`. `Network(..., seed=42)` initializes
weights reproducibly in [-0.1, 0.1]. Connections added
later accept an explicit `weight`; repeated connection calls preserve existing
weights. Visualization modules read connections directly from
`receiver.weights.items()`. Give nodes unique names when visualizing them.

### Detach and reconnect between training steps

You can change an existing connection weight through `receiver.weights[source]`.
To add or remove connections, use `connectRHS`/`connectLHS` and
`disconnectRHS`/`disconnectLHS` rather than inserting or deleting dictionary
entries. These methods keep incoming weights and outgoing connections consistent.

```python
hidden = n1.layers[1].neurons[0]
incoming = dict(hidden.weights)
outgoing = {target: target.weights[hidden] for target in hidden.neuronsRHS}
n1.detach(hidden)

# Restore the neuron, including its saved connection weights.
n1.layers[1].neurons.append(hidden)
for source, weight in incoming.items():
    source.connectRHS(hidden, weight=weight)
for target, weight in outgoing.items():
    hidden.connectRHS(target, weight=weight)
```

The neuron's bias, activation, and learning rate remain on the object. For one
edge, use `source.disconnectRHS(target)` and `source.connectRHS(target, weight)`.
Removing an edge does not cascade. A non-input neuron without incoming
connections outputs `activation(bias)`; an empty hidden layer left by detachment
is allowed. Connections must run from earlier to later layers; input/output neurons cannot be detached.
Change structure only between training steps, not concurrently with training.

## Examples

The examples are notebooks with runnable cells for data preparation, network
construction, training, evaluation, and exploration. Each notebook imports
rendering functions from its neighboring `visualizations.py`; the learning
process stays visible in the cells.

Each example is an independent project with its own dependencies and setup
instructions. Create a separate environment for each, install the local
Neuralplex checkout alongside that project, and launch its notebook:

- [Nibble challenge setup](https://github.com/adamjpatterson/neuralplex/blob/main/examples/the_nibble_challenge/README.md)
- [Digits experiment setup](https://github.com/adamjpatterson/neuralplex/blob/main/examples/the_digits_experiment/README.md)

Run cells from top to bottom. When changing training settings, restart the
kernel and run all cells; rerunning only training continues from the current
weights. Each notebook imports functions from its adjacent `visualizations.py`.

```text
examples/
├── the_nibble_challenge/
│   ├── pyproject.toml
│   ├── README.md
│   ├── main.ipynb
│   └── visualizations.py
└── the_digits_experiment/
    ├── pyproject.toml
    ├── README.md
    ├── main.ipynb
    └── visualizations.py
```

### The nibble challenge

Open [the nibble challenge notebook](https://github.com/adamjpatterson/neuralplex/blob/main/examples/the_nibble_challenge/main.ipynb).
It introduces all sixteen four-bit inputs, constructs a 4 → 8 → 1 network,
compares initial and trained predictions, and lets you change an input to
inspect weights and activations. With seed 42, a step of 0.01, and 10,000
training examples, the displayed R² rounds to 1.000000000.

This example teaches learning on the full nibble domain; it is not a held-out
generalization benchmark or part of the correctness suite.

##### Visualizations of the network before and after training

![Nibble network before and after training](https://raw.githubusercontent.com/adamjpatterson/neuralplex/main/images/nibble_multiweight.png)

- Edge thickness is proportional to the absolute connection weight. Blue is
  positive; orange is negative. Zero-weight edges have zero width.
- Node area above a small visibility floor is proportional to absolute activation.
  Green nodes are inputs, red nodes are hidden, and yellow is the output.
- Both panels use the same input, positions, and scales. Node size represents
  activity, not importance, and does not encode the activation's sign.

The notebook imports `visualize_network` from
[its visualization module](https://github.com/adamjpatterson/neuralplex/blob/main/examples/the_nibble_challenge/visualizations.py).
Figures display inline; the example does not save image files or reports.
The static README illustration is generated separately by the documentation
project described below.

### The digits experiment

Open [the digits experiment notebook](https://github.com/adamjpatterson/neuralplex/blob/main/examples/the_digits_experiment/main.ipynb).
It starts by displaying the [8×8 digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html),
then normalizes pixels, creates a training/test split, constructs a
64 → 16 → 10 network, trains, and evaluates unseen images. You can inspect an
individual prediction and its ten output scores. Scores are not probabilities.

The reproducible settings use seed 42, 1,347 training images, 450 test images,
20 epochs, and a step of 0.001. The example achieves **90.22% test accuracy**,
compared with **10.22%** for always predicting the most common training digit.
This is a learning demonstration, not a claim about the best possible accuracy.
Reserve a validation subset from training data before tuning parameters.

The notebook imports `plot_predictions` and `plot_confusion_matrix` from
[its visualization module](https://github.com/adamjpatterson/neuralplex/blob/main/examples/the_digits_experiment/visualizations.py).
All results and figures display in the notebook, with no separate output files.

#### Digits predictions and confusion matrix

The prediction figure shows the first twelve test images, not selected
successes. The confusion matrix includes all 450 test images; rows show the
true digit and columns show the predicted digit.

![Digits predictions](https://raw.githubusercontent.com/adamjpatterson/neuralplex/main/images/digits_predictions.png)

![Digits confusion matrix](https://raw.githubusercontent.com/adamjpatterson/neuralplex/main/images/digits_confusion_matrix.png)

## Tests

The single formal suite in `tests/test.py` checks forward calculations, manual
gradients, parameter updates, detachment and reconnection, and input validation.
It runs independently of the notebooks and optional visualization dependencies.

#### Run all tests.

```bash
python -m unittest -v
```

## Documentation maintenance

### Regenerating README figures

The [documentation-image project](https://github.com/adamjpatterson/neuralplex/blob/main/docs/README.md) has its own dependencies and
fixed seed-42 scenarios. From `docs/`, after following its environment setup:

```bash
python generate_images.py
```

It generates the nibble network image, digits predictions, and confusion matrix
directly from Python, without importing or running the examples. Only this
workflow writes the README images. The notebooks are for exploration and retain
inline results for reading on GitHub.

## Support

If you have a feature request or run into any issues, feel free to submit an [issue](https://github.com/adamjpatterson/neuralplex/issues). You’re also welcome to reach out to one of the authors.

- [Adam Patterson](https://github.com/adamjpatterson)
