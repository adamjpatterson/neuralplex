"""Public API correctness, including numerical gradient checks."""

import unittest

from neuralplex import Layer, Network, Neuron, get_edge_data


class TestCorrectness(unittest.TestCase):
    def test_forward_and_update(self):
        model = Network([Layer([Neuron(), Neuron()]), Layer([Neuron()], step=0.1)], seed=42)
        a, b = model.layers[0].neurons
        output = model.layers[1].neurons[0]
        output.weights = {a: 2.0, b: -1.0}
        output.bias = 3.0
        self.assertEqual(model.predict([4, 5]), [6])
        model.train([4, 5], [5])
        self.assertAlmostEqual(output.weights[a], 1.6)
        self.assertAlmostEqual(output.weights[b], -1.5)
        self.assertAlmostEqual(output.bias, 2.9)

    def test_gradients_and_reset(self):
        model = Network([Layer([Neuron(), Neuron()]), Layer([Neuron(activation='tanh') for _ in range(3)], step=0.001), Layer([Neuron(), Neuron()], step=0.001)], seed=42)
        model.input_layer.neurons[0].connectRHS(model.output_layer.neurons[0], weight=0.2)
        for inputs, targets in [([0.4, -0.7], [0.2, -0.3]), ([-0.1, 0.5], [0.6, 0.1])]:
            expected = []
            for layer in model.layers[1:]:
                for neuron in layer.neurons:
                    for source in [*neuron.weights, None]:
                        original = neuron.bias if source is None else neuron.weights[source]
                        losses = []
                        for offset in [1e-6, -1e-6]:
                            if source is None:
                                neuron.bias = original + offset
                            else:
                                neuron.weights[source] = original + offset
                            losses.append(sum((p-y)**2 / 2 for p, y in zip(model.predict(inputs), targets)))
                        if source is None:
                            neuron.bias = original
                        else:
                            neuron.weights[source] = original
                        expected.append((neuron, source, original, (losses[0]-losses[1])/2e-6))
            model.train(inputs, targets)
            for neuron, source, original, gradient in expected:
                updated = neuron.bias if source is None else neuron.weights[source]
                self.assertAlmostEqual((original-updated)/neuron.step, gradient, places=7)

    def test_perfect_prediction_and_dimensions(self):
        model = Network([Layer([Neuron(), Neuron()]), Layer([Neuron(activation='tanh') for _ in range(3)], step=0.001), Layer([Neuron(), Neuron()], step=0.001)], seed=42)
        targets = model.predict([0.2, 0.4])
        before = [(dict(n.weights), n.bias) for layer in model.layers for n in layer.neurons]
        model.train([0.2, 0.4], targets)
        self.assertEqual(before, [(dict(n.weights), n.bias) for layer in model.layers for n in layer.neurons])
        with self.assertRaises(ValueError):
            model.predict([1])
        with self.assertRaises(ValueError):
            model.train([1, 2], [1])


    def test_detach_and_reattach(self):
        inputs = [Neuron(name=f'i{i}') for i in range(3)]
        hidden, output = Neuron(name='h'), Neuron(name='o')
        model = Network([Layer(inputs), Layer([hidden], step=0.01), Layer([output], step=0.01)], seed=42)
        incoming = dict(hidden.weights)
        outgoing = output.weights[hidden]
        before = model.predict([1, 2, 3])
        model.detach(hidden)
        self.assertEqual(hidden.weights, {})
        self.assertEqual(hidden.neuronsRHS, [])
        self.assertTrue(all(not n.neuronsRHS for n in inputs))
        self.assertEqual(model.predict([1, 2, 3]), [0])
        model.layers[1].neurons.append(hidden)
        for source, weight in incoming.items():
            source.connectRHS(hidden, weight)
        hidden.connectRHS(output, outgoing)
        hidden.connectRHS(output, 999)  # Existing edges keep their weights.
        self.assertEqual(model.predict([1, 2, 3]), before)
        self.assertIn({'source': 'h', 'target': 'o', 'weight': outgoing}, get_edge_data(model))
        self.assertEqual(len(get_edge_data(model)), 4)
        inputs[0].disconnectRHS(hidden)
        inputs[0].disconnectRHS(hidden)
        model.train([1, 2, 3], [1])
        with self.assertRaises(ValueError):
            model.detach(output)
        output.connectRHS(inputs[0])
        with self.assertRaises(ValueError):
            model.predict([1, 2, 3])

    def test_missing_step_and_external_connections(self):
        a, b, outside = Neuron(), Neuron(), Neuron()
        model = Network([Layer([a]), Layer([b])])
        with self.assertRaises(ValueError):
            model.train([1], [1])
        a.connectRHS(outside)
        with self.assertRaises(ValueError):
            model.predict([1])


if __name__ == '__main__':
    unittest.main()
