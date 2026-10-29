import random
import unittest

from neuralplex import Layer, Network, Neuron


class TestNibbleChallenge(unittest.TestCase):
    def test_nibbles(self):
        """Learn decimal values from four bits, then check all 16 nibbles."""
        rng = random.Random(13)
        network = Network([
            Layer([Neuron() for _ in range(size)], step=0.01)
            for size in [4, 8, 1]
        ], seed=13)
        examples = [
            ([int(bit) for bit in f'{number:04b}'], number)
            for number in range(16)
        ]
        before = [network.predict(bits)[0] for bits, number in examples]

        print("\nNibble Challenge: training a 4 → 8 → 1 network.")
        for iteration in range(10000):
            bits, number = rng.choice(examples)
            network.train(bits, [number])
            if (iteration + 1) % 1000 == 0:
                print(f'Training iteration: {iteration + 1}')

        predictions = [network.predict(bits)[0] for bits, number in examples]
        print('Nibble  Truth     Before      After')
        for number, (initial, prediction) in enumerate(zip(before, predictions)):
            print(f'{number:04b}    {number:2d}    {initial:9.4f}  {prediction:9.4f}')

        initial_error = sum((p-y)**2 for y, p in enumerate(before))
        final_error = sum((p-y)**2 for y, p in enumerate(predictions))
        r2 = 1 - final_error / sum((y-7.5)**2 for y in range(16))
        print(f'Mean squared error: {initial_error / 16:.6f} → {final_error / 16:.6f}')
        print(f'R2: {r2:.9f}')

        self.assertLess(final_error, initial_error)
        self.assertGreaterEqual(r2, .99)
        for number, prediction in enumerate(predictions):
            with self.subTest(nibble=f'{number:04b}'):
                self.assertAlmostEqual(prediction, number, delta=0.05)


if __name__ == '__main__':
    unittest.main()
