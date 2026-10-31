# README image generation

This independent project generates the repository's three README illustrations.
It declares its own dependencies and runs directly from Python. It does not
import example modules or execute notebooks.

## Setup and run

Use Python 3.11 or later. From this directory in the Neuralplex checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .. .
python generate_images.py
```

On Windows, activate with `.venv\Scripts\activate` instead. Install the local
Neuralplex checkout as shown: a published release may not provide the current
API. The generator resolves the repository's image directory from its own
file location, so its output does not depend on your working directory.

## Fixed scenarios

Every random seed is **42**. The generator owns these settings independently
of any notebook edits:

- Nibbles: a linear 4 → 8 → 1 network, step 0.01, 10,000 training examples,
  and input `1010` for the before/after picture. Positions and visual scales
  are shared between the two panels.
- Digits: a 64 → 16 (tanh) → 10 (linear) network, step 0.001, 20 epochs, and
  a stratified 75%/25% training/test split. Pixels are divided by 16. Only
  training images are passed to `train`.
- Prediction samples: the first twelve test images in split order, not selected
  successes. The confusion matrix uses all 450 test images.

The script prints the nibble R² and digits accuracy, and overwrites:

- `../images/nibble_multiweight.png`
- `../images/digits_predictions.png`
- `../images/digits_confusion_matrix.png`

Review the regenerated images and update any reported README results when
changing these settings. This is documentation tooling, not a package test or
a learning notebook. Its rendering functions are local so this project can be
installed and run without either example project.

With these settings and seed 42, the verified run prints nibble R²
`1.000000000` and digits accuracy `90.22%`.
