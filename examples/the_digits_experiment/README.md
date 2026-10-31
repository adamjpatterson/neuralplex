# The digits experiment

The notebook prepares data, constructs and trains the network, evaluates it,
and displays visualizations imported from the adjacent `visualizations.py`.

## Setup

Use Python 3.11 or later. Starting from this directory in the Neuralplex checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ../.. .
python -m jupyter lab main.ipynb
```

On Windows, activate with `.venv\Scripts\activate` instead. The combined install
command supplies the current local Neuralplex checkout for the `neuralplex`
dependency declared in this project's `pyproject.toml`.

## Explore

Select this environment's Python kernel and run the cells from top to bottom.
The kernel working directory should be this example directory, so
`from visualizations import ...` resolves to its neighboring module. If you
already have another Jupyter environment open, launch the command above from
this project's activated environment.

The default seed is **42**. Restart the kernel and run all cells when changing
training settings; re-running only the training cell continues from its current
weights. Change the selected input after training to inspect another prediction.
