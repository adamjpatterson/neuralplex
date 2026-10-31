# The digits experiment

An independent example project with its own Python environment and dependencies.
The notebook prepares data, constructs and trains the network, evaluates it,
and displays visualizations imported from the adjacent `visualizations.py`.
It does not export reports or images; all results appear inside the notebook.

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
dependency declared in this project's `pyproject.toml`. Use this checkout:
an older PyPI release may not provide the API used here. There is no dependency
on the other example or the documentation-image project.

## Explore

Select this environment's Python kernel and run the cells from top to bottom.
The kernel working directory should be this example directory, so
`from visualizations import ...` resolves to its neighboring module. If you
already have another Jupyter environment open, launch the command above from
this project's activated environment.

The default seed is **42**. Restart the kernel and run all cells when changing
training settings; re-running only the training cell continues from its current
weights. Change the selected input after training to inspect another prediction.
Rendered results are retained in `main.ipynb` for reading on GitHub. Saving the
notebook is the only persistence needed for this example.

README illustrations are maintained separately by the
[documentation-image project](../../docs/README.md).
