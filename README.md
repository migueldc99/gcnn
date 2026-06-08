# Graph Convolutional Neural Networks (GCNN)

Property prediction for small molecules using Graph Convolutional Neural Networks.

## Project Structure

```
gcnn/
├── config/             # Experiment configuration (YAML)
├── data/               # Molecular datasets (not versioned)
├── notebooks/          # Jupyter notebooks (data prep → training → analysis)
├── outputs/            # Training runs, logs, predictions (not versioned)
└── src/gcnn/           # Installable Python package
    ├── graphs/         # Molecular graph construction (ABC + implementations)
    ├── training/       # Training loop, callbacks, transforms
    ├── features.py     # Gaussian basis feature engineering
    ├── model.py        # GCNN architecture (CGConv layers)
    ├── dataset.py      # PyTorch Geometric dataset class
    ├── paths.py        # Centralized path resolution
    └── utils.py        # Utilities (parameter counting, etc.)
```

## Installation

```bash
# Clone and install in editable mode
git clone <repo-url>
cd gcnn
pip install -e .

# For development (adds matplotlib, jupyter, pandas, etc.)
pip install -e ".[dev]"
```

## Usage

1. **Prepare data**: Download the QM9 dataset (see `data/README.md`), then run `notebooks/1_gen_database.ipynb` to split into train/validation/test.
2. **Train model**: Run `notebooks/2_train_model.ipynb` (configuration via `config/sample.yml`).
3. **Analyze results**: Run `notebooks/3_graphics.ipynb` to visualize training curves and predictions.

## Dependencies

Managed via `pyproject.toml`. Core requirements:
- PyTorch, PyTorch Geometric (+ torch-scatter, torch-sparse)
- NumPy, SciPy, scikit-learn
- Mendeleev (periodic table data)
- PyYAML, TensorBoard

## Authors

Miguel Dalmau Casañal (migueldcj6@gmail.com)

## License

MIT License (see LICENSE file)
