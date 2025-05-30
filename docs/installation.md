# Installation

Knightmare requires Python 3.10 or newer.

```bash
git clone https://github.com/CarloTUM/Knightmare.git
cd Knightmare
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## GPU support

The `pip install` above installs the CPU PyTorch wheel. For CUDA
acceleration follow the official PyTorch guide and install the wheel
matching your CUDA toolkit, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Verifying the install

```bash
knightmare info
```

The command prints the package version, the detected PyTorch version,
whether CUDA is available, and the resolved data and models directories.
