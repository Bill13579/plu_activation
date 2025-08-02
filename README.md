## From Taylor Series to Fourier Synthesis: The Periodic Linear Unit - Code Repository

This repository contains the PyTorch implementation of the **Periodic Linear Unit (PLU)** activation function, as described in the paper "From Taylor Series to Fourier Synthesis: The Periodic Linear Unit".

PLU represents a novel approach to neural network activations, moving beyond traditional piecewise linear approximations (like ReLU) towards a Fourier-like synthesis using powerful, learnable wave-like basis functions. This enables significantly higher parameter efficiency, allowing minimal networks (e.g., two neurons) to solve complex non-linear tasks like spiral classification.

## Installation

The repository is structured as a standard `pyproject.toml` project.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bill13579/plu_activation
    cd plu_activation
    ```

2.  **Install the package in editable mode:**
    This will install the `plu_activation` package and all its dependencies (including `torch`, `numpy`, `matplotlib`) as specified in `pyproject.toml`.
    ```bash
    pip install -e .
    ```

## Examples

The `spiral_plu_example.py` script demonstrates the classic spiral classification task, reproducing the experiments and visualizations from the paper.

To run the example:

```bash
python spiral_plu_example.py
```

Additionally, the video files and graphs from the paper are accessible under the "Examples" directory.

