### From Taylor Series to Fourier Synthesis: The Periodic Linear Unit - Code Repository

**arXiv:** https://arxiv.org/abs/2508.01175

This repository contains the PyTorch implementation of the **Periodic Linear Unit (PLU)** activation function, as described in the paper "From Taylor Series to Fourier Synthesis: The Periodic Linear Unit".

PLU represents a novel approach to neural network activations, moving beyond traditional piecewise linear approximations (like ReLU) towards a Fourier-like synthesis using powerful, learnable wave-like basis functions. This enables significantly higher parameter efficiency, allowing minimal networks (e.g., two neurons) to solve complex non-linear tasks like spiral classification.

### Alternate Formulations

In addition to the formulation described within the main paper, an alternative formulation of the Periodic Linear Unit is also included in this repository using e^x reparameterization in place of the original RR formulation.

The alternate formulation utilizes:
```
α_eff = exp(α)
β_eff = exp(β)
```

Initial values for α and β are set to be α=0.0 (α_eff=1.0) and β=0.0 (β_eff=1.0) respectively. We start this neutral point.

As either `α` or `β` gets pushed above `0.0`, the sine-wave begins oscillating faster and contributes more to the sum respectively.<br>
Since changing `α` to be higher, and thus increasing the oscillation, has a larger effect 
on the final output and loss when compared to a change at the lower end, a "larger leap of faith" is sometimes necessary to cross any hills in the loss plane to reach a better minimum.<br>
Since `e^x` has a higher and higher derivative above `0.0`, this leap of faith is mathematically provided, as a small `dα` leads to a larger and larger `dα_eff`.

As either `α` or `β` gets pushed below `0.0`, the sine-wave begins to oscillate slower and slower and contributes less to the sum respectively.<br>
Since both of those lead to a collapse to linearity, we want to disincentivize the optimizer from taking a step in that direction more and more as α veers closer to negative infinity.<br>
Since `e^x` has a lower and lower derivative below `0.0`, each further `dα` step taken in that direction leads to a smaller and smaller `dα_eff`, thereby making each push towards zero have less impact on the final loss, achieving our desired outcome.

By using `e^x` reparameterization, we lose control over a specific lower bound for frequency and amplitude on the sine-wave component, but the benefit of being able to have the model find a pretty good local minimum on its own without hyperparameter searching is well worth the trade-off in a variety of use-cases. This is especially useful when using a separate learnable PLU activation for each hidden neuron, as hyperparameter searching is prohibitively expensive in that situation.

This version of the formulation with the alternate reparameterization is quite effective on a variety of models.

In tests on a variety of models, it yields twice the final loss/performance (both in convergence speed gains, as well in the final loss reduction compared to baseline standard-issue activations) as compared to the original formulation using the parameters specified in the code provided.

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

