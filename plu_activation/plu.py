import torch
import torch.nn as nn

class PeriodicLinearUnitExponential(nn.Module):
    def __init__(self, shape=(1,), init_alpha=0.0, init_beta=0.0):
        """
        An alternative formulation of the Periodic Linear Unit using e^x reparameterization in place of the original RR formulation.

        We start at a neutral point, where α=0.0 (α_eff=1.0) and β=0.0 (β_eff=1.0).
        As either α or β gets pushed above 0.0, the sine-wave begins oscillating faster and contributes more to the sum respectively.
        Since changing α to be higher, and thus increasing the oscillation, has a larger effect 
        on the final output and loss when compared to a change at the lower end, a "larger leap of faith" 
        is sometimes necessary to cross any hills in the loss plane to reach a better minimum.
        Since e^x has a higher and higher derivative above 0.0, this leap of faith is mathematically provided, as a small dα leads to a larger and larger dα_eff.
        As either α or β gets pushed below 0.0, the sine-wave begins to oscillate slower and slower and contributes less to the sum respectively.
        Since both of those lead to a collapse to linearity, we want to disincentivize the optimizer from taking a step in that direction more and more as α veers 
        closer to negative infinity.
        Since e^x has a lower and lower derivative below 0.0, each further dα step taken in that direction leads to a smaller and smaller dα_eff, thereby making each push 
        towards zero have less impact on the final loss, achieving our desired outcome.
        By using e^x reparameterization, we lose control over a specific lower bound for frequency and amplitude on the sine-wave component, but the benefit of 
        being able to have the model find a pretty good local minimum on its own without hyperparameter searching is well worth the trade-off in a variety of use-cases. 
        This is especially useful when using a separate learnable PLU activation for each hidden neuron, as hyperparameter searching is prohibitively expensive in that situation.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.full(shape, init_alpha))
        self.beta = nn.Parameter(torch.full(shape, init_beta))
    
    def forward(self, x):
        # exponential reparameterization
        alpha_eff = torch.exp(self.alpha)
        beta_eff = torch.exp(self.beta)
        return x + beta_eff * torch.sin(alpha_eff * x)

class PeriodicLinearUnit(nn.Module):
    def __init__(self, shape=(1,), init_alpha=1.0, init_beta=1.0, init_rho_alpha=5.0, init_rho_beta=0.15):
        """
        A **Periodic Linear Unit (PLU)** is composed of a scaled linear sum of the sine function and x with α and β reparameterization, as described in the paper "From Taylor Series to Fourier Synthesis: The Periodic Linear Unit".

        Formula: PLU(x, α, ρ_α, β, ρ_β) = x + (β_eff / (1 + |β_eff|)) * sin(|α_eff| * x)
        
        Where the effective parameters, α_eff and β_eff, are reparameterized using learnable repulsion terms, ρ_α and ρ_β as follows:

        α_eff = α + ρ_α / α
        β_eff = β + ρ_β / β

        The `x` component serves as a residual path. By including it, the function is identity-like at its core, 
        allowing gradients can flow through it easily just like a residual connection. It prevents the vanishing gradient problem and makes training much more stable.

        The `sin(x)` component then provides periodic non-linearity.
        
        Args:
            init_alpha (float):
                Period multiplier
            init_beta (float): 
                Signed, soft-bounded strength of periodic non-linearity
            init_rho_alpha (float):
                Repulsion term for keeping α away from 0.0, keeping α_eff >= sqrt(ρ_α)
            init_rho_beta (float):
                Repulsion term for keeping β away from 0.0, keeping β_eff >= sqrt(ρ_β)
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.full(shape, init_alpha))
        self.beta = nn.Parameter(torch.full(shape, init_beta))
        self.rho_alpha = nn.Parameter(torch.full(shape, init_rho_alpha))
        self.rho_beta = nn.Parameter(torch.full(shape, init_rho_beta))
    
    def forward(self, x):
        # repulsive reparameterization
        alpha_eff = self.alpha + self.rho_alpha / self.alpha
        beta_eff = self.beta + self.rho_beta / self.beta
        return x + (beta_eff / (1.0 + torch.abs(beta_eff))) * torch.sin(torch.abs(alpha_eff) * x)

