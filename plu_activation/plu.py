import torch
import torch.nn as nn

class PeriodicLinearUnitExponential(nn.Module):
    """
    An alternative formulation of the Periodic Linear Unit using e^x reparameterization instead of the original RR formulation.
    """
    def __init__(self, shape=(1,), init_alpha=0.0, init_beta=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full(shape, init_alpha))
        self.beta = nn.Parameter(torch.full(shape, init_beta))
    
    def forward(self, x):
        # exponential reparameterization
        alpha_eff = torch.exp(self.alpha)
        beta_eff = torch.exp(self.beta)
        return x + beta_eff * torch.sin(torch.abs(alpha_eff) * x)

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

