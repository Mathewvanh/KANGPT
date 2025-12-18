"""
KAN Layer - Kolmogorov-Arnold Network Layer using Chebyshev polynomials.

This implements learnable polynomial transformations as an alternative to linear layers.
Each output is a weighted sum of Chebyshev polynomial basis functions applied to inputs.
"""

import torch
import torch.nn as nn


class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network Layer using Chebyshev polynomial basis.

    Computes: output_j = sum_i sum_k C[j,i,k] * T_k(x_i)

    where T_k is the Chebyshev polynomial of the first kind of degree k.

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        degree: Maximum polynomial degree (uses degrees 0 to degree)
    """

    def __init__(self, in_dim: int, out_dim: int, degree: int = 5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        self.num_degrees = degree + 1  # 0 to degree inclusive

        # Learnable coefficients: (out_dim, in_dim, num_degrees)
        self.C = self._initialize_coefficients()

    def _initialize_coefficients(self) -> nn.Parameter:
        """Initialize coefficients with Xavier uniform and degree-based scaling."""
        param = nn.Parameter(torch.empty(self.out_dim, self.in_dim, self.num_degrees))
        nn.init.xavier_uniform_(param)

        # Scale by polynomial degree for stability
        if self.degree > 3:
            scale = 1.0 / (self.degree ** 0.5)
            param.data *= scale

        return param

    def chebyshev(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Chebyshev polynomials T_0(x) to T_degree(x).

        Uses the recurrence relation: T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
        with T_0(x) = 1, T_1(x) = x

        Args:
            x: Input tensor of shape (..., in_dim), should be in [-1, 1]

        Returns:
            Tensor of shape (..., in_dim, num_degrees) containing polynomial values
        """
        T_all = []

        # T_0(x) = 1
        T0 = torch.ones_like(x)
        T_all.append(T0)

        if self.degree >= 1:
            # T_1(x) = x
            T1 = x
            T_all.append(T1)

        # Recurrence: T_n = 2x * T_{n-1} - T_{n-2}
        T_prev2, T_prev1 = T0, T1 if self.degree >= 1 else T0
        for n in range(2, self.degree + 1):
            T_n = 2 * x * T_prev1 - T_prev2
            T_all.append(T_n)
            T_prev2, T_prev1 = T_prev1, T_n

        # Stack: shape (..., in_dim, num_degrees)
        return torch.stack(T_all, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_dim) or (in_dim,)

        Returns:
            Output tensor of shape (batch, out_dim) or (out_dim,)
        """
        # Compute polynomial basis
        basis = self.chebyshev(x)  # (..., in_dim, num_degrees)

        if x.dim() == 2:
            # Batch mode: (batch, in_dim) -> (batch, out_dim)
            # basis: (batch, in_dim, num_degrees)
            # C: (out_dim, in_dim, num_degrees)
            out = torch.einsum('bid,oid->bo', basis, self.C)
        else:
            # Single sample: (in_dim,) -> (out_dim,)
            out = torch.einsum('id,oid->o', basis, self.C)

        return out
