"""
Shared backbone network for Preference Distribution Learning (PDL).

This module defines a four-layer fully connected network (with Softmax output)
that is used as both:
  - the Proxy model F(lambda|omega), which predicts the direction of the
    normalized loss vector for a given preference;
  - the Generator model G(lambda|phi), which transforms uniformly sampled
    preferences into adapted ones.

The Softmax output guarantees that the generated preference lies on the
probability simplex (each component >= 0 and components sum to 1).
"""

import torch.nn as nn


class MLP(nn.Module):
    """Four-layer MLP shared by the Proxy and Generator models in PDL."""

    def __init__(self, dim):
        """
        Args:
            dim (int): dimensionality of the preference vector
                       (i.e., the number of objectives in the MOO problem).
        """
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 32),
            nn.LeakyReLU(),
            nn.Linear(dim * 32, dim * 32),
            nn.LeakyReLU(),
            nn.Linear(dim * 32, dim * 16),
            nn.LeakyReLU(),
            nn.Linear(dim * 16, dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.mlp(x)