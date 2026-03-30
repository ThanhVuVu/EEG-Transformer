import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class FullAttnRes(nn.Module):
    """Full Attention Residuals.

    Each layer l forms its input h_l by attending over all preceding outputs
    v_0..v_{l-1} using a learned pseudo-query vector w_l. Sources are RMSNorm'd
    before scoring.

    Inputs:
        embedding: [B, T, D]
        layer_fns: list length L, each maps [B, T, D] -> [B, T, D]

    Outputs:
        output: [B, T, D] final layer output v_L
        weight_matrix: [L, L+1] average source weights for visualization
    """

    def __init__(self, num_layers: int, d_model: int):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        # One pseudo-query per layer, initialized to zero.
        self.w = nn.Parameter(torch.zeros(num_layers, d_model))
        self.norm = RMSNorm(d_model)

    def compute_weights(self, layer_idx: int, sources: torch.Tensor) -> torch.Tensor:
        """Args:
            sources: [num_sources, B, T, D]
        Returns:
            weights: [num_sources, B, T] (softmax over sources)
        """
        K = self.norm(sources)
        q = self.w[layer_idx]  # [D]
        logits = torch.einsum("d, n b t d -> n b t", q, K)
        return F.softmax(logits, dim=0)

    def forward(self, embedding: torch.Tensor, layer_fns: list[nn.Module]):
        L = len(layer_fns)
        assert L == self.num_layers, f"Expected {self.num_layers} layers, got {L}"

        layer_outputs = [embedding]  # v_0
        weight_matrix = embedding.new_zeros((L, L + 1))

        for l in range(L):
            sources = torch.stack(layer_outputs)  # [l+1, B, T, D]
            weights = self.compute_weights(l, sources)  # [l+1, B, T]

            h_l = torch.einsum("n b t, n b t d -> b t d", weights, sources)

            avg_w = weights.mean(dim=(1, 2)).detach()
            weight_matrix[l, : l + 1] = avg_w

            v_l = layer_fns[l](h_l)
            layer_outputs.append(v_l)

        return layer_outputs[-1], weight_matrix

