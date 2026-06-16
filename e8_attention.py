"""
Toy experiment: E8-lattice quantized attention vs standard softmax attention.

Measures, per training step:
  - representation rank (effective rank of attention output, via singular values)
  - gradient variance (variance of grad norms across a batch of micro-batches)

NOT RUN — no torch / no network in this sandbox. Written for correctness,
not verified by execution. Run locally and sanity-check before trusting numbers.

Honest framing: this is a toy on random/synthetic data, meant only to surface
whether STE gradient noise and routing-collapse are visible at small scale.
It is not evidence for or against the architecture at production scale.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. E8 root system: explicit 240-vector codebook
# ---------------------------------------------------------------------------
def build_e8_roots() -> torch.Tensor:
    """
    Constructs all 240 roots of E8 in R^8.

    E8 roots come in two families:
      (a) All vectors with two entries =/- 1, rest 0: C(8,2)*4 = 28*4 = 112 roots.
          (i.e. permutations of (+-1, +-1, 0,0,0,0,0,0))
      (b) All vectors (+-1/2)^8 with an EVEN number of minus signs: 2^7 = 128 roots.

    112 + 128 = 240. Norm^2 = 2 for every root in both families, matching the
    standard E8 normalization.
    """
    roots = []

    # Family (a): integer roots, two nonzero entries of +-1
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = [0.0] * 8
                    v[i] = si
                    v[j] = sj
                    roots.append(v)
    assert len(roots) == 112

    # Family (b): half-integer roots, all entries +-1/2, even number of minus signs
    for bits in range(256):
        signs = [1.0 if (bits >> k) & 1 == 0 else -1.0 for k in range(8)]
        if signs.count(-1.0) % 2 == 0:
            roots.append([0.5 * s for s in signs])
    assert len(roots) == 240

    R = torch.tensor(roots, dtype=torch.float32)
    norms_sq = (R * R).sum(dim=1)
    assert torch.allclose(norms_sq, torch.full_like(norms_sq, 2.0), atol=1e-5), \
        "All E8 roots must have squared norm 2"
    return R  # shape: (240, 8)


E8_ROOTS = build_e8_roots()  # (240, 8), fixed, not trained


# ---------------------------------------------------------------------------
# 2. Learned projection R^d_k -> R^8, then VQ snap to nearest E8 root (STE)
# ---------------------------------------------------------------------------
class E8Quantize(torch.autograd.Function):
    """
    Forward: snap each 8-dim vector to its nearest E8 root (hard, non-differentiable).
    Backward: straight-through -- pass the incoming gradient through unchanged,
    as if the quantization step were the identity function.

    This is the standard VQ-VAE-style STE (van den Oord et al. 2017), applied
    here to the E8 codebook instead of a learned codebook.
    """

    @staticmethod
    def forward(ctx, x, codebook):
        # x: (..., 8), codebook: (240, 8)
        # nearest neighbor by Euclidean distance == nearest by max inner product
        # since all codebook vectors have equal norm.
        sims = x @ codebook.T                      # (..., 240)
        idx = sims.argmax(dim=-1)                   # (...,)
        snapped = codebook[idx]                     # (..., 8)
        ctx.save_for_backward(idx)
        return snapped

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: identity gradient w.r.t. x, no gradient w.r.t. codebook
        # (codebook is fixed/non-trainable here, so second return value is unused).
        return grad_output, None


def e8_snap(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    return E8Quantize.apply(x, codebook)


class E8Projection(nn.Module):
    """Learned linear map d_k -> 8, followed by E8 lattice snapping."""

    def __init__(self, d_k: int):
        super().__init__()
        self.proj = nn.Linear(d_k, 8, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)                      # (..., 8), continuous
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(2.0)  # match root norm^2=2
        snapped = e8_snap(z, E8_ROOTS.to(x.device))
        return snapped


# ---------------------------------------------------------------------------
# 3. Adjacency Kronecker Tensor: binary routing from snapped roots
# ---------------------------------------------------------------------------
def e8_adjacency(psi_q: torch.Tensor, psi_k: torch.Tensor) -> torch.Tensor:
    """
    psi_q, psi_k: (..., seq, 8), already snapped to E8 roots.
    Returns A_ij in {0, 1}: 1 iff roots share a canonical edge
    (inner product == 1, the standard E8 root-graph adjacency condition).
    """
    inner = torch.einsum("...id,...jd->...ij", psi_q, psi_k)
    # exact equality is valid here since both are snapped lattice points,
    # so inner products land on the fixed set {-2,-1,0,1,2}.
    A = (inner.round() == 1).float()
    return A


class E8Attention(nn.Module):
    """Discrete E8-routed attention: binary adjacency replaces softmax."""

    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_model)
        self.e8_q = E8Projection(d_k)
        self.e8_k = E8Projection(d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        psi_q, psi_k = self.e8_q(Q), self.e8_k(K)
        A = e8_adjacency(psi_q, psi_k)                  # (batch, seq, seq), 0/1
        # normalize per-row so rows with zero edges don't silently zero out output;
        # this is a practical necessity the pure formalism glosses over.
        row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        A_norm = A / row_sums
        out = A_norm @ V
        return out, A


class SoftmaxAttention(nn.Module):
    """Standard baseline for comparison."""

    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_model)
        self.d_k = d_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        A = F.softmax(scores, dim=-1)
        out = A @ V
        return out, A


# ---------------------------------------------------------------------------
# 4. Instrumentation: effective rank + gradient variance
# ---------------------------------------------------------------------------
def effective_rank(matrix: torch.Tensor, eps: float = 1e-7) -> float:
    """
    Effective rank via normalized singular value entropy (Roy & Vetterli 2007):
    erank = exp(H(p)), where p_i = sigma_i / sum(sigma), H = entropy.
    Lower erank = more representation collapse.
    """
    m = matrix.reshape(-1, matrix.shape[-1]).detach()
    if m.shape[0] < 2:
        return float("nan")
    s = torch.linalg.svdvals(m)
    s = s / (s.sum() + eps)
    s = s[s > eps]
    entropy = -(s * s.log()).sum()
    return entropy.exp().item()


def grad_norm_variance(model: nn.Module) -> float:
    """Variance of per-parameter gradient norms -- a crude proxy for
    how uneven/unstable the gradient signal is across the network."""
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.norm().item())
    if len(norms) < 2:
        return float("nan")
    t = torch.tensor(norms)
    return t.var(unbiased=True).item()


# ---------------------------------------------------------------------------
# 5. Minimal training loop on synthetic data
# ---------------------------------------------------------------------------
def run_experiment(steps: int = 200, batch: int = 16, seq: int = 32,
                    d_model: int = 64, d_k: int = 16, seed: int = 0):
    torch.manual_seed(seed)

    # Synthetic task: predict a random linear readout of the input sequence.
    # Not meant to be a meaningful task -- just enough signal to produce
    # nontrivial gradients for comparison.
    target_proj = nn.Linear(d_model, d_model)
    for p in target_proj.parameters():
        p.requires_grad_(False)

    models = {
        "softmax": SoftmaxAttention(d_model, d_k),
        "e8": E8Attention(d_model, d_k),
    }
    optims = {name: torch.optim.Adam(m.parameters(), lr=1e-3) for name, m in models.items()}

    log = {name: {"step": [], "erank": [], "grad_var": [], "loss": [], "edge_density": []}
           for name in models}

    for step in range(steps):
        x = torch.randn(batch, seq, d_model)
        with torch.no_grad():
            y_target = target_proj(x)

        for name, model in models.items():
            optims[name].zero_grad()
            out, A = model(x)
            loss = F.mse_loss(out, y_target)
            loss.backward()
            optims[name].step()

            if step % 10 == 0:
                log[name]["step"].append(step)
                log[name]["erank"].append(effective_rank(out))
                log[name]["grad_var"].append(grad_norm_variance(model))
                log[name]["loss"].append(loss.item())
                if name == "e8":
                    log[name]["edge_density"].append(A.mean().item())

    return log


if __name__ == "__main__":
    results = run_experiment()
    for name, data in results.items():
        print(f"\n=== {name} ===")
        for i, step in enumerate(data["step"]):
            line = f"step {step:4d} | loss {data['loss'][i]:.4f} | erank {data['erank'][i]:.2f} | grad_var {data['grad_var'][i]:.6f}"
            if name == "e8":
                line += f" | edge_density {data['edge_density'][i]:.4f}"
            print(line)
