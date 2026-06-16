"""
Control experiment: Random 240-vector codebook vs E8 lattice roots.

Tests whether observed behavior (effective rank, gradient variance, sparsity)
is an artifact of discretization + sparsity alone, or whether E8 topological
structure contributes meaningfully.

Design:
  - Generate random codebook with SAME cardinality (240) and norm (sqrt(2))
  - Use FIXED seed for reproducibility across runs
  - Use ORTHOGONAL initialization of the random vectors to avoid accidental clustering
  - Run identical synthetic task on both codebooks with identical input seeding
  - Compare: effective rank, gradient variance, edge density (sparsity), convergence

Interpretation:
  - If curves overlap within 1 std, discretization alone explains the behavior
  - If they diverge significantly, E8 geometry provides a real advantage
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# 1. E8 root system (unchanged from e8_attention.py)
# ---------------------------------------------------------------------------
def build_e8_roots() -> torch.Tensor:
    """Constructs all 240 roots of E8 in R^8."""
    roots = []

    # Family (a): integer roots
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = [0.0] * 8
                    v[i] = si
                    v[j] = sj
                    roots.append(v)
    assert len(roots) == 112

    # Family (b): half-integer roots, even number of minus signs
    for bits in range(256):
        signs = [1.0 if (bits >> k) & 1 == 0 else -1.0 for k in range(8)]
        if signs.count(-1.0) % 2 == 0:
            roots.append([0.5 * s for s in signs])
    assert len(roots) == 240

    R = torch.tensor(roots, dtype=torch.float32)
    norms_sq = (R * R).sum(dim=1)
    assert torch.allclose(norms_sq, torch.full_like(norms_sq, 2.0), atol=1e-5)
    return R


# ---------------------------------------------------------------------------
# 2. Random codebook control: 240 vectors, orthogonal init, norm=sqrt(2)
# ---------------------------------------------------------------------------
def build_random_codebook(seed: int = 42, dim: int = 8, n_vecs: int = 240) -> torch.Tensor:
    """
    Generate random codebook with:
      - Exactly 240 vectors
      - Each with squared norm = 2 (matching E8 roots)
      - Orthogonal-ish initialization (QR decomposition of random matrix)
      - Deterministic (fixed seed)
    
    Strategy:
      1. Generate n_vecs * dim random values from N(0,1)
      2. Reshape to (n_vecs, dim)
      3. Apply QR decomposition → orthogonal basis (handles 240 > 8 via tiling)
      4. Normalize each to have ||v||^2 = 2
    
    This ensures:
      - No accidental clustering from poor initialization
      - Maximum spread in the space
      - Fair comparison: both codebooks have same norm and cardinality
      - Deterministic: same seed → same codebook
    """
    rng = np.random.RandomState(seed)
    
    # Generate random matrix: (240, 8)
    # Use QR trick: stack multiple orthogonal blocks since 240 > 8
    n_blocks = (n_vecs + dim - 1) // dim  # ceil(240 / 8) = 30 blocks
    vecs = []
    
    for block_idx in range(n_blocks):
        # Generate random (8, 8) matrix
        A = rng.randn(dim, dim).astype(np.float32)
        # QR decomposition gives orthogonal Q
        Q, R = np.linalg.qr(A)
        # Take rows of Q (orthonormal vectors)
        vecs.append(Q.T)  # (8, 8)
    
    # Stack and take first 240 vectors
    codebook = np.vstack(vecs)[:n_vecs]  # (240, 8)
    assert codebook.shape == (n_vecs, dim)
    
    # Normalize to have ||v||^2 = 2 (matching E8)
    norms_sq = (codebook ** 2).sum(axis=1, keepdims=True)
    codebook = codebook / np.sqrt(norms_sq) * math.sqrt(2.0)
    
    # Verify
    norms_sq_final = (codebook ** 2).sum(axis=1)
    assert np.allclose(norms_sq_final, 2.0, atol=1e-5), \
        f"Codebook norms not 2.0: {norms_sq_final[:5]}"
    
    return torch.tensor(codebook, dtype=torch.float32)


E8_ROOTS = build_e8_roots()
RANDOM_CODEBOOK = build_random_codebook(seed=42)

print(f"E8 roots shape: {E8_ROOTS.shape}, first 3 norms squared: {(E8_ROOTS[:3]**2).sum(dim=1)}")
print(f"Random codebook shape: {RANDOM_CODEBOOK.shape}, first 3 norms squared: {(RANDOM_CODEBOOK[:3]**2).sum(dim=1)}")


# ---------------------------------------------------------------------------
# 3. Quantization layers: E8 vs Random
# ---------------------------------------------------------------------------
class E8Quantize(torch.autograd.Function):
    """E8 lattice snapping via STE."""
    @staticmethod
    def forward(ctx, x, codebook):
        sims = x @ codebook.T
        idx = sims.argmax(dim=-1)
        snapped = codebook[idx]
        ctx.save_for_backward(idx)
        return snapped

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class RandomQuantize(torch.autograd.Function):
    """Random codebook snapping via STE (identical logic, different codebook)."""
    @staticmethod
    def forward(ctx, x, codebook):
        sims = x @ codebook.T
        idx = sims.argmax(dim=-1)
        snapped = codebook[idx]
        ctx.save_for_backward(idx)
        return snapped

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def e8_snap(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    return E8Quantize.apply(x, codebook)


def random_snap(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    return RandomQuantize.apply(x, codebook)


# ---------------------------------------------------------------------------
# 4. Projection layers
# ---------------------------------------------------------------------------
class E8Projection(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.proj = nn.Linear(d_k, 8, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(2.0)
        snapped = e8_snap(z, E8_ROOTS.to(x.device))
        return snapped


class RandomProjection(nn.Module):
    """Identical to E8Projection but uses random codebook."""
    def __init__(self, d_k: int):
        super().__init__()
        self.proj = nn.Linear(d_k, 8, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(2.0)
        snapped = random_snap(z, RANDOM_CODEBOOK.to(x.device))
        return snapped


# ---------------------------------------------------------------------------
# 5. Attention layers
# ---------------------------------------------------------------------------
class E8Attention(nn.Module):
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
        A = (torch.einsum("...id,...jd->...ij", psi_q, psi_k).round() == 1).float()
        row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        A_norm = A / row_sums
        out = A_norm @ V
        return out, A


class RandomAttention(nn.Module):
    """E8 attention but using random codebook."""
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_model)
        self.random_q = RandomProjection(d_k)
        self.random_k = RandomProjection(d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        psi_q, psi_k = self.random_q(Q), self.random_k(K)
        A = (torch.einsum("...id,...jd->...ij", psi_q, psi_k).round() == 1).float()
        row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        A_norm = A / row_sums
        out = A_norm @ V
        return out, A


class SoftmaxAttention(nn.Module):
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
# 6. Instrumentation
# ---------------------------------------------------------------------------
def effective_rank(matrix: torch.Tensor, eps: float = 1e-7) -> float:
    """Effective rank via singular value entropy."""
    m = matrix.reshape(-1, matrix.shape[-1]).detach()
    if m.shape[0] < 2:
        return float("nan")
    s = torch.linalg.svdvals(m)
    s = s / (s.sum() + eps)
    s = s[s > eps]
    entropy = -(s * s.log()).sum()
    return entropy.exp().item()


def grad_norm_variance(model: nn.Module) -> float:
    """Variance of per-parameter gradient norms."""
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.norm().item())
    if len(norms) < 2:
        return float("nan")
    t = torch.tensor(norms)
    return t.var(unbiased=True).item()


# ---------------------------------------------------------------------------
# 7. Training loop with identical data seeding
# ---------------------------------------------------------------------------
def run_control_experiment(steps: int = 200, batch: int = 16, seq: int = 32,
                            d_model: int = 64, d_k: int = 16, seed: int = 0):
    """
    Run three models in parallel:
      1. Softmax (baseline)
      2. E8 lattice routing
      3. Random 240-vector routing
    
    All three use:
      - Identical synthetic input data (deterministic seed)
      - Identical target (fixed random projection)
      - Identical initialization (per-model seed offset)
    """
    torch.manual_seed(seed)
    
    # Fixed target projection (same for all models)
    target_proj = nn.Linear(d_model, d_model)
    for p in target_proj.parameters():
        p.requires_grad_(False)
    
    # Three models with separate, deterministic initializations
    torch.manual_seed(seed + 100)  # softmax
    softmax_model = SoftmaxAttention(d_model, d_k)
    
    torch.manual_seed(seed + 101)  # E8
    e8_model = E8Attention(d_model, d_k)
    
    torch.manual_seed(seed + 102)  # random
    random_model = RandomAttention(d_model, d_k)
    
    models = {
        "softmax": softmax_model,
        "e8": e8_model,
        "random": random_model,
    }
    optims = {name: torch.optim.Adam(m.parameters(), lr=1e-3) for name, m in models.items()}
    
    log = {name: {"step": [], "erank": [], "grad_var": [], "loss": [], "sparsity": []}
           for name in models}
    
    # CRITICAL: Use master seed for data generation so all models see identical inputs
    torch.manual_seed(seed)
    
    for step in range(steps):
        # Generate input ONCE, reuse for all models (ensures fair comparison)
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
                # Sparsity: fraction of exactly zero weights (binary routing)
                sparsity = (A == 0).float().mean().item()
                log[name]["sparsity"].append(sparsity)
    
    return log


# ---------------------------------------------------------------------------
# 8. Results and analysis
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("CONTROL EXPERIMENT: E8 Lattice vs Random 240-Vector Codebook")
    print("="*80)
    
    results = run_control_experiment(steps=200, seed=0)
    
    print("\n### SOFTMAX BASELINE ###")
    for i, step in enumerate(results["softmax"]["step"]):
        print(f"step {step:4d} | loss {results['softmax']['loss'][i]:.4f} | "
              f"erank {results['softmax']['erank'][i]:6.2f} | "
              f"grad_var {results['softmax']['grad_var'][i]:8.6f} | "
              f"sparsity {results['softmax']['sparsity'][i]:.4f}")
    
    print("\n### E8 LATTICE ###")
    for i, step in enumerate(results["e8"]["step"]):
        print(f"step {step:4d} | loss {results['e8']['loss'][i]:.4f} | "
              f"erank {results['e8']['erank'][i]:6.2f} | "
              f"grad_var {results['e8']['grad_var'][i]:8.6f} | "
              f"sparsity {results['e8']['sparsity'][i]:.4f}")
    
    print("\n### RANDOM 240-VECTOR CODEBOOK (CONTROL) ###")
    for i, step in enumerate(results["random"]["step"]):
        print(f"step {step:4d} | loss {results['random']['loss'][i]:.4f} | "
              f"erank {results['random']['erank'][i]:6.2f} | "
              f"grad_var {results['random']['grad_var'][i]:8.6f} | "
              f"sparsity {results['random']['sparsity'][i]:.4f}")
    
    # Statistical summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (final 5 checkpoints, steps 150-190)")
    print("="*80)
    
    for name in ["softmax", "e8", "random"]:
        losses = results[name]["loss"][-5:]
        eranks = results[name]["erank"][-5:]
        grad_vars = results[name]["grad_var"][-5:]
        sparsities = results[name]["sparsity"][-5:]
        
        print(f"\n{name.upper()}:")
        print(f"  Loss:        {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"  Erank:       {np.mean(eranks):.2f} ± {np.std(eranks):.2f}")
        print(f"  Grad Var:    {np.mean(grad_vars):.6f} ± {np.std(grad_vars):.6f}")
        print(f"  Sparsity:    {np.mean(sparsities):.4f} ± {np.std(sparsities):.4f}")
    
    # Comparison: E8 vs Random
    print("\n" + "="*80)
    print("CONTROL INTERPRETATION")
    print("="*80)
    
    e8_erank = np.mean(results["e8"]["erank"][-5:])
    random_erank = np.mean(results["random"]["erank"][-5:])
    e8_sparsity = np.mean(results["e8"]["sparsity"][-5:])
    random_sparsity = np.mean(results["random"]["sparsity"][-5:])
    
    print(f"\nEffective Rank Gap (E8 vs Random): {e8_erank - random_erank:.2f}")
    print(f"Sparsity Gap (E8 vs Random): {e8_sparsity - random_sparsity:.4f}")
    
    if abs(e8_erank - random_erank) < 0.5:
        print("\n✓ E8 and Random have SIMILAR effective ranks → discretization dominates")
    else:
        print("\n✗ E8 and Random DIVERGE in effective rank → E8 structure matters")
    
    if abs(e8_sparsity - random_sparsity) < 0.02:
        print("✓ E8 and Random have SIMILAR sparsity → codebook size (240) dominates")
    else:
        print("✗ E8 and Random DIVERGE in sparsity → E8 adjacency structure matters")
    
    print("\n" + "="*80)
