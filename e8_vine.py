"""
E8 Lattice + Vine Modulation: Topological Gates + Dynamic Flux Weighting

Architecture:
  - E8 lattice: Fixed binary adjacency δ(⟨Ψ(Q), Ψ(K)⟩, 1) ∈ {0, 1}
  - Vine: Learned function Vine(Q, K) that scales signal magnitude through open gates
  - Attention: A_ij = Vine(Q_i, K_j) ⊙ δ(⟨Ψ(Q_i), Ψ(K_j)⟩, 1)

Key insight:
  - The lattice is rigid: it dictates which pathways CAN be used
  - The Vine is plastic: it learns how much signal flows through each allowed pathway
  - This breaks the null result: now E8's adjacency structure directly shapes
    the optimization landscape by constraining where gradients can flow

Control still applies:
  - Random 240-vector lattice (same topology constraint, different structure)
  - Softmax (full dense attention, no topological constraint)
  
Expected outcome:
  - E8 + Vine should outperform (Random + Vine) if adjacency structure matters
  - Vine should learn to use sparse routing more effectively than dense softmax
  - Edge density should be meaningful (non-trivial fraction of ~1% usable edges)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# 1. E8 root system
# ---------------------------------------------------------------------------
def build_e8_roots() -> torch.Tensor:
    """Constructs all 240 roots of E8 in R^8."""
    roots = []

    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = [0.0] * 8
                    v[i] = si
                    v[j] = sj
                    roots.append(v)
    assert len(roots) == 112

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
# 2. Random codebook (for control)
# ---------------------------------------------------------------------------
def build_random_codebook(seed: int = 42, dim: int = 8, n_vecs: int = 240) -> torch.Tensor:
    """Generate random 240-vector orthogonal codebook with ||v||^2 = 2."""
    rng = np.random.RandomState(seed)
    n_blocks = (n_vecs + dim - 1) // dim
    vecs = []
    
    for _ in range(n_blocks):
        A = rng.randn(dim, dim).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        vecs.append(Q.T)
    
    codebook = np.vstack(vecs)[:n_vecs]
    norms_sq = (codebook ** 2).sum(axis=1, keepdims=True)
    codebook = codebook / np.sqrt(norms_sq) * math.sqrt(2.0)
    
    return torch.tensor(codebook, dtype=torch.float32)


E8_ROOTS = build_e8_roots()
RANDOM_CODEBOOK = build_random_codebook(seed=42)


# ---------------------------------------------------------------------------
# 3. E8 Projection: map to 8D and snap to nearest root
# ---------------------------------------------------------------------------
class E8Quantize(torch.autograd.Function):
    """Quantize to E8 roots with STE."""
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
    """Quantize to random codebook with STE."""
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


class E8Projection(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.proj = nn.Linear(d_k, 8, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(2.0)
        snapped = E8Quantize.apply(z, E8_ROOTS.to(x.device))
        return snapped


class RandomProjection(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.proj = nn.Linear(d_k, 8, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(2.0)
        snapped = RandomQuantize.apply(z, RANDOM_CODEBOOK.to(x.device))
        return snapped


# ---------------------------------------------------------------------------
# 4. Vine: Learned flux modulation function
#    Vine(Q, K) → scalar weight in [0, 1] to scale signal through each edge
# ---------------------------------------------------------------------------
class VineGate(nn.Module):
    """
    Learned function that scores how much signal should flow through an open edge.
    
    Input: query (d_k,) and key (d_k,) vectors
    Output: scalar in [0, 1] representing gate strength
    
    Design: simple MLP on (query - key) difference
    This learns to upweight/downweight edges based on semantic similarity.
    """
    def __init__(self, d_k: int, hidden: int = 32):
        super().__init__()
        # Compute similarity features and pass through gate network
        self.net = nn.Sequential(
            nn.Linear(d_k, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Q: (batch, seq, d_k)
        K: (batch, seq, d_k)
        Output: (batch, seq, seq) attention weights
        """
        batch, seq_q, d_k = Q.shape
        _, seq_k, _ = K.shape
        
        # Expand to (batch, seq_q, seq_k, d_k)
        Q_exp = Q.unsqueeze(2).expand(batch, seq_q, seq_k, d_k)
        K_exp = K.unsqueeze(1).expand(batch, seq_q, seq_k, d_k)
        
        # Compute pairwise difference
        diff = Q_exp - K_exp  # (batch, seq_q, seq_k, d_k)
        
        # Pass through gate network
        gate = self.net(diff)  # (batch, seq_q, seq_k, 1)
        return gate.squeeze(-1)  # (batch, seq_q, seq_k)


# ---------------------------------------------------------------------------
# 5. Lattice Adjacency: binary mask from E8 (or random)
# ---------------------------------------------------------------------------
def e8_adjacency(psi_q: torch.Tensor, psi_k: torch.Tensor) -> torch.Tensor:
    """
    Binary adjacency from E8 roots: A_ij = 1 iff ⟨psi_q_i, psi_k_j⟩ == 1
    psi_q, psi_k: (batch, seq, 8), already snapped to E8 roots
    Output: (batch, seq, seq) binary mask
    """
    inner = torch.einsum("...id,...jd->...ij", psi_q, psi_k)
    A = (inner.round() == 1).float()
    return A


# ---------------------------------------------------------------------------
# 6. E8 + Vine Attention
# ---------------------------------------------------------------------------
class E8VineAttention(nn.Module):
    """
    Attention: A_ij = Vine(Q_i, K_j) ⊙ δ(⟨Ψ(Q_i), Ψ(K_j)⟩, 1)
    
    where:
      - δ(...) is the binary E8 adjacency (fixed topological constraint)
      - Vine(...) is a learned gate that scales flux through open edges
      - ⊙ is element-wise multiplication (Hadamard product)
    """
    def __init__(self, d_model: int, d_k: int, hidden_vine: int = 32):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_model)
        self.e8_q = E8Projection(d_k)
        self.e8_k = E8Projection(d_k)
        self.vine = VineGate(d_k, hidden=hidden_vine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        psi_q, psi_k = self.e8_q(Q), self.e8_k(K)
        
        # Topological constraint: binary adjacency from E8 roots
        delta = e8_adjacency(psi_q, psi_k)  # (batch, seq, seq)
        
        # Dynamic flux modulation: Vine learns to scale through open gates
        vine_weights = self.vine(Q, K)  # (batch, seq, seq)
        
        # Combined attention: topology gates * dynamic weights
        A = vine_weights * delta  # (batch, seq, seq)
        
        # Row-normalize
        row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        A_norm = A / row_sums
        
        out = A_norm @ V
        return out, A, delta, vine_weights


class RandomVineAttention(nn.Module):
    """E8+Vine but using random 240-vector codebook for control."""
    def __init__(self, d_model: int, d_k: int, hidden_vine: int = 32):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_model)
        self.random_q = RandomProjection(d_k)
        self.random_k = RandomProjection(d_k)
        self.vine = VineGate(d_k, hidden=hidden_vine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        psi_q, psi_k = self.random_q(Q), self.random_k(K)
        
        # Topological constraint: binary adjacency from random codebook
        delta = e8_adjacency(psi_q, psi_k)
        
        # Dynamic flux modulation
        vine_weights = self.vine(Q, K)
        
        # Combined attention
        A = vine_weights * delta
        row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        A_norm = A / row_sums
        
        out = A_norm @ V
        return out, A, delta, vine_weights


class SoftmaxAttention(nn.Module):
    """Baseline: standard softmax attention (no topological constraint)."""
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
        return out, A, None, None


# ---------------------------------------------------------------------------
# 7. Instrumentation
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
# 8. Training loop
# ---------------------------------------------------------------------------
def run_vine_experiment(steps: int = 200, batch: int = 16, seq: int = 32,
                        d_model: int = 64, d_k: int = 16, seed: int = 0):
    """
    Run three models:
      1. Softmax (baseline, no topological constraint)
      2. E8 + Vine (E8 adjacency + learned gating)
      3. Random + Vine (random adjacency + learned gating)
    
    Hypothesis:
      - If E8 structure matters, E8+Vine should outperform Random+Vine
      - Vine should learn to selectively upweight useful edges
      - Both should eventually outperform softmax on complex/structured tasks
        (though may lose on pure supervised regression)
    """
    torch.manual_seed(seed)
    
    # Fixed target
    target_proj = nn.Linear(d_model, d_model)
    for p in target_proj.parameters():
        p.requires_grad_(False)
    
    # Three models with independent initialization
    torch.manual_seed(seed + 100)
    softmax_model = SoftmaxAttention(d_model, d_k)
    
    torch.manual_seed(seed + 101)
    e8_model = E8VineAttention(d_model, d_k, hidden_vine=32)
    
    torch.manual_seed(seed + 102)
    random_model = RandomVineAttention(d_model, d_k, hidden_vine=32)
    
    models = {
        "softmax": softmax_model,
        "e8_vine": e8_model,
        "random_vine": random_model,
    }
    optims = {name: torch.optim.Adam(m.parameters(), lr=1e-3) for name, m in models.items()}
    
    log = {name: {"step": [], "erank": [], "grad_var": [], "loss": [], 
                   "sparsity": [], "delta_density": [], "vine_mean": []}
           for name in models}
    
    # Master seed for reproducible data
    torch.manual_seed(seed)
    
    for step in range(steps):
        x = torch.randn(batch, seq, d_model)
        with torch.no_grad():
            y_target = target_proj(x)
        
        for name, model in models.items():
            optims[name].zero_grad()
            result = model(x)
            out = result[0]
            A = result[1]
            delta = result[2]  # None for softmax
            vine_weights = result[3]  # None for softmax
            
            loss = F.mse_loss(out, y_target)
            loss.backward()
            optims[name].step()
            
            if step % 10 == 0:
                log[name]["step"].append(step)
                log[name]["erank"].append(effective_rank(out))
                log[name]["grad_var"].append(grad_norm_variance(model))
                log[name]["loss"].append(loss.item())
                
                if name == "softmax":
                    # Softmax: effective sparsity is density of top-k
                    sparsity = (A < 0.01).float().mean().item()
                    log[name]["sparsity"].append(sparsity)
                    log[name]["delta_density"].append(0.0)
                    log[name]["vine_mean"].append(0.0)
                else:
                    # E8+Vine or Random+Vine
                    sparsity = (A < 0.001).float().mean().item()
                    delta_density = delta.mean().item()
                    vine_mean = vine_weights.mean().item()
                    log[name]["sparsity"].append(sparsity)
                    log[name]["delta_density"].append(delta_density)
                    log[name]["vine_mean"].append(vine_mean)
    
    return log


# ---------------------------------------------------------------------------
# 9. Results
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*90)
    print("E8 LATTICE + VINE MODULATION: Topological Gates × Dynamic Flux")
    print("="*90)
    
    results = run_vine_experiment(steps=200, seed=0)
    
    print("\n### SOFTMAX BASELINE ###")
    print("(No topological constraint, full dense attention)")
    for i, step in enumerate(results["softmax"]["step"]):
        print(f"step {step:4d} | loss {results['softmax']['loss'][i]:.4f} | "
              f"erank {results['softmax']['erank'][i]:6.2f} | "
              f"grad_var {results['softmax']['grad_var'][i]:8.6f}")
    
    print("\n### E8 + VINE (TREATMENT) ###")
    print("(E8 adjacency gates topological pathways; Vine modulates flux through open gates)")
    for i, step in enumerate(results["e8_vine"]["step"]):
        print(f"step {step:4d} | loss {results['e8_vine']['loss'][i]:.4f} | "
              f"erank {results['e8_vine']['erank'][i]:6.2f} | "
              f"grad_var {results['e8_vine']['grad_var'][i]:8.6f} | "
              f"Δ_density {results['e8_vine']['delta_density'][i]:.4f} | "
              f"Vine_mean {results['e8_vine']['vine_mean'][i]:.4f}")
    
    print("\n### RANDOM + VINE (CONTROL) ###")
    print("(Random adjacency gates topological pathways; Vine modulates flux through open gates)")
    for i, step in enumerate(results["random_vine"]["step"]):
        print(f"step {step:4d} | loss {results['random_vine']['loss'][i]:.4f} | "
              f"erank {results['random_vine']['erank'][i]:6.2f} | "
              f"grad_var {results['random_vine']['grad_var'][i]:8.6f} | "
              f"Δ_density {results['random_vine']['delta_density'][i]:.4f} | "
              f"Vine_mean {results['random_vine']['vine_mean'][i]:.4f}")
    
    # Statistical summary
    print("\n" + "="*90)
    print("SUMMARY: Final 5 checkpoints (steps 150-190)")
    print("="*90)
    
    for name in ["softmax", "e8_vine", "random_vine"]:
        losses = results[name]["loss"][-5:]
        eranks = results[name]["erank"][-5:]
        grad_vars = results[name]["grad_var"][-5:]
        
        print(f"\n{name.upper()}:")
        print(f"  Loss:        {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"  Erank:       {np.mean(eranks):.2f} ± {np.std(eranks):.2f}")
        print(f"  Grad Var:    {np.mean(grad_vars):.6f} ± {np.std(grad_vars):.6f}")
        
        if name != "softmax":
            vine_means = results[name]["vine_mean"][-5:]
            delta_dens = results[name]["delta_density"][-5:]
            print(f"  Vine Mean:   {np.mean(vine_means):.4f} ± {np.std(vine_means):.4f}")
            print(f"  Δ Density:   {np.mean(delta_dens):.4f} ± {np.std(delta_dens):.4f}")
    
    # Comparison
    print("\n" + "="*90)
    print("CONTROL INTERPRETATION")
    print("="*90)
    
    e8_loss = np.mean(results["e8_vine"]["loss"][-5:])
    random_loss = np.mean(results["random_vine"]["loss"][-5:])
    softmax_loss = np.mean(results["softmax"]["loss"][-5:])
    
    e8_erank = np.mean(results["e8_vine"]["erank"][-5:])
    random_erank = np.mean(results["random_vine"]["erank"][-5:])
    
    print(f"\nLoss comparison (lower is better):")
    print(f"  Softmax:        {softmax_loss:.4f}")
    print(f"  E8+Vine:        {e8_loss:.4f} (Δ = {e8_loss - softmax_loss:+.4f})")
    print(f"  Random+Vine:    {random_loss:.4f} (Δ = {random_loss - softmax_loss:+.4f})")
    
    print(f"\nEffective rank comparison (higher = more expressive):")
    print(f"  E8+Vine vs Random+Vine: {e8_erank - random_erank:+.2f}")
    
    if abs(e8_loss - random_loss) < 0.01:
        print("\n✓ E8+Vine and Random+Vine converge to SIMILAR loss")
        print("  → Vine learns equally well regardless of lattice structure")
    else:
        loss_diff = e8_loss - random_loss
        winner = "E8+Vine" if loss_diff < 0 else "Random+Vine"
        print(f"\n✗ E8+Vine and Random+Vine DIVERGE by {abs(loss_diff):.4f}")
        print(f"  → {winner} has an advantage: E8 structure DOES matter when combined with learned gating")
    
    print("\n" + "="*90)
