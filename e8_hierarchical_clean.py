"""
e8_hierarchical_clean.py

Clean version of hierarchical pivot with file output redirection and adversarial graph ablation.

Part 1: Rerun tree and cycle with clean logging (no duplicate prints)
Part 2: Adversarial graph ablation — manually construct a graph structurally opposite to E8:
        - E8 has ~240 nodes with ~56 edges per node (sparse, degree-56)
        - Adversarial: Complete graph or near-complete (every node connects to most others)
        - Also test: High-clustering coefficient + dense local neighborhoods
        - Prediction: If accidental-alignment is true, E8 should lose badly here
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. Graph generators
# ---------------------------------------------------------------------------
def build_tree(seq_len: int, seed: int = 0):
    """Binary tree: each node i has children 2i+1, 2i+2."""
    rng = random.Random(seed)
    adj = [set() for _ in range(seq_len)]
    for i in range(seq_len):
        for child in (2 * i + 1, 2 * i + 2):
            if child < seq_len:
                adj[i].add(child)
                adj[child].add(i)
    dist = _compute_distances(seq_len, adj)
    return adj, dist


def build_cycle(seq_len: int, seed: int = 0):
    """Ring graph: each node connects to its two neighbors."""
    rng = random.Random(seed)
    adj = [set() for _ in range(seq_len)]
    for i in range(seq_len):
        j = (i + 1) % seq_len
        adj[i].add(j)
        adj[j].add(i)
    dist = _compute_distances(seq_len, adj)
    return adj, dist


def build_adversarial_dense(seq_len: int, seed: int = 0):
    """
    Adversarial to E8's sparse structure: complete graph or near-complete.
    E8 adjacency has ~56 edges per node out of 240 possible.
    This graph has high degree: every node connects to ~90% of others.
    
    Intuition: if E8's advantage on cycle/tree is about accident matching,
    it should fail here. If E8 still does well, something else is going on.
    """
    rng = random.Random(seed)
    adj = [set() for _ in range(seq_len)]
    
    # High-density random graph: each pair (i,j) with i<j connects with probability 0.7
    p_edge = 0.7
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if rng.random() < p_edge:
                adj[i].add(j)
                adj[j].add(i)
    
    dist = _compute_distances(seq_len, adj)
    return adj, dist


def build_adversarial_clustering(seq_len: int, seed: int = 0):
    """
    Adversarial via clustering: divide nodes into K clusters, within-cluster
    edges are dense, between-cluster edges are sparse. This is structurally
    opposite to E8's globally uniform sparsity.
    
    E8: ~240 nodes, ~56 edges per node = uniform sparsity
    This: nodes organize into local cliques (high local clustering coefficient)
    """
    rng = random.Random(seed)
    n_clusters = max(2, seq_len // 8)  # ~4 clusters for seq_len=32
    cluster_size = seq_len // n_clusters
    
    adj = [set() for _ in range(seq_len)]
    
    # Within-cluster: connect every node to every other in its cluster (dense)
    for c in range(n_clusters):
        start = c * cluster_size
        end = start + cluster_size if c < n_clusters - 1 else seq_len
        for i in range(start, end):
            for j in range(i + 1, end):
                adj[i].add(j)
                adj[j].add(i)
    
    # Between-cluster: sparse random edges (~10% chance)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if i // cluster_size != j // cluster_size:
                if rng.random() < 0.1:
                    adj[i].add(j)
                    adj[j].add(i)
    
    dist = _compute_distances(seq_len, adj)
    return adj, dist


def _compute_distances(seq_len: int, adj) -> torch.Tensor:
    """BFS to compute all-pairs shortest-path distances."""
    dist = torch.full((seq_len, seq_len), float(seq_len), dtype=torch.float32)
    for src in range(seq_len):
        visited = {src: 0}
        frontier = [src]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in visited:
                        visited[v] = d
                        nxt.append(v)
            frontier = nxt
        for node, dd in visited.items():
            dist[src, node] = dd
    return dist


# ---------------------------------------------------------------------------
# 2. Task generator
# ---------------------------------------------------------------------------
def make_hierarchical_batch(batch: int, seq_len: int, d_model: int,
                              adj, dist: torch.Tensor, seed_offset: int = 0):
    """Generate (x, y) where y depends on graph structure."""
    g = torch.Generator().manual_seed(1000 + seed_offset)
    x = torch.randn(batch, seq_len, d_model, generator=g)

    torch.manual_seed(42)
    W1 = torch.randn(d_model, d_model) * 0.5
    W2 = torch.randn(d_model, d_model) * 0.5

    self_term = torch.tanh(x @ W1)

    weight = 1.0 / (1.0 + dist)
    weight = weight / weight.sum(dim=-1, keepdim=True)
    neighbor_feat = torch.tanh(x @ W2)
    neighbor_term = torch.einsum("ij,bjd->bid", weight, neighbor_feat)

    y = self_term + neighbor_term
    return x, y


# ---------------------------------------------------------------------------
# 3. E8 roots and codebook
# ---------------------------------------------------------------------------
def build_e8_roots() -> torch.Tensor:
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


def build_random_codebook(n: int = 240, dim: int = 8, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, dim, generator=g)
    v = v / v.norm(dim=-1, keepdim=True) * math.sqrt(2.0)
    return v


E8_ROOTS = build_e8_roots()
RANDOM_CODEBOOK = build_random_codebook()


# ---------------------------------------------------------------------------
# 4. Attention layers (unchanged)
# ---------------------------------------------------------------------------
class VQSnap(torch.autograd.Function):
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


def vq_snap(x, codebook):
    return VQSnap.apply(x, codebook)


class LatticeProjection(nn.Module):
    def __init__(self, d_k: int, codebook: torch.Tensor):
        super().__init__()
        self.proj = nn.Linear(d_k, 8, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.register_buffer("codebook", codebook)

    def forward(self, x):
        z = self.proj(x)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(2.0)
        return vq_snap(z, self.codebook)


def lattice_adjacency(psi_q, psi_k):
    inner = torch.einsum("...id,...jd->...ij", psi_q, psi_k)
    return (inner.round() == 1).float()


class LatticeAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, codebook: torch.Tensor):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_model)
        self.lat_q = LatticeProjection(d_k, codebook)
        self.lat_k = LatticeProjection(d_k, codebook)

    def forward(self, x):
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        psi_q, psi_k = self.lat_q(Q), self.lat_k(K)
        A = lattice_adjacency(psi_q, psi_k)
        row_sums = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        out = (A / row_sums) @ V
        return out, A


class SoftmaxAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_model)
        self.d_k = d_k

    def forward(self, x):
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        A = F.softmax(scores, dim=-1)
        out = A @ V
        return out, A


# ---------------------------------------------------------------------------
# 5. Instrumentation
# ---------------------------------------------------------------------------
def effective_rank(matrix, eps=1e-7):
    m = matrix.reshape(-1, matrix.shape[-1]).detach()
    if m.shape[0] < 2:
        return float("nan")
    s = torch.linalg.svdvals(m)
    s = s / (s.sum() + eps)
    s = s[s > eps]
    return (-(s * s.log()).sum()).exp().item()


def grad_norm_variance(model):
    norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    if len(norms) < 2:
        return float("nan")
    return torch.tensor(norms).var(unbiased=True).item()


def graph_alignment_score(A: torch.Tensor, dist: torch.Tensor) -> float:
    """Overlap of learned adjacency A with true edges (dist == 1)."""
    true_edges = (dist == 1).float()
    A_mean = A.mean(dim=0)
    overlap = (A_mean * true_edges).sum()
    denom = true_edges.sum().clamp(min=1.0)
    return (overlap / denom).item()


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------
def run_experiment(steps=200, batch=16, seq=32, d_model=64, d_k=16,
                    graph_builder=None, graph_name="unknown", seed=0, log_file=None):
    """Run single-seed experiment with optional file logging."""
    torch.manual_seed(seed)
    adj, dist = graph_builder(seq, seed=seed)

    models = {
        "softmax": SoftmaxAttention(d_model, d_k),
        "e8": LatticeAttention(d_model, d_k, E8_ROOTS),
        "random": LatticeAttention(d_model, d_k, RANDOM_CODEBOOK),
    }
    optims = {name: torch.optim.Adam(m.parameters(), lr=1e-3) for name, m in models.items()}

    log = {name: {"step": [], "erank": [], "grad_var": [], "loss": [],
                   "sparsity": [], "graph_align": []} for name in models}

    for step in range(steps):
        x, y_target = make_hierarchical_batch(batch, seq, d_model, adj, dist,
                                                seed_offset=step)
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
                if name == "softmax":
                    sparsity = (A < 1e-6).float().mean().item()
                else:
                    sparsity = 1.0 - A.mean().item()
                log[name]["sparsity"].append(sparsity)
                log[name]["graph_align"].append(graph_alignment_score(A, dist))

    return log


def run_multi_seed(n_seeds=5, log_file=None, **kwargs):
    """Run experiment across seeds, collect final metrics."""
    finals = {name: {"loss": [], "graph_align": [], "erank": []} for name in
              ("softmax", "e8", "random")}

    for s in range(n_seeds):
        log = run_experiment(seed=s, **kwargs)
        for name in finals:
            finals[name]["loss"].append(log[name]["loss"][-1])
            finals[name]["graph_align"].append(log[name]["graph_align"][-1])
            finals[name]["erank"].append(log[name]["erank"][-1])

    return finals


# ---------------------------------------------------------------------------
# 7. Main: clean logging
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Part 1: Clean tree and cycle runs
    for graph_kind, builder in [("tree", build_tree), ("cycle", build_cycle)]:
        log_filename = f"e8_hierarchical_{graph_kind}_{timestamp}.log"
        
        with open(log_filename, "w") as f:
            f.write(f"{'='*90}\n")
            f.write(f"E8 Hierarchical Pivot: {graph_kind.upper()} Graph\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"{'='*90}\n\n")
            
            # Single-seed run
            f.write(f"Single-seed run (TREE GRAPH):\n")
            f.write(f"{'='*90}\n")
            log = run_experiment(graph_builder=builder, graph_name=graph_kind, seed=0)
            
            for name, data in log.items():
                f.write(f"\n=== {name.upper()} ===\n")
                for i, step in enumerate(data["step"]):
                    f.write(f"step {step:4d} | loss {data['loss'][i]:.4f} | erank {data['erank'][i]:.2f} "
                          f"| grad_var {data['grad_var'][i]:.6f} | sparsity {data['sparsity'][i]:.4f} "
                          f"| graph_align {data['graph_align'][i]:.4f}\n")
            
            # Multi-seed run
            f.write(f"\n\n{'='*90}\n")
            f.write(f"Multi-seed run (n=5 seeds) -- DEFINITIVE STATISTICS\n")
            f.write(f"{'='*90}\n\n")
            
            finals = run_multi_seed(n_seeds=5, graph_builder=builder, graph_name=graph_kind)
            
            for name in ["softmax", "e8", "random"]:
                loss_arr = np.array(finals[name]["loss"])
                align_arr = np.array(finals[name]["graph_align"])
                erank_arr = np.array(finals[name]["erank"])
                
                f.write(f"\n{name.upper()}:\n")
                f.write(f"  Loss:        {loss_arr.mean():.4f} ± {loss_arr.std():.4f}\n")
                f.write(f"  Graph Align: {align_arr.mean():.4f} ± {align_arr.std():.4f}\n")
                f.write(f"  Erank:       {erank_arr.mean():.4f} ± {erank_arr.std():.4f}\n")
            
            # Explicit comparison
            e8_loss = np.array(finals["e8"]["loss"]).mean()
            random_loss = np.array(finals["random"]["loss"]).mean()
            e8_align = np.array(finals["e8"]["graph_align"]).mean()
            random_align = np.array(finals["random"]["graph_align"]).mean()
            
            f.write(f"\n\nCOMPARISON (E8 vs Random):\n")
            f.write(f"  Loss Δ:       {e8_loss - random_loss:+.4f}\n")
            f.write(f"  Align Δ:      {e8_align - random_align:+.4f}  ({(e8_align - random_align)*100:+.2f}pp)\n")
        
        print(f"✓ Logged {graph_kind} results to {log_filename}")
    
    # Part 2: Adversarial graph ablations
    print(f"\n{'='*90}")
    print("ADVERSARIAL GRAPH ABLATION: Testing E8 on structurally opposite graphs")
    print(f"{'='*90}\n")
    
    for graph_name, builder in [
        ("adversarial_dense", build_adversarial_dense),
        ("adversarial_clustering", build_adversarial_clustering),
    ]:
        log_filename = f"e8_ablation_{graph_name}_{timestamp}.log"
        
        with open(log_filename, "w") as f:
            f.write(f"{'='*90}\n")
            f.write(f"E8 Adversarial Ablation: {graph_name.upper()}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"{'='*90}\n\n")
            
            f.write(f"Hypothesis: If E8's advantage on tree/cycle is accidental alignment,\n")
            f.write(f"it should LOSE on graphs deliberately constructed to be opposite.\n\n")
            
            # Single-seed run
            f.write(f"Single-seed run:\n")
            f.write(f"{'='*90}\n")
            log = run_experiment(graph_builder=builder, graph_name=graph_name, seed=0)
            
            for name, data in log.items():
                f.write(f"\n=== {name.upper()} ===\n")
                for i, step in enumerate(data["step"]):
                    f.write(f"step {step:4d} | loss {data['loss'][i]:.4f} | erank {data['erank'][i]:.2f} "
                          f"| graph_align {data['graph_align'][i]:.4f}\n")
            
            # Multi-seed run
            f.write(f"\n\n{'='*90}\n")
            f.write(f"Multi-seed run (n=5 seeds)\n")
            f.write(f"{'='*90}\n\n")
            
            finals = run_multi_seed(n_seeds=5, graph_builder=builder, graph_name=graph_name)
            
            for name in ["softmax", "e8", "random"]:
                loss_arr = np.array(finals[name]["loss"])
                align_arr = np.array(finals[name]["graph_align"])
                
                f.write(f"\n{name.upper()}:\n")
                f.write(f"  Loss:        {loss_arr.mean():.4f} ± {loss_arr.std():.4f}\n")
                f.write(f"  Graph Align: {align_arr.mean():.4f} ± {align_arr.std():.4f}\n")
            
            # Explicit comparison
            e8_loss = np.array(finals["e8"]["loss"]).mean()
            random_loss = np.array(finals["random"]["loss"]).mean()
            e8_align = np.array(finals["e8"]["graph_align"]).mean()
            random_align = np.array(finals["random"]["graph_align"]).mean()
            
            f.write(f"\n\nCOMPARISON (E8 vs Random):\n")
            f.write(f"  Loss Δ:       {e8_loss - random_loss:+.4f}\n")
            f.write(f"  Align Δ:      {e8_align - random_align:+.4f}  ({(e8_align - random_align)*100:+.2f}pp)\n")
            
            if e8_loss > random_loss + 0.01:
                f.write(f"\n  INTERPRETATION: E8 LOSES on adversarial graph → accidental alignment confirmed.\n")
            elif abs(e8_loss - random_loss) < 0.01:
                f.write(f"\n  INTERPRETATION: E8 and Random tie → E8 structure doesn't hurt or help.\n")
            else:
                f.write(f"\n  INTERPRETATION: E8 still competitive or wins → something else is going on.\n")
        
        print(f"✓ Logged adversarial ablation ({graph_name}) to {log_filename}")
    
    print(f"\n✓ All runs complete. Check .log files for clean output.")
