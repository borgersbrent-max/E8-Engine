"""
e8_hierarchical_debug.py

Corrected version with explicit graph diagnostics before training:
1. Verify each graph builder produces a DIFFERENT distance matrix
2. Print graph properties (edge count, avg degree, diameter) for each condition
3. Show first true_edges matrix so we can spot bugs in graph construction
4. Fix hardcoded "TREE GRAPH" string in file output
5. No silent fallbacks; catch and report errors explicitly
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. Graph generators with explicit error handling
# ---------------------------------------------------------------------------
def build_tree(seq_len: int, seed: int = 0):
    """Binary tree: each node i has children 2i+1, 2i+2."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    rng = random.Random(seed)
    adj = [set() for _ in range(seq_len)]
    for i in range(seq_len):
        for child in (2 * i + 1, 2 * i + 2):
            if child < seq_len:
                adj[i].add(child)
                adj[child].add(i)
    dist = _compute_distances(seq_len, adj)
    return adj, dist, "tree"


def build_cycle(seq_len: int, seed: int = 0):
    """Ring graph: each node connects to its two neighbors."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    rng = random.Random(seed)
    adj = [set() for _ in range(seq_len)]
    for i in range(seq_len):
        j = (i + 1) % seq_len
        adj[i].add(j)
        adj[j].add(i)
    dist = _compute_distances(seq_len, adj)
    return adj, dist, "cycle"


def build_adversarial_dense(seq_len: int, seed: int = 0):
    """
    Dense random graph: ~70% edge probability.
    Opposite of E8's ~12% sparsity.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    rng = random.Random(seed)
    adj = [set() for _ in range(seq_len)]
    
    p_edge = 0.7
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if rng.random() < p_edge:
                adj[i].add(j)
                adj[j].add(i)
    
    dist = _compute_distances(seq_len, adj)
    return adj, dist, "adversarial_dense"


def build_adversarial_clustering(seq_len: int, seed: int = 0):
    """
    High-clustering graph: nodes in clusters with dense within-cluster,
    sparse between-cluster edges.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    rng = random.Random(seed)
    n_clusters = max(2, seq_len // 8)
    cluster_size = seq_len // n_clusters
    
    adj = [set() for _ in range(seq_len)]
    
    # Within-cluster: dense
    for c in range(n_clusters):
        start = c * cluster_size
        end = start + cluster_size if c < n_clusters - 1 else seq_len
        for i in range(start, end):
            for j in range(i + 1, end):
                adj[i].add(j)
                adj[j].add(i)
    
    # Between-cluster: sparse (10% probability)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if i // cluster_size != j // cluster_size:
                if rng.random() < 0.1:
                    adj[i].add(j)
                    adj[j].add(i)
    
    dist = _compute_distances(seq_len, adj)
    return adj, dist, "adversarial_clustering"


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


def graph_stats(adj, dist, name: str) -> dict:
    """Compute and return graph properties."""
    n = len(adj)
    edge_count = sum(len(neighbors) for neighbors in adj) // 2
    avg_degree = 2 * edge_count / n if n > 0 else 0
    diameter = int(dist.max().item())
    
    # Count edges at distance 1 (true neighbors in routing)
    true_edges_at_dist_1 = int(((dist == 1).float().sum() / 2).item())
    
    stats = {
        "name": name,
        "nodes": n,
        "edges": edge_count,
        "avg_degree": avg_degree,
        "diameter": diameter,
        "true_edges_at_dist_1": true_edges_at_dist_1,
    }
    return stats


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
# 4. Attention layers
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
                    graph_builder=None, seed=0):
    """Run single-seed experiment."""
    torch.manual_seed(seed)
    adj, dist, graph_name = graph_builder(seq, seed=seed)

    models = {
        "softmax": SoftmaxAttention(d_model, d_k),
        "e8": LatticeAttention(d_model, d_k, E8_ROOTS),
        "random": LatticeAttention(d_model, d_k, RANDOM_CODEBOOK),
    }
    optims = {name: torch.optim.Adam(m.parameters(), lr=1e-3) for name, m in models.items()}

    log = {name: {"step": [], "erank": [], "loss": [], "graph_align": []} for name in models}

    for step in range(steps):
        x, y_target = make_hierarchical_batch(batch, seq, d_model, adj, dist, seed_offset=step)
        for name, model in models.items():
            optims[name].zero_grad()
            out, A = model(x)
            loss = F.mse_loss(out, y_target)
            loss.backward()
            optims[name].step()

            if step % 10 == 0:
                log[name]["step"].append(step)
                log[name]["erank"].append(effective_rank(out))
                log[name]["loss"].append(loss.item())
                log[name]["graph_align"].append(graph_alignment_score(A, dist))

    return log


def run_multi_seed(n_seeds=5, **kwargs):
    """Run experiment across seeds."""
    finals = {name: {"loss": [], "graph_align": []} for name in ("softmax", "e8", "random")}

    for s in range(n_seeds):
        log = run_experiment(seed=s, **kwargs)
        for name in finals:
            finals[name]["loss"].append(log[name]["loss"][-1])
            finals[name]["graph_align"].append(log[name]["graph_align"][-1])

    return finals


# ---------------------------------------------------------------------------
# 7. Main: diagnostic version
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*90}")
    print("GRAPH DIAGNOSTICS: Verify each graph is distinct")
    print(f"{'='*90}\n")
    
    graph_configs = [
        ("tree", build_tree),
        ("cycle", build_cycle),
        ("adversarial_dense", build_adversarial_dense),
        ("adversarial_clustering", build_adversarial_clustering),
    ]
    
    all_dists = {}
    
    for graph_name, builder in graph_configs:
        try:
            adj, dist, actual_name = builder(seq_len=32, seed=0)
            stats = graph_stats(adj, dist, actual_name)
            all_dists[actual_name] = dist
            
            print(f"\n{actual_name.upper()}:")
            print(f"  Nodes: {stats['nodes']}")
            print(f"  Edges: {stats['edges']}")
            print(f"  Avg Degree: {stats['avg_degree']:.2f}")
            print(f"  Diameter: {stats['diameter']}")
            print(f"  True Edges (dist==1): {stats['true_edges_at_dist_1']}")
            
            # Show first row of distance matrix to spot issues
            print(f"  Distance matrix first row: {dist[0, :10].tolist()}")
            print(f"  True edges mask (dist==1) first row: {(dist[0, :10] == 1).tolist()}")
            
        except Exception as e:
            print(f"\n{graph_name.upper()}: ERROR - {e}")
            print(f"  (This graph condition will be skipped)")
    
    # Verify distinctness
    print(f"\n{'='*90}")
    print("DISTINCTNESS CHECK")
    print(f"{'='*90}\n")
    
    names = list(all_dists.keys())
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            are_equal = torch.allclose(all_dists[name1], all_dists[name2])
            status = "⚠️  IDENTICAL" if are_equal else "✓ DISTINCT"
            print(f"{name1:25s} vs {name2:25s}: {status}")
    
    # Now run training with proper logging
    print(f"\n{'='*90}")
    print("TRAINING RUNS")
    print(f"{'='*90}\n")
    
    for graph_name, builder in graph_configs:
        log_filename = f"e8_debug_{graph_name}_{timestamp}.log"
        
        try:
            with open(log_filename, "w") as f:
                f.write(f"{'='*90}\n")
                f.write(f"E8 Hierarchical Debug: {graph_name.upper()}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"{'='*90}\n\n")
                
                # Single-seed run
                f.write(f"Single-seed run (seed=0, {graph_name.upper()}):\n")
                f.write(f"{'='*90}\n")
                log = run_experiment(graph_builder=builder, seed=0)
                
                for name, data in log.items():
                    f.write(f"\n=== {name.upper()} ===\n")
                    for i, step in enumerate(data["step"]):
                        f.write(f"step {step:4d} | loss {data['loss'][i]:.4f} | "
                              f"erank {data['erank'][i]:.2f} | "
                              f"graph_align {data['graph_align'][i]:.4f}\n")
                
                # Multi-seed run
                f.write(f"\n\n{'='*90}\n")
                f.write(f"Multi-seed run (n=5 seeds, {graph_name.upper()})\n")
                f.write(f"{'='*90}\n\n")
                
                finals = run_multi_seed(n_seeds=5, graph_builder=builder)
                
                for name in ["softmax", "e8", "random"]:
                    loss_arr = np.array(finals[name]["loss"])
                    align_arr = np.array(finals[name]["graph_align"])
                    
                    f.write(f"\n{name.upper()}:\n")
                    f.write(f"  Loss:        {loss_arr.mean():.4f} ± {loss_arr.std():.4f}\n")
                    f.write(f"  Graph Align: {align_arr.mean():.4f} ± {align_arr.std():.4f}\n")
                
                # Comparison
                e8_loss = np.array(finals["e8"]["loss"]).mean()
                random_loss = np.array(finals["random"]["loss"]).mean()
                
                f.write(f"\nCOMPARISON (E8 vs Random):\n")
                f.write(f"  Loss Δ: {e8_loss - random_loss:+.4f}\n")
            
            print(f"✓ {graph_name:30s} → {log_filename}")
            
        except Exception as e:
            print(f"✗ {graph_name:30s} → ERROR: {e}")
    
    print(f"\n{'='*90}")
    print("Diagnostic run complete. Check .log files for results.")
    print(f"{'='*90}\n")
