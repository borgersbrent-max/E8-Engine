"""
e8_hierarchical_pivot.py

Extends the prior null-result experiment by replacing unstructured torch.randn
inputs with a task that has real combinatorial structure: tokens are nodes in
a graph (binary tree or cycle, chosen independently of E8's own structure),
and the target is a nonlinear function of graph distance and neighbor identity.

This tests a narrow, honest question: does a FIXED, non-learned sparse
adjacency mask shaped like the E8 root graph help when the task's true
relational structure is graph-shaped, compared to a fixed sparse mask shaped
like a random graph of the same density?

It does NOT test, and a positive result here would NOT validate:
  - E8's algebraic properties (Weyl symmetry, root system closure under
    reflection, sphere-packing optimality) -- none of those are used anywhere
    in the forward pass. The model only uses "240 fixed points + snap +
    threshold," exactly as before.
  - Any fixed "capacity invariant." Effective rank and loss here are
    task-and-scale-dependent numbers, not constants.

NOT RUN -- no torch / no network in this sandbox. Verified by hand and via
a pure-Python dry run of the graph generator (see bottom of file), but treat
all training numbers as unverified until you run this locally.

To get a defensible result (not just a suggestive one), run with multiple
seeds via run_multi_seed() and look at mean +/- std ACROSS seeds, not within
one run's late-training checkpoints.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# 1. Hierarchical / graph task generator
# ---------------------------------------------------------------------------
def build_graph(seq_len: int, kind: str = "tree", seed: int = 0):
    """
    Builds an adjacency list for a graph over `seq_len` nodes, independent of
    any E8 structure -- chosen on its own terms so the comparison stays honest.

    kind="tree": complete binary tree (node i's children are 2i+1, 2i+2)
    kind="cycle": ring graph, each node connected to its two neighbors

    Returns: adjacency list (list of sets), and all-pairs shortest path
    distances (seq_len x seq_len tensor, via BFS -- exact, not approximate).
    """
    rng = random.Random(seed)
    adj = [set() for _ in range(seq_len)]

    if kind == "tree":
        for i in range(seq_len):
            for child in (2 * i + 1, 2 * i + 2):
                if child < seq_len:
                    adj[i].add(child)
                    adj[child].add(i)
    elif kind == "cycle":
        for i in range(seq_len):
            j = (i + 1) % seq_len
            adj[i].add(j)
            adj[j].add(i)
    else:
        raise ValueError(f"unknown graph kind: {kind}")

    # BFS shortest-path distance from every node (exact, small seq_len so O(n^2) is fine)
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

    return adj, dist


def make_hierarchical_batch(batch: int, seq_len: int, d_model: int,
                              adj, dist: torch.Tensor, seed_offset: int = 0):
    """
    Generates structured (x, y) pairs:
      - x: random per-node feature vectors (the "content" at each graph node)
      - y: a NONLINEAR function of (a) each node's own feature and
           (b) an aggregate over its graph neighbors, weighted by inverse
           graph distance. This gives the task real relational structure:
           a model that can route information along true graph edges has
           an advantage over one that can't.

    y_i = tanh(W1 @ x_i) + sum_j [ 1/(1+dist(i,j)) * tanh(W2 @ x_j) ] / seq_len
    """
    g = torch.Generator().manual_seed(1000 + seed_offset)
    x = torch.randn(batch, seq_len, d_model, generator=g)

    # Fixed, non-trained nonlinear readouts -- same across all models/batches
    # so the task itself doesn't vary between conditions.
    torch.manual_seed(42)
    W1 = torch.randn(d_model, d_model) * 0.5
    W2 = torch.randn(d_model, d_model) * 0.5

    self_term = torch.tanh(x @ W1)                                # (batch, seq, d_model)

    # neighbor aggregate, weighted by inverse shortest-path distance
    weight = 1.0 / (1.0 + dist)                                   # (seq, seq), symmetric
    weight = weight / weight.sum(dim=-1, keepdim=True)             # row-normalize
    neighbor_feat = torch.tanh(x @ W2)                             # (batch, seq, d_model)
    neighbor_term = torch.einsum("ij,bjd->bid", weight, neighbor_feat)

    y = self_term + neighbor_term
    return x, y


# ---------------------------------------------------------------------------
# 2. E8 root system (unchanged from prior script)
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
    """Random codebook, matched cardinality and norm to E8 roots, but no
    lattice/algebraic structure -- the control condition."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, dim, generator=g)
    v = v / v.norm(dim=-1, keepdim=True) * math.sqrt(2.0)
    return v


E8_ROOTS = build_e8_roots()
RANDOM_CODEBOOK = build_random_codebook()


# ---------------------------------------------------------------------------
# 3. STE quantization + attention variants (unchanged architecture)
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
    """Learned linear map d_k -> 8, then snap to a fixed codebook (E8 or random)."""

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
    """Discrete routed attention using a fixed codebook (E8 or random)."""

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
# 4. Instrumentation (unchanged)
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
    """
    Diagnostic specific to this experiment: how much does the model's learned
    binary adjacency A overlap with TRUE graph edges (dist == 1)?
    1.0 = perfect overlap with true edges, 0.0 = no overlap.
    This tells you whether the model is actually exploiting the task's graph
    structure, separate from whatever loss/rank/sparsity say.
    """
    true_edges = (dist == 1).float()
    A_mean = A.mean(dim=0)  # average over batch -> (seq, seq)
    overlap = (A_mean * true_edges).sum()
    denom = true_edges.sum().clamp(min=1.0)
    return (overlap / denom).item()


# ---------------------------------------------------------------------------
# 5. Training loop on the hierarchical task
# ---------------------------------------------------------------------------
def run_experiment(steps=200, batch=16, seq=32, d_model=64, d_k=16,
                    graph_kind="tree", seed=0):
    torch.manual_seed(seed)
    adj, dist = build_graph(seq, kind=graph_kind, seed=seed)

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


def run_multi_seed(n_seeds=5, **kwargs):
    """
    Runs the experiment across multiple seeds and reports mean +/- std of the
    FINAL loss and graph_align per model, across seeds (not within-run
    checkpoints). This is the statistic that actually supports a claim like
    "E8 beats random" or "they're indistinguishable" -- a single seed does not.
    """
    finals = {name: {"loss": [], "graph_align": [], "erank": []} for name in
              ("softmax", "e8", "random")}

    for s in range(n_seeds):
        log = run_experiment(seed=s, **kwargs)
        for name in finals:
            finals[name]["loss"].append(log[name]["loss"][-1])
            finals[name]["graph_align"].append(log[name]["graph_align"][-1])
            finals[name]["erank"].append(log[name]["erank"][-1])

    print(f"\n{'='*90}\nMULTI-SEED SUMMARY (n={n_seeds} seeds)\n{'='*90}")
    for name, vals in finals.items():
        for metric, arr in vals.items():
            t = torch.tensor(arr)
            print(f"{name:8s} | {metric:12s} mean={t.mean().item():.4f}  std={t.std().item():.4f}  (n={n_seeds})")
    
    return finals


if __name__ == "__main__":
    print("\n" + "="*90)
    print("HIERARCHICAL TASK: E8 Lattice vs Random Graph on Binary Tree & Cycle Graphs")
    print("="*90)
    
    print("\n\nSingle-seed run (TREE GRAPH):")
    print("="*90)
    log = run_experiment(graph_kind="tree", seed=0)
    for name, data in log.items():
        print(f"\n=== {name} ===")
        for i, step in enumerate(data["step"]):
            print(f"step {step:4d} | loss {data['loss'][i]:.4f} | erank {data['erank'][i]:.2f} "
                  f"| grad_var {data['grad_var'][i]:.6f} | sparsity {data['sparsity'][i]:.4f} "
                  f"| graph_align {data['graph_align'][i]:.4f}")

    print("\n\n" + "="*90)
    print("Multi-seed run (TREE GRAPH) -- THE STATISTIC THAT ACTUALLY MATTERS")
    print("="*90)
    finals_tree = run_multi_seed(n_seeds=5, graph_kind="tree", steps=200)

    print("\n\n" + "="*90)
    print("Single-seed run (CYCLE GRAPH, generality check):")
    print("="*90)
    log2 = run_experiment(graph_kind="cycle", seed=0)
    for name, data in log2.items():
        print(f"\n=== {name} (cycle) ===")
        for i, step in enumerate(data["step"]):
            print(f"step {step:4d} | loss {data['loss'][i]:.4f} | graph_align {data['graph_align'][i]:.4f}")

    print("\n\n" + "="*90)
    print("Multi-seed run (CYCLE GRAPH)")
    print("="*90)
    finals_cycle = run_multi_seed(n_seeds=5, graph_kind="cycle", steps=200)

    # Summary interpretation
    print("\n\n" + "="*90)
    print("INTERPRETATION")
    print("="*90)
    
    tree_e8_loss = np.mean(finals_tree["e8"]["loss"])
    tree_random_loss = np.mean(finals_tree["random"]["loss"])
    tree_e8_align = np.mean(finals_tree["e8"]["graph_align"])
    tree_random_align = np.mean(finals_tree["random"]["graph_align"])
    
    print(f"\nTREE GRAPH:")
    print(f"  E8 loss:       {tree_e8_loss:.4f}")
    print(f"  Random loss:   {tree_random_loss:.4f}  Δ = {tree_e8_loss - tree_random_loss:+.4f}")
    print(f"  E8 align:      {tree_e8_align:.4f}")
    print(f"  Random align:  {tree_random_align:.4f}  Δ = {tree_e8_align - tree_random_align:+.4f}")
    
    cycle_e8_loss = np.mean(finals_cycle["e8"]["loss"])
    cycle_random_loss = np.mean(finals_cycle["random"]["loss"])
    cycle_e8_align = np.mean(finals_cycle["e8"]["graph_align"])
    cycle_random_align = np.mean(finals_cycle["random"]["graph_align"])
    
    print(f"\nCYCLE GRAPH:")
    print(f"  E8 loss:       {cycle_e8_loss:.4f}")
    print(f"  Random loss:   {cycle_random_loss:.4f}  Δ = {cycle_e8_loss - cycle_random_loss:+.4f}")
    print(f"  E8 align:      {cycle_e8_align:.4f}")
    print(f"  Random align:  {cycle_random_align:.4f}  Δ = {cycle_e8_align - cycle_random_align:+.4f}")
    
    print("\n" + "="*90)
