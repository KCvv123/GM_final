"""
ACO-based TSP path optimisation for aerial viewpoint trajectories.

Paper: Sampling-Based Path Planning for High-Quality Aerial 3D
       Reconstruction of Urban Scenes  (Section 3.4)

Cost function (eq 9-11):
    c_t(i,j) = normalised Euclidean distance
    c_r(i,j) = 0.5 * (1 - cos(angle between view directions))
    c(i,j)   = 0.5 * c_t + 0.5 * c_r

Usage:
    python3 aco_tsp.py [--input  output/20230130/xuexiao2.txt]
                       [--out-raw      output/20230130/xuexiao2-aco.txt]
                       [--out-smooth   output/20230130/xuexiao2-aco-smooth.txt]
                       [--out-smith    output/trajectory/smithFormatPathXueXiao-2-aco.txt]
                       [--ants 20] [--iters 100] [--alpha 1] [--beta 2] [--rho 0.1]
"""

import argparse
import math
import os
import time

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_viewpoints(path: str):
    """Return (pos, dirs) as (N,3) float64 arrays."""
    data = np.loadtxt(path, delimiter=",")
    return data[:, :3].copy(), data[:, 3:].copy()


def save_raw(path: str, pos: np.ndarray, dirs: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = np.hstack([pos, dirs])
    np.savetxt(path, rows, delimiter=",", fmt="%.10g")
    print(f"  raw viewpoints -> {path}  ({len(pos)} lines)")


def save_smith(path: str, pos: np.ndarray, dirs: np.ndarray):
    """Replicate the Smith-format output from io_utils.cpp::viewPointsToSmithPath."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i, (p, d) in enumerate(zip(pos, dirs)):
            px, py, pz = p
            nx, ny, nz = d
            dxy = math.sqrt(nx * nx + ny * ny)
            pitch = math.atan2(nz, dxy) * 180.0 / math.pi
            yaw   = math.atan2(ny, nx)  * 180.0 / math.pi
            yaw   = 90.0 - yaw
            f.write(f"{i:04d},{-px*100:.8g},{py*100:.8g},{pz*100:.8g}"
                    f",{-pitch:.8g},0,{yaw:.8g}\n")
    print(f"  Smith format   -> {path}  ({len(pos)} lines)")


# ---------------------------------------------------------------------------
# Cost matrix
# ---------------------------------------------------------------------------

def build_cost_matrix(pos: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """
    Build N×N cost matrix.  Diagonal is set to inf.
    Uses scipy cdist if available, falls back to pure numpy.
    """
    n = len(pos)
    print(f"  Building {n}×{n} cost matrix...", flush=True)
    t0 = time.time()

    # --- travel cost c_t ---
    try:
        from scipy.spatial.distance import cdist
        dist = cdist(pos, pos, "euclidean")
    except ImportError:
        diff = pos[:, None, :] - pos[None, :, :]   # (n,n,3)
        dist = np.linalg.norm(diff, axis=2)

    mask = dist > 0
    b_min = dist[mask].min() if mask.any() else 0.0
    b_max = dist.max()
    denom = b_max - b_min
    ct = (dist - b_min) / (denom if denom > 1e-10 else 1.0)

    # --- rotation cost c_r ---
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    d = dirs / norms
    cos_sim = np.clip(d @ d.T, -1.0, 1.0)
    cr = 0.5 * (1.0 - cos_sim)

    cost = 0.5 * ct + 0.5 * cr
    np.fill_diagonal(cost, np.inf)

    print(f"  Cost matrix done in {time.time()-t0:.1f}s  "
          f"(c range [{cost[cost < np.inf].min():.4f}, {cost[cost < np.inf].max():.4f}])")
    return cost


# ---------------------------------------------------------------------------
# Nearest-neighbour initialisation (fast warm start for ACO)
# ---------------------------------------------------------------------------

def nearest_neighbour_tour(cost: np.ndarray) -> list:
    n = len(cost)
    visited = np.zeros(n, dtype=bool)
    c = np.zeros(n)          # working copy of current row
    tour = []
    cur = 0
    visited[cur] = True
    tour.append(cur)
    for _ in range(n - 1):
        c[:] = cost[cur]
        c[visited] = np.inf
        nxt = int(np.argmin(c))
        visited[nxt] = True
        tour.append(nxt)
        cur = nxt
    return tour


def tour_cost(tour: list, cost: np.ndarray) -> float:
    return sum(cost[tour[i], tour[i + 1]] for i in range(len(tour) - 1))


# ---------------------------------------------------------------------------
# Cubic Bezier smoothing
# ---------------------------------------------------------------------------

def _normalize_rows(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-10)


def smooth_cubic_bezier(pos: np.ndarray,
                        dirs: np.ndarray,
                        samples_per_segment: int = 4,
                        tension: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth an open ordered path with cubic Bezier segments.

    The path still passes through every selected viewpoint. Extra intermediate
    points approximate the continuous cubic Bezier trajectory mentioned in
    Paper Section 3.4. Directions are linearly interpolated and normalized.
    """
    n = len(pos)
    if n <= 2 or samples_per_segment <= 1:
        return pos.copy(), dirs.copy()

    samples_per_segment = max(2, samples_per_segment)
    out_pos = [pos[0]]
    out_dirs = [dirs[0]]

    for i in range(n - 1):
        p0 = pos[i]
        p3 = pos[i + 1]
        prev_p = pos[i - 1] if i > 0 else p0
        next_p = pos[i + 2] if i + 2 < n else p3
        seg_len = np.linalg.norm(p3 - p0)

        t0 = p3 - prev_p
        t1 = next_p - p0
        t0_norm = np.linalg.norm(t0)
        t1_norm = np.linalg.norm(t1)
        t0 = t0 / max(t0_norm, 1e-10)
        t1 = t1 / max(t1_norm, 1e-10)

        p1 = p0 + t0 * seg_len * tension
        p2 = p3 - t1 * seg_len * tension

        for step in range(1, samples_per_segment + 1):
            u = step / samples_per_segment
            omt = 1.0 - u
            p = (omt ** 3) * p0 + 3.0 * (omt ** 2) * u * p1 \
                + 3.0 * omt * (u ** 2) * p2 + (u ** 3) * p3
            d = (1.0 - u) * dirs[i] + u * dirs[i + 1]
            out_pos.append(p)
            out_dirs.append(d)

    return np.asarray(out_pos), _normalize_rows(np.asarray(out_dirs))


# ---------------------------------------------------------------------------
# ACO
# ---------------------------------------------------------------------------

def aco_tsp(cost: np.ndarray,
            n_ants:  int   = 20,
            n_iter:  int   = 100,
            alpha:   float = 1.0,
            beta:    float = 2.0,
            rho:     float = 0.1) -> tuple:
    """
    Ant Colony Optimisation for TSP.
    Returns (best_tour, best_cost).
    """
    n = len(cost)

    # Heuristic: η = 1 / cost  (set to 0 where cost is inf or 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where((cost > 0) & np.isfinite(cost), 1.0 / cost, 0.0)

    # Pheromone initialised to uniform small value
    tau = np.ones((n, n), dtype=np.float64) * 0.1

    # Warm-start best tour with nearest-neighbour heuristic
    nn_tour  = nearest_neighbour_tour(cost)
    nn_cost  = tour_cost(nn_tour, cost)
    best_tour = nn_tour[:]
    best_cost = nn_cost
    print(f"  Nearest-neighbour warm-start cost: {nn_cost:.4f}")

    t0 = time.time()
    for it in range(n_iter):
        iter_tours = []
        iter_costs = []

        for _ in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            start   = np.random.randint(n)
            tour    = np.empty(n, dtype=np.int32)
            tour[0] = start
            visited[start] = True

            for step in range(1, n):
                cur = tour[step - 1]
                # attractiveness = tau^alpha * eta^beta  (vectorised)
                attract = (tau[cur] ** alpha) * (eta[cur] ** beta)
                attract[visited] = 0.0
                total = attract.sum()
                if total < 1e-300:
                    cands = np.where(~visited)[0]
                    nxt = int(np.random.choice(cands))
                else:
                    probs = attract / total
                    nxt   = int(np.random.choice(n, p=probs))
                tour[step]  = nxt
                visited[nxt] = True

            tc = tour_cost(tour.tolist(), cost)
            iter_tours.append(tour)
            iter_costs.append(tc)

            if tc < best_cost:
                best_cost = tc
                best_tour = tour.tolist()

        # Pheromone evaporation
        tau *= (1.0 - rho)
        tau  = np.maximum(tau, 1e-10)

        # Pheromone deposit  (all ants + elite deposit for best tour)
        for tour_arr, tc in zip(iter_tours, iter_costs):
            if np.isfinite(tc) and tc > 0:
                dep = 1.0 / tc
                for i in range(n - 1):
                    tau[tour_arr[i], tour_arr[i + 1]] += dep
                    tau[tour_arr[i + 1], tour_arr[i]] += dep
        # Elite: extra deposit on global best
        if np.isfinite(best_cost) and best_cost > 0:
            elite_dep = n_ants / best_cost
            for i in range(n - 1):
                tau[best_tour[i], best_tour[i + 1]] += elite_dep
                tau[best_tour[i + 1], best_tour[i]] += elite_dep

        if (it + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  iter {it+1:3d}/{n_iter}  best={best_cost:.4f}  "
                  f"elapsed={elapsed:.0f}s", flush=True)

    return best_tour, best_cost


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ACO TSP for viewpoint ordering")
    parser.add_argument("--input",     default="output/20230130/xuexiao2.txt")
    parser.add_argument("--out-raw",   default="output/20230130/xuexiao2-aco.txt")
    parser.add_argument("--out-smooth", default="output/20230130/xuexiao2-aco-smooth.txt")
    parser.add_argument("--out-smith", default="output/trajectory/smithFormatPathXueXiao-2-aco.txt")
    parser.add_argument("--out-smooth-smith", default="output/trajectory/smithFormatPathXueXiao-2-aco-smooth.txt")
    parser.add_argument("--ants",  type=int,   default=20)
    parser.add_argument("--iters", type=int,   default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta",  type=float, default=2.0)
    parser.add_argument("--rho",   type=float, default=0.1)
    parser.add_argument("--smooth-samples", type=int, default=4,
                        help="Bezier samples per segment, including the segment endpoint")
    parser.add_argument("--smooth-tension", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)

    print(f"\n=== ACO TSP  ({args.ants} ants, {args.iters} iters) ===")
    print(f"Input: {args.input}")

    # 1. Load
    pos, dirs = load_viewpoints(args.input)
    n = len(pos)
    print(f"Loaded {n} viewpoints")

    # 2. Cost matrix
    cost = build_cost_matrix(pos, dirs)

    # 3. Baseline: original order cost
    original_cost = tour_cost(list(range(n)), cost)
    print(f"  Original (greedy) order cost : {original_cost:.4f}")

    # 4. ACO
    print("\nRunning ACO...", flush=True)
    t_start = time.time()
    best_tour, best_cost = aco_tsp(cost,
                                   n_ants=args.ants,
                                   n_iter=args.iters,
                                   alpha=args.alpha,
                                   beta=args.beta,
                                   rho=args.rho)
    elapsed = time.time() - t_start

    improvement = (original_cost - best_cost) / original_cost * 100
    print(f"\n=== Results ===")
    print(f"  ACO best cost     : {best_cost:.4f}")
    print(f"  Original cost     : {original_cost:.4f}")
    print(f"  Improvement       : {improvement:.1f}%")
    print(f"  Total time        : {elapsed:.1f}s")

    # 5. Reorder and save
    idx = np.array(best_tour, dtype=int)
    pos_ordered  = pos[idx]
    dirs_ordered = dirs[idx]
    pos_smooth, dirs_smooth = smooth_cubic_bezier(
        pos_ordered,
        dirs_ordered,
        samples_per_segment=args.smooth_samples,
        tension=args.smooth_tension,
    )

    print("\nSaving outputs...")
    save_raw(args.out_raw, pos_ordered, dirs_ordered)
    save_smith(args.out_smith, pos_ordered, dirs_ordered)
    save_raw(args.out_smooth, pos_smooth, dirs_smooth)
    save_smith(args.out_smooth_smith, pos_smooth, dirs_smooth)
    print(f"  Bezier samples   : {len(pos_smooth)} points "
          f"({args.smooth_samples} per segment)")

    print("\nDone.")


if __name__ == "__main__":
    main()
