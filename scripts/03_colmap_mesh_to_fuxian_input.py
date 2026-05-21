"""
Step 3: Convert COLMAP Poisson mesh → FuXian (117-paper) PLY input files.

Paper section 3.2.2 (Pre-Processing):
  1. Bounding box extraction + PCA axes
  2. Poisson surface reconstruction  ← already done by COLMAP poisson_mesher
  3. Poisson disk sampling on mesh surface → directed sample points

Output:
  data/town01_mesh.ply     – triangle mesh (for Embree ray casting in FuXian)
  data/town01_samples.ply  – Poisson-disk sampled points with outward normals

Quality scoring (Paper Section 3.2.3, Eq 1-5):
  If --dense-cloud is provided (path to COLMAP fused.ply), computes per-sample quality:
    gc = w_i * grad(si) · ni, with grad(si) approximated by kernel-smoothed input normals
    gs = sum_j w_pj * exp(-(1 - N_pj · ni)^2 / (1 - cos(theta_t))^2)
    g = gc * gs, min-max normalized to [0, 1]
  Quality is written into the samples PLY for the C++ code to use (type=1 reading).

Usage:
  python3 scripts/03_colmap_mesh_to_fuxian_input.py \
      --mesh   /data/town01/colmap_first_pass/dense/0/meshed-poisson.ply \
      --out-dir data/ \
      --num-samples 4000 \
      --seed 42
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# PLY reader: read binary or ASCII PLY mesh (vertices + faces)
# ---------------------------------------------------------------------------

def read_ply_mesh(path: Path):
    """
    Read a PLY file and return (vertices, faces).
    vertices: list of (x, y, z)  – positions only (normals computed per-face)
    faces:    list of (i, j, k)  – vertex indices
    Supports: binary_little_endian and ascii, float/double xyz, list faces.
    """
    with open(path, "rb") as f:
        # --- Parse header ---
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        fmt = "ascii"
        n_verts = 0
        n_faces = 0
        vert_props: list[tuple[str, str]] = []   # (type, name) in order
        in_vertex_element = False

        for line in header_lines:
            if line.startswith("format"):
                parts = line.split()
                fmt = parts[1]  # ascii / binary_little_endian / binary_big_endian
            elif line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
                in_vertex_element = True
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
                in_vertex_element = False
            elif line.startswith("element"):
                in_vertex_element = False
            elif line.startswith("property") and not line.startswith("property list"):
                if in_vertex_element:
                    parts = line.split()
                    vert_props.append((parts[1], parts[2]))  # (type, name)

        # --- Read data ---
        if fmt == "ascii":
            vertices, faces = _read_ply_ascii(f, n_verts, n_faces, vert_props)
        elif fmt == "binary_little_endian":
            vertices, faces = _read_ply_binary(f, n_verts, n_faces, vert_props,
                                                endian="<")
        elif fmt == "binary_big_endian":
            vertices, faces = _read_ply_binary(f, n_verts, n_faces, vert_props,
                                                endian=">")
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")

    print(f"[ply] read {len(vertices)} vertices, {len(faces)} faces from {path}",
          flush=True)
    return vertices, faces


def _read_ply_ascii(f, n_verts, n_faces, vert_props):
    names = [name for (_, name) in vert_props]
    xi = names.index("x")
    yi = names.index("y")
    zi = names.index("z")

    vertices = []
    for _ in range(n_verts):
        parts = f.readline().decode("ascii").split()
        vertices.append((float(parts[xi]), float(parts[yi]), float(parts[zi])))

    faces = []
    for _ in range(n_faces):
        parts = f.readline().decode("ascii").split()
        n = int(parts[0])
        if n == 3:
            faces.append((int(parts[1]), int(parts[2]), int(parts[3])))
        elif n == 4:  # quad → two triangles
            a, b, c, d = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            faces.append((a, b, c))
            faces.append((a, c, d))
    return vertices, faces


def _ply_dtype_size(type_str: str) -> tuple[str, int]:
    """Return (struct_char, byte_size) for a PLY property type."""
    MAP = {
        "float": ("f", 4), "float32": ("f", 4),
        "double": ("d", 8), "float64": ("d", 8),
        "int": ("i", 4), "int32": ("i", 4),
        "uint": ("I", 4), "uint32": ("I", 4),
        "short": ("h", 2), "int16": ("h", 2),
        "ushort": ("H", 2), "uint16": ("H", 2),
        "char": ("b", 1), "int8": ("b", 1),
        "uchar": ("B", 1), "uint8": ("B", 1),
    }
    return MAP.get(type_str, ("f", 4))


def _read_ply_binary(f, n_verts, n_faces, vert_props, endian):
    # Build per-property (struct_char, byte_size) using actual declared types.
    prop_info = [_ply_dtype_size(ptype) for (ptype, _) in vert_props]
    names = [name for (_, name) in vert_props]
    xi = names.index("x")
    yi = names.index("y")
    zi = names.index("z")
    row_bytes = sum(size for (_, size) in prop_info)

    vertices = []
    for _ in range(n_verts):
        raw = f.read(row_bytes)
        if len(raw) < row_bytes:
            break
        offset = 0
        vals = []
        for (sc, sz) in prop_info:
            (val,) = struct.unpack_from(f"{endian}{sc}", raw, offset)
            vals.append(val)
            offset += sz
        vertices.append((vals[xi], vals[yi], vals[zi]))

    faces = []
    for _ in range(n_faces):
        n_raw = f.read(1)
        if not n_raw:
            break
        n = struct.unpack("B", n_raw)[0]
        idx_fmt = f"{endian}{n}I"
        idx_bytes = n * 4
        idxs = struct.unpack(idx_fmt, f.read(idx_bytes))
        if n == 3:
            faces.append((idxs[0], idxs[1], idxs[2]))
        elif n == 4:
            faces.append((idxs[0], idxs[1], idxs[2]))
            faces.append((idxs[0], idxs[2], idxs[3]))

    return vertices, faces


# ---------------------------------------------------------------------------
# Face normal computation
# ---------------------------------------------------------------------------

def _face_normal_and_area(v0, v1, v2):
    """Return (nx, ny, nz, area) for triangle v0-v1-v2."""
    ax, ay, az = v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]
    bx, by, bz = v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]
    # cross product
    cx = ay*bz - az*by
    cy = az*bx - ax*bz
    cz = ax*by - ay*bx
    area = 0.5 * math.sqrt(cx*cx + cy*cy + cz*cz)
    length = math.sqrt(cx*cx + cy*cy + cz*cz)
    if length < 1e-12:
        return 0.0, 0.0, 1.0, 0.0
    return cx/length, cy/length, cz/length, area


# ---------------------------------------------------------------------------
# Poisson-disk-style sampling on mesh surface (area-weighted)
# ---------------------------------------------------------------------------

def poisson_disk_sample_mesh(
    vertices: list,
    faces: list,
    n_samples: int,
    rng: random.Random,
    min_dist: float = 0.0,
) -> list[tuple[float, float, float, float, float, float]]:
    """
    Sample n_samples points on the mesh surface with outward normals.
    Uses area-weighted random sampling with Poisson-disk rejection:
    candidate points closer than min_dist to any accepted sample are rejected.

    If min_dist is 0 or negative, it is auto-estimated from mesh area:
      min_dist = 0.7 * sqrt(total_area / n_samples)

    Returns list of (x, y, z, nx, ny, nz).
    """
    # 1. Compute face areas and normals
    face_data = []
    total_area = 0.0
    for (i, j, k) in faces:
        v0, v1, v2 = vertices[i], vertices[j], vertices[k]
        nx, ny, nz, area = _face_normal_and_area(v0, v1, v2)
        face_data.append((v0, v1, v2, nx, ny, nz, area))
        total_area += area

    if total_area < 1e-9:
        print("[WARN] Zero total area – mesh may be degenerate.", flush=True)
        return []

    if min_dist <= 0:
        min_dist = 0.7 * math.sqrt(total_area / n_samples)
    min_dist_sq = min_dist * min_dist
    print(f"[poisson] min_dist = {min_dist:.3f} m  (area = {total_area:.1f} m²)", flush=True)

    # 2. Build cumulative weight array for area-weighted sampling
    cum = []
    s = 0.0
    for fd in face_data:
        s += fd[6] / total_area
        cum.append(s)

    # 3. Spatial grid for fast neighbour lookup
    cell_size = min_dist
    grid: dict[tuple[int,int,int], list[int]] = {}

    def _grid_key(x, y, z):
        return (int(math.floor(x / cell_size)),
                int(math.floor(y / cell_size)),
                int(math.floor(z / cell_size)))

    def _too_close(px, py, pz) -> bool:
        gx, gy, gz = _grid_key(px, py, pz)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for idx in grid.get((gx+dx, gy+dy, gz+dz), ()):
                        sx, sy, sz = samples[idx][0], samples[idx][1], samples[idx][2]
                        if (px-sx)**2 + (py-sy)**2 + (pz-sz)**2 < min_dist_sq:
                            return True
        return False

    # 4. Sample with rejection
    samples: list[tuple] = []
    max_attempts = n_samples * 30
    attempts = 0
    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        r = rng.random()
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        v0, v1, v2, nx, ny, nz, _ = face_data[lo]

        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        w = 1.0 - u - v
        px = w*v0[0] + u*v1[0] + v*v2[0]
        py = w*v0[1] + u*v1[1] + v*v2[1]
        pz = w*v0[2] + u*v1[2] + v*v2[2]

        if _too_close(px, py, pz):
            continue

        idx = len(samples)
        samples.append((px, py, pz, nx, ny, nz))
        key = _grid_key(px, py, pz)
        grid.setdefault(key, []).append(idx)

    if len(samples) < n_samples:
        print(f"[poisson] Only placed {len(samples)}/{n_samples} samples "
              f"(min_dist={min_dist:.3f} m may be too large for this mesh).", flush=True)

    return samples


# ---------------------------------------------------------------------------
# Dense cloud reader + quality scoring (Paper Section 3.2.3, Eq 1-5)
# ---------------------------------------------------------------------------

def read_dense_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read COLMAP fused.ply (binary LE, x,y,z,nx,ny,nz,r,g,b per vertex).
    Returns (positions Nx3 float32, normals Nx3 float32)."""
    with open(path, "rb") as f:
        # Parse header
        n_verts = 0
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            if line == "end_header":
                break

        # COLMAP fused.ply layout: 6 floats (xyz, nxnynz) + 3 bytes (rgb) = 27 bytes
        row_bytes = 6 * 4 + 3
        raw = f.read(n_verts * row_bytes)

    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    data = np.frombuffer(raw, dtype=dt, count=n_verts)
    positions = np.column_stack([data["x"], data["y"], data["z"]])
    normals = np.column_stack([data["nx"], data["ny"], data["nz"]])
    print(f"[quality] Read {n_verts} dense cloud points from {path}", flush=True)
    return positions, normals


def compute_quality_scores(
    samples: list[tuple[float, float, float, float, float, float]],
    dense_pos: np.ndarray,
    dense_normals: np.ndarray,
    radius: float = 3.0,
    theta_t_deg: float = 60.0,
) -> list[float]:
    """Compute quality g = gc * gs for each sample point (Paper Eq 1-5).

    The paper says grad(si) is calculated during Poisson reconstruction but does
    not expose that field. We approximate it with the kernel-smoothed input
    normal vector field from nearby dense-cloud points.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(dense_pos)
    sample_pos = np.array([(s[0], s[1], s[2]) for s in samples], dtype=np.float32)
    sample_normals = np.array([(s[3], s[4], s[5]) for s in samples], dtype=np.float64)

    dense_normals = np.asarray(dense_normals, dtype=np.float64)
    dense_norm = np.linalg.norm(dense_normals, axis=1, keepdims=True)
    dense_normals = dense_normals / np.maximum(dense_norm, 1e-12)

    sample_norm = np.linalg.norm(sample_normals, axis=1, keepdims=True)
    sample_normals = sample_normals / np.maximum(sample_norm, 1e-12)

    # Batch query: all neighbors within radius for each sample
    neighbor_lists = tree.query_ball_point(sample_pos, radius)

    gc_values = np.zeros(len(samples), dtype=np.float64)
    gs_values = np.zeros(len(samples), dtype=np.float64)
    n_max = max((len(nbr_idx) for nbr_idx in neighbor_lists), default=0)
    theta_t = math.radians(theta_t_deg)
    theta_denom = max((1.0 - math.cos(theta_t)) ** 2, 1e-12)

    for i, nbr_idx in enumerate(neighbor_lists):
        if not nbr_idx or n_max <= 0:
            continue

        nbr_pos = dense_pos[nbr_idx]
        nbr_nrm = dense_normals[nbr_idx]
        n_i = sample_normals[i]
        d2 = np.sum((nbr_pos - sample_pos[i]) ** 2, axis=1)
        d = np.sqrt(d2)
        w_pj = np.exp(-d / max(radius, 1e-12))

        grad = np.sum(w_pj[:, None] * nbr_nrm, axis=0)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm

        count = len(nbr_idx)
        w_i = (1.0 - ((count - n_max) / n_max)) ** 2
        gc_values[i] = w_i * max(0.0, float(np.dot(grad, n_i)))

        dots = np.clip(nbr_nrm @ n_i, -1.0, 1.0)
        gs_values[i] = float(np.sum(w_pj * np.exp(-((1.0 - dots) ** 2) / theta_denom)))

    raw_quality = gc_values * gs_values
    q_min = raw_quality.min() if len(raw_quality) else 0.0
    q_max = raw_quality.max() if len(raw_quality) else 0.0
    if q_max > q_min:
        quality = (raw_quality - q_min) / (q_max - q_min)
    else:
        quality = np.zeros_like(raw_quality)

    print(f"[quality] neighbor count range: [{min((len(n) for n in neighbor_lists), default=0)}, {n_max}]", flush=True)
    print(f"[quality] gc range: [{gc_values.min():.3f}, {gc_values.max():.3f}]", flush=True)
    print(f"[quality] gs range: [{gs_values.min():.3f}, {gs_values.max():.3f}]", flush=True)
    print(f"[quality] raw g range: [{raw_quality.min():.3f}, {raw_quality.max():.3f}]", flush=True)
    print(f"[quality] final quality range: [{quality.min():.3f}, {quality.max():.3f}]", flush=True)

    return quality.tolist()


# ---------------------------------------------------------------------------
# PLY writers (same format as original xuexiao data)
# ---------------------------------------------------------------------------

def write_mesh_ply(path: Path, vertices: list, faces: list) -> None:
    """Write binary little-endian PLY mesh with per-vertex normals (float x y z nx ny nz)."""
    # Compute per-vertex normals by averaging adjacent face normals
    v_normals = [[0.0, 0.0, 0.0] for _ in vertices]
    for (i, j, k) in faces:
        v0, v1, v2 = vertices[i], vertices[j], vertices[k]
        nx, ny, nz, _ = _face_normal_and_area(v0, v1, v2)
        for idx in (i, j, k):
            v_normals[idx][0] += nx
            v_normals[idx][1] += ny
            v_normals[idx][2] += nz

    # Normalize
    norm_verts = []
    for (x, y, z), (nx, ny, nz) in zip(vertices, v_normals):
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length < 1e-9:
            nx, ny, nz = 0.0, 0.0, 1.0
        else:
            nx, ny, nz = nx/length, ny/length, nz/length
        norm_verts.append((x, y, z, nx, ny, nz))

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Generated from COLMAP Poisson mesh for 117-paper FuXian\n"
        f"element vertex {len(norm_verts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        f"element face {len(faces)}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for v in norm_verts:
            f.write(struct.pack("<6f", *v))
        for tri in faces:
            f.write(struct.pack("<B3I", 3, tri[0], tri[1], tri[2]))

    print(f"[write] mesh → {path}  ({len(norm_verts)} verts, {len(faces)} faces)",
          flush=True)


def write_sample_ply(path: Path, samples: list,
                     qualities: list[float] | None = None) -> None:
    """Write ASCII PLY sample-points. If qualities provided, includes quality field (type=1)."""
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Generated from COLMAP Poisson mesh via Poisson disk sampling\n")
        f.write(f"element vertex {len(samples)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        if qualities is not None:
            f.write("property float quality\n")
        f.write("end_header\n")
        for i, (x, y, z, nx, ny, nz) in enumerate(samples):
            line = f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f}"
            if qualities is not None:
                line += f" {qualities[i]:.6f}"
            f.write(line + "\n")

    tag = "with quality" if qualities is not None else "no quality"
    print(f"[write] samples → {path}  ({len(samples)} points, {tag})", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True, type=Path,
                    help="COLMAP Poisson mesh PLY (meshed-poisson.ply)")
    ap.add_argument("--dense-cloud", type=Path, default=None,
                    help="COLMAP fused.ply for quality scoring (gc, gs)")
    ap.add_argument("--quality-radius", type=float, default=3.0,
                    help="Neighbourhood radius R for Paper Eq 2/3 quality scoring")
    ap.add_argument("--theta-t-deg", type=float, default=60.0,
                    help="Angle threshold theta_t in degrees for Paper Eq 2 quality scoring")
    ap.add_argument("--out-dir", default="data", type=Path,
                    help="Output directory for FuXian PLY files (default: data/)")
    ap.add_argument("--num-samples", type=int, default=4000,
                    help="Number of Poisson-disk sample points (default: 4000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="town01",
                    help="Map tag for output filenames (default: town01)")
    args = ap.parse_args()

    print(f"[step3] Reading coarse Poisson mesh: {args.mesh}", flush=True)
    vertices, faces = read_ply_mesh(args.mesh)

    if not vertices:
        print("[ERROR] Empty mesh. Did poisson_mesher complete successfully?", flush=True)
        sys.exit(1)

    # Print bounding box for reference
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    print(f"[step3] Mesh bounds:", flush=True)
    print(f"  X: {min(xs):.2f} – {max(xs):.2f}", flush=True)
    print(f"  Y: {min(ys):.2f} – {max(ys):.2f}", flush=True)
    print(f"  Z: {min(zs):.2f} – {max(zs):.2f}", flush=True)

    print(f"[step3] Poisson-disk sampling {args.num_samples} points on mesh...",
          flush=True)
    rng = random.Random(args.seed)
    samples = poisson_disk_sample_mesh(vertices, faces, args.num_samples, rng)
    print(f"[step3] Sampled {len(samples)} points", flush=True)

    # Filter outlier samples (Poisson mesh artifacts far from real geometry)
    z_vals = sorted(s[2] for s in samples)
    z_p5  = z_vals[max(0, len(z_vals) * 5 // 100)]
    z_p95 = z_vals[min(len(z_vals) - 1, len(z_vals) * 95 // 100)]
    z_iqr = z_p95 - z_p5
    z_lo = z_p5 - 1.5 * z_iqr
    z_hi = z_p95 + 1.5 * z_iqr
    before = len(samples)
    samples = [s for s in samples if z_lo <= s[2] <= z_hi]
    print(f"[step3] Z filter [{z_lo:.1f}, {z_hi:.1f}]: kept {len(samples)}/{before} samples",
          flush=True)

    # Quality scoring from coarse dense cloud (Paper Section 3.2.3)
    qualities = None
    if args.dense_cloud and args.dense_cloud.exists():
        print(f"[step3] Computing quality scores from {args.dense_cloud}...", flush=True)
        dense_pos, dense_nrm = read_dense_cloud(args.dense_cloud)
        qualities = compute_quality_scores(
            samples,
            dense_pos,
            dense_nrm,
            radius=args.quality_radius,
            theta_t_deg=args.theta_t_deg,
        )
        q_nonzero = sum(1 for q in qualities if q > 0.01)
        print(f"[step3] Quality: {q_nonzero}/{len(qualities)} samples with q > 0.01",
              flush=True)
    elif args.dense_cloud:
        print(f"[WARN] Dense cloud not found: {args.dense_cloud}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_mesh_ply(args.out_dir / f"{args.tag}_mesh.ply", vertices, faces)
    write_sample_ply(args.out_dir / f"{args.tag}_samples.ply", samples, qualities)

    print("\n[step3] Done. Next step:", flush=True)
    print("  ./scripts/04_run_fuxian.sh", flush=True)


if __name__ == "__main__":
    main()
