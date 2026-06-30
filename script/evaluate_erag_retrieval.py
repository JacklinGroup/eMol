#!/usr/bin/env python3
"""
eRAG retrieval evaluation: use eRAG retriever to sample pairs for each metric.
For each QM9 query, retrieve best reference via eRAG pipeline,
compute the specified metric similarity, and fill bins.
"""
import csv, json, os, pickle, time
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import (MACCSkeys, DataStructs, rdFMCS,
                         rdFingerprintGenerator)
from scipy.spatial.distance import cdist
from tqdm import tqdm

rdBase.BlockLogs()
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)
os.close(_devnull)

# ─── Config ──────────────────────────────────────────────────────────────
SDF_PATH = "/data/sunyang/qm9/data/raw/gdb9.sdf"
POOL_PATH = Path.home() / "emol" / "output" / "qm9_pool_5000.pkl"
SAMPLED_RAG_DIR = "/data/sunyang/SampledRagData"
OUTPUT_DIR = Path.home() / "emol" / "output"
CACHE_DIR = Path.home() / "emol" / "cache"
INDEX_PATH = CACHE_DIR / "erag_index.pkl"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NUM_QUERIES = 5000
TARGET_PER_BIN = 5000
RANDOM_SEED = 42
NUM_WORKERS = cpu_count()

BIN_LABELS = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
BIN_EDGES = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.001)]

METRICS = ["maccs", "ecfp4", "mcs", "local-env", "radial-ed"]


def get_bin_label(v):
    for (lo, hi), lbl in zip(BIN_EDGES, BIN_LABELS):
        if lo <= v < hi:
            return lbl
    return None


def smi_to_heavy(smi):
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    mh = Chem.RemoveHs(m)
    if mh.GetNumAtoms() < 3:
        return None
    try:
        Chem.GetSymmSSSR(mh)
    except Exception:
        pass
    return mh


# ═══════════════════════════════════════════════════════════════════════════
#  Similarity Metrics
# ═══════════════════════════════════════════════════════════════════════════

def mol_to_fp(mol, metric):
    if mol is None:
        return None
    if metric == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    elif metric == "ecfp4":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return gen.GetFingerprint(mol)
    return None


def fp_similarity(fp_a, fp_b):
    if fp_a is None or fp_b is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def mcs_similarity(mol_i, mol_j, timeout=5):
    if mol_i is None or mol_j is None:
        return None
    res = rdFMCS.FindMCS(
        [mol_i, mol_j],
        timeout=timeout,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
    )
    if res.canceled or res.numAtoms == 0:
        return 0.0
    n_total = max(mol_i.GetNumAtoms(), mol_j.GetNumAtoms())
    return res.numAtoms / n_total


# ── Local-Env descriptor ──────────────────────────────────────────────────
def local_density_features(atom_coords, electron_coords, density, radius=2.0):
    eps = 1e-12
    distances = cdist(atom_coords, electron_coords, metric="euclidean")
    weights = np.exp(-((distances / radius) ** 2))
    w_norm = weights / np.maximum(weights.sum(axis=1, keepdims=True), eps)
    dm = density[None, :]

    mean_dens = (w_norm * dm).sum(axis=1)
    var_dens = (w_norm * (dm - mean_dens[:, None]) ** 2).sum(axis=1)
    std_dens = np.sqrt(np.maximum(var_dens, 0))
    l1 = (w_norm * np.abs(dm)).sum(axis=1)
    l2 = np.sqrt((w_norm * dm ** 2).sum(axis=1))
    wsum = (weights * dm).sum(axis=1)
    pos_mask = (dm > 0).astype(float)
    wmass = (weights * dm * pos_mask).sum(axis=1)
    mean_dist = (w_norm * distances).sum(axis=1)
    var_dist = (w_norm * (distances - mean_dist[:, None]) ** 2).sum(axis=1)
    std_dist = np.sqrt(np.maximum(var_dist, 0))
    cov = (w_norm * (dm - mean_dens[:, None]) * (distances - mean_dist[:, None])).sum(axis=1)
    w_pos = np.maximum(w_norm, eps)
    w_pos = w_pos / w_pos.sum(axis=1, keepdims=True)
    entropy = -(w_pos * np.log(w_pos)).sum(axis=1)

    return np.stack([mean_dens, std_dens, l1, l2, wsum,
                     wmass, mean_dist, std_dist, cov, entropy], axis=1)


def histogram_pool(features, bins=20):
    D = features.shape[1]
    pooled = []
    for d in range(D):
        col = features[:, d]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-10:
            pooled.append(np.zeros(bins))
        else:
            h, _ = np.histogram(col, bins=bins, range=(lo, hi + 1e-10), density=True)
            pooled.append(h)
    return np.concatenate(pooled)


def local_env_similarity(coords_i, elec_i, dens_i, coords_j, elec_j, dens_j):
    feat_i = local_density_features(coords_i, elec_i, dens_i)
    feat_j = local_density_features(coords_j, elec_j, dens_j)
    vec_i = histogram_pool(feat_i)
    vec_j = histogram_pool(feat_j)
    ni, nj = np.linalg.norm(vec_i), np.linalg.norm(vec_j)
    if ni < 1e-10 or nj < 1e-10:
        return 0.0
    return float(np.dot(vec_i, vec_j) / (ni * nj))


# ── Radial-ED descriptor ──────────────────────────────────────────────────
def nearest_atom_radial_hist(atom_coords, electron_coords, density,
                              n_bins=32, max_radius=4.0):
    eps = 1e-12
    distances = cdist(electron_coords, atom_coords, metric="euclidean")
    owner = np.argmin(distances, axis=1)
    owner_dist = distances[np.arange(len(owner)), owner]
    edges = np.linspace(0, max_radius, n_bins + 1)
    N_atoms = len(atom_coords)
    hist = np.zeros((N_atoms, n_bins))
    for a in range(N_atoms):
        mask = owner == a
        if mask.any():
            hist[a], _ = np.histogram(owner_dist[mask], bins=edges,
                                      weights=density[mask])
    row_sum = hist.sum(axis=1, keepdims=True)
    return hist / np.maximum(row_sum, eps)


def radial_ed_similarity(coords_i, elec_i, dens_i, coords_j, elec_j, dens_j):
    hist_i = nearest_atom_radial_hist(coords_i, elec_i, dens_i)
    hist_j = nearest_atom_radial_hist(coords_j, elec_j, dens_j)
    vec_i = histogram_pool(hist_i)
    vec_j = histogram_pool(hist_j)
    ni, nj = np.linalg.norm(vec_i), np.linalg.norm(vec_j)
    if ni < 1e-10 or nj < 1e-10:
        return 0.0
    return float(np.dot(vec_i, vec_j) / (ni * nj))


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_queries():
    """Load 5000 query molecules from GDB-9."""
    with open(POOL_PATH, "rb") as f:
        pool = sorted(pickle.load(f)[:NUM_QUERIES])

    suppl = Chem.SDMolSupplier(SDF_PATH, sanitize=True)
    q_records = []
    for idx, mol in enumerate(suppl):
        if len(q_records) >= NUM_QUERIES:
            break
        if idx not in pool or mol is None:
            continue
        hm = Chem.RemoveHs(mol)
        if hm.GetNumAtoms() < 3:
            continue
        try:
            Chem.GetSymmSSSR(hm)
        except Exception:
            pass
        coords = []
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        coords = np.array(coords, dtype=np.float64)
        z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
        q_records.append({"mol": hm, "coords": coords, "z": z, "idx": idx})
    return q_records


# ═══════════════════════════════════════════════════════════════════════════
#  eRAG Retrieval
# ═══════════════════════════════════════════════════════════════════════════

def build_erag_index():
    """Build or load the eRAG index."""
    if INDEX_PATH.exists():
        print("  Loading existing index...")
        from erag import load_index
        return load_index(str(INDEX_PATH))

    print("  Building index from SampledRagData...")
    from erag import build_index
    index = build_index(db_dir=SAMPLED_RAG_DIR, output_path=str(INDEX_PATH))
    return index


def retrieve_for_query(retriever, query_coords, query_z, topk=1):
    """Retrieve best reference for a query using eRAG pipeline."""
    try:
        result = retriever.retrieve_best_reference(query_coords, query_z)
        return result
    except Exception as e:
        return None


def compute_metric_similarity(query_record, ref_record, metric, retriever_result=None):
    """Compute similarity between query and reference using the specified metric."""
    if metric == "maccs":
        fp_q = mol_to_fp(query_record["mol"], "maccs")
        fp_r = mol_to_fp(ref_record["mol"], "maccs")
        return fp_similarity(fp_q, fp_r)
    elif metric == "ecfp4":
        fp_q = mol_to_fp(query_record["mol"], "ecfp4")
        fp_r = mol_to_fp(ref_record["mol"], "ecfp4")
        return fp_similarity(fp_q, fp_r)
    elif metric == "mcs":
        return mcs_similarity(query_record["mol"], ref_record["mol"])
    elif metric == "local-env":
        return local_env_similarity(
            query_record["coords"], query_record["ed_coords"], query_record["ed_density"],
            ref_record["coords"], ref_record["ed_coords"], ref_record["ed_density"])
    elif metric == "radial-ed":
        return radial_ed_similarity(
            query_record["coords"], query_record["ed_coords"], query_record["ed_density"],
            ref_record["coords"], ref_record["ed_coords"], ref_record["ed_density"])
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="eRAG retrieval evaluation")
    parser.add_argument("--metric", choices=METRICS, required=True,
                        help="Binning metric")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    metric = args.metric

    np.random.seed(RANDOM_SEED)
    t0 = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV = OUTPUT_DIR / f"erag_retrieval_{metric}.csv"

    print("=" * 60)
    print(f"eRAG Retrieval — metric: {metric}, workers: {args.workers}")
    print("=" * 60)

    # ── Step 1: Load queries ──────────────────────────────────────────────
    print("\nStep 1: Loading queries from GDB-9")
    t1 = time.time()
    q_records = load_queries()
    n_q = len(q_records)
    print(f"  {n_q} queries ({time.time()-t1:.1f}s)")

    # ── Step 2: Build/load eRAG index ────────────────────────────────────
    print("\nStep 2: Building eRAG index")
    t1 = time.time()
    index = build_erag_index()
    print(f"  Index loaded ({time.time()-t1:.1f}s)")

    # ── Step 3: Initialize retriever ─────────────────────────────────────
    print("\nStep 3: Initializing eRAG retriever")
    from erag import ElectronRetriever
    retriever = ElectronRetriever(
        electron_db_path=SAMPLED_RAG_DIR,
        index_path=str(INDEX_PATH),
        coarse_topk=32,
        exact_topk=8,
    )
    print("  Retriever ready")

    # ── Step 4: Retrieve and compute similarities ────────────────────────
    print(f"\nStep 4: Retrieving references for {n_q} queries")
    bins = defaultdict(list)
    all_results = []

    for qi, q_rec in enumerate(tqdm(q_records, desc="  Retrieving")):
        # Retrieve best reference via eRAG
        result = retrieve_for_query(retriever, q_rec["coords"], q_rec["z"])
        if result is None:
            continue

        # Load reference molecule from index
        ref_locator = result.get("source_pkl")
        ref_mol_id = result.get("reference", {}).get("mol_id")
        if ref_mol_id is None:
            continue

        # Get reference record from index
        bucket_key = result["candidate_spec"].bucket_key
        bucket_index = result["candidate_spec"].bucket_index
        bucket_payload = index["buckets"][bucket_key]
        shard_idx = int(bucket_payload["shard_idx"][bucket_index])
        record_key = bucket_payload["record_key"][bucket_index]

        # Load the full reference record
        from erag import get_record_from_payload
        payload = retriever.get_payload(shard_idx)
        record = get_record_from_payload(payload, record_key)

        if record is None or "mol" not in record or "electronic_density" not in record:
            continue

        # Extract reference data
        ref_mol_payload = record["mol"]
        ref_z = np.asarray(ref_mol_payload.get("x"), np.int64)
        ref_coords = np.asarray(ref_mol_payload.get("coords"), np.float64)
        ref_ed = record["electronic_density"]
        ref_ed_coords = np.asarray(ref_ed.get("coords"), np.float64)
        ref_ed_density = np.asarray(ref_ed.get("density"), np.float64)

        # Build reference molecule
        ref_mol = Chem.RWMol()
        for an in ref_z:
            ref_mol.AddAtom(Chem.Atom(int(an)))
        ref_mol = ref_mol.GetMol()
        conf = Chem.Conformer(len(ref_z))
        for ci, coord in enumerate(ref_coords):
            conf.SetAtomPosition(ci, coord.tolist())
        ref_mol.AddConformer(conf, assignId=True)
        from rdkit.Chem import rdDetermineBonds
        rdDetermineBonds.DetermineConnectivity(ref_mol)
        ref_mol.UpdatePropertyCache()
        ref_hm = Chem.RemoveHs(ref_mol)

        ref_record = {
            "mol": ref_hm,
            "coords": ref_coords,
            "ed_coords": ref_ed_coords,
            "ed_density": ref_ed_density,
        }

        # Compute metric similarity
        sim = compute_metric_similarity(q_rec, ref_record, metric, result)
        if sim is None:
            continue

        # Get eRAG similarity score
        erag_sim = result.get("similarity", 0.0)

        # Add to bin
        bl = get_bin_label(sim)
        if bl is not None:
            bins[bl].append((qi, ref_mol_id, sim, erag_sim))

    # ── Step 5: Trim to TARGET_PER_BIN ───────────────────────────────────
    print(f"\nStep 5: Trimming to {TARGET_PER_BIN} per bin")
    final_results = []
    bin_counts = {}
    for lbl in BIN_LABELS:
        candidates = bins.get(lbl, [])
        candidates.sort(key=lambda x: -x[2])  # Sort by similarity descending
        selected = candidates[:TARGET_PER_BIN]
        bin_counts[lbl] = len(selected)
        final_results.extend([(q, r, s, e, lbl) for q, r, s, e in selected])

    # ── Save ───────────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w") as f:
        w = csv.writer(f)
        w.writerow(["query_idx", "ref_mol_id", "metric_similarity", "erag_similarity", "bin"])
        for q_idx, ref_id, sim, erag_sim, bl in final_results:
            q_id = q_records[q_idx]["idx"]
            w.writerow([q_id, ref_id, f"{sim:.6f}", f"{erag_sim:.6f}", bl])

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done! {len(final_results)} pairs collected in {elapsed:.0f}s")
    print(f"  Metric:    {metric}")
    print(f"  CSV:       {OUTPUT_CSV}")
    print(f"  Bin counts:{dict(bin_counts)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
