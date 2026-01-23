import argparse
import logging
import os
import sys
import re
from glob import glob
from pathlib import Path

from src.null_background_ngram import build_null_distribution
from src.clustering import cdhit_style_cluster_numba_masstag, nw_masstag_numba
from src.alignment import *
from src.fill_PTM import load_unimod, UniModMassIndex, fill_delta_with_unimod,parse_peptide_mods
from src.DBSCAN1D import DBSCAN1D
import numpy as np
import pandas as pd
import numba

logger = logging.getLogger(__name__)

def build_cluster(in_csv: pd.DataFrame):
    temp = in_csv
    nodes_mass = numba.typed.List()
    for s in temp['node_mass']:
        # 假设 s 形如 "m0;m1;...;mk"
        t = np.fromstring(s, sep=';', dtype=np.float32)
        # t[1:] - t[:-1] 是相邻 mass 的差，按你原先的写法保留
        diffs = t[1:] - t[:-1]
        nodes_mass.append(diffs.astype(np.float32))

    logger.info("Total peptides: %s", len(nodes_mass))

    # 3) CD-HIT 风格聚类（Numba 内核，无多进程）
    null_model = build_null_distribution(nodes_mass, score_fn=nw_masstag_numba, min_ngram=6, max_ngram=15, target_per_L=50_000)
    clusters, reps = cdhit_style_cluster_numba_masstag(
        nodes_mass,
        null_model,
        p_thresh=1e-4,
        sim_threshold=0.7,
        L_min_use=8
    )

    cluster_sizes = [len(c) for c in clusters]

    clusters_filt = [c for c in clusters if len(c) > 1]
    reps_filt = [r for r, c in zip(reps, clusters) if len(c) > 1]
    cluster_sizes_np = np.array([len(c) for c in clusters_filt], dtype=int)

    logger.info("Num clusters (size>1): %s", len(clusters_filt))
    if cluster_sizes_np.size > 0:
        logger.info("Cluster size stats: min=%s, max=%s, median=%s, sum=%s",
            cluster_sizes_np.min(),
            cluster_sizes_np.max(),
            np.median(cluster_sizes_np),
            cluster_sizes_np.sum(),
        )
    else:
        logger.info("All clusters are singletons.")

    return nodes_mass, clusters_filt


def masstag_to_delta_mass(nodes_mass, clusters_filt, temp):
    result = pd.DataFrame(columns=temp.columns)
    result["raw_sequence"] = []
    result["filled_sequence"] = []
    result["cluster_center"] = []

    for cluster in clusters_filt:
        token_seqs = []
        for idx in cluster:
            toks = masses_to_tokens_with_list(
                nodes_mass[idx],
                aa_list,
                aa_tol=0.1,
            )
            token_seqs.append(toks)
        center_idx = cluster[0]
        center = token_seqs[0]
        aligned_token_seqs = []
        for toks in token_seqs:
            aln_a, aln_b = nw_tokens(center, toks)
            aligned_token_seqs.append(aln_b)
        consensus = build_consensus_bases(aligned_token_seqs)
        filled = single_aa_fill(
            aligned_token_seqs,
            consensus,
            mass_zero_tol=0.02,
            single_aa_mass_tol=0.5,
            max_abs_delta=150.0,
        )

        filled2 = collapsed_block_fill(
            filled,
            block_mass_tol=0.5,
        )

        ungapped = ["".join(tok for tok in seq if tok != "-") for seq in filled2]
        raw = ["".join(tok for tok in seq if tok != "-") for seq in token_seqs]

        temp_psm = temp.iloc[cluster].copy()
        temp_psm["raw_sequence"] = raw
        temp_psm["filled_sequence"] = ungapped
        temp_psm["cluster_center"] = int(center_idx)
        result = pd.concat([result, temp_psm])

    result = result.reset_index()
    return result

def topk_ptm_annotation(filled_result, k=3):
    ptm_freq = dict()
    for row in filled_result.itertuples():
        pep = row.filled_sequence
        ptms = parse_peptide_mods(pep)
        for p in ptms:
            if p['delta_mass'] is not None:
                ptm = p['residue']+'['+str(p['delta_mass']) + ']'
                if ptm in ptm_freq.keys():
                    ptm_freq[ptm] += 1
                else:
                    ptm_freq[ptm] = 1
    return ptm_freq


def cluster_ptm_pairs(ptm_counts, eps=0.02, min_samples=1, mass_precision=3):
    grouped = {}
    residue_order = []
    leftovers = {}
    ptm_re = re.compile(r"^([A-Za-z])\[(.+)\]$")

    for ptm, count in ptm_counts.items():
        match = ptm_re.match(ptm)
        if not match:
            leftovers[ptm] = leftovers.get(ptm, 0) + count
            continue
        residue = match.group(1)
        try:
            mass = float(match.group(2))
        except ValueError:
            leftovers[ptm] = leftovers.get(ptm, 0) + count
            continue
        if residue not in grouped:
            grouped[residue] = {"masses": [], "counts": []}
            residue_order.append(residue)
        grouped[residue]["masses"].append(mass)
        grouped[residue]["counts"].append(count)

    clustered = {}
    for residue in residue_order:
        masses = np.asarray(grouped[residue]["masses"], dtype=float)
        counts = np.asarray(grouped[residue]["counts"], dtype=float)
        if masses.size == 0:
            continue
        db = DBSCAN1D(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(masses)
        for label in np.unique(labels):
            if label < 0:
                continue
            mask = labels == label
            center = masses[mask].mean()
            key = f"{residue}[{center:.{mass_precision}f}]"
            clustered[key] = clustered.get(key, 0) + counts[mask].sum()

    for ptm, count in leftovers.items():
        clustered[ptm] = clustered.get(ptm, 0) + count

    return clustered


def delta_mass_to_ptm(filled_result: pd.DataFrame, index: UniModMassIndex, k, tol=0.01):
    df_ptm = pd.DataFrame(columns=filled_result.columns.tolist() + ["mod_residue", "mod_name"])
    ptm_freq = dict()
    for row in filled_result.itertuples():
        pep = row.filled_sequence
        mods_residue, mods_name = fill_delta_with_unimod(pep, index, tol=tol)
        if mods_residue is not None:
            mods_residue = ";".join(mods_residue)
            mods_name = ";".join(mods_name)
            new_row = row._asdict()
            new_row["mod_residue"] = mods_residue
            new_row["mod_name"] = mods_name
            ptm = mods_residue[0]+'|'+mods_name.split('|')[-1]
            df_ptm = pd.concat([df_ptm, pd.DataFrame([new_row])], ignore_index=True)
            if ptm in ptm_freq.keys():
                ptm_freq[ptm] += 1
            else:
                ptm_freq[ptm] = 1
    return df_ptm, ptm_freq

def parse_args():
    parser = argparse.ArgumentParser(description="Run the peptide clustering and optional UniMod annotation workflow.")
    parser.add_argument("pep_path", help="Glob pattern for input CSVs, e.g. /path/to/YBC*csv")
    parser.add_argument(
        "--topk-ptm",
        type=int,
        default=None,
        help="Compute top-k PTM frequencies from filled peptides",
    )
    parser.add_argument(
        "--use-unimod",
        action="store_true",
        help="Enable UniMod annotation (default)",
        default=False,
    )
    return parser.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    args = parse_args()
    temp = []
    for f in glob(args.pep_path):
        temp_psm = pd.read_csv(f)
        temp_psm['file_name'] = os.path.basename(f)
        temp.append(temp_psm)
    temp = pd.concat(temp, ignore_index=True)

    pep_dir = Path(args.pep_path).parent
    output_path = str(pep_dir / "filled_peptides.csv")

    # build sequence cluster
    nodes_mass, clusters_filt = build_cluster(
        temp
    )

    # msa on clusters and fill mass tag gaps with delta masses + aa
    filled_result = masstag_to_delta_mass(
        nodes_mass,
        clusters_filt,
        temp,
    )

    # save filled result
    filled_result.to_csv(output_path, index=False)

    if args.topk_ptm is not None:
        topk = args.topk_ptm
    else:
        topk = 3
    if args.use_unimod:
        # load UniMod database
        mods = load_unimod()
        index = UniModMassIndex(mods)

        # annotate PTMs
        ptm_result, ptm_freq = delta_mass_to_ptm(filled_result, index, topk)
        # save PTM annotated result
        output_path_ptm = os.path.splitext(output_path)[0] + "_with_PTM.csv"
        ptm_result.to_csv(output_path_ptm, index=False)
    else:
        ptm_freq = topk_ptm_annotation(filled_result, topk)
        # print(topk_ptm)
    clustered_topk_ptm = cluster_ptm_pairs(ptm_freq, eps=0.02)
    ptm_freq_clustered = dict(sorted(clustered_topk_ptm.items(), key=lambda kv: kv[1], reverse=True))
    print("topk_ptm_annotation:", ";".join(list(ptm_freq_clustered.keys())[:topk]))

if __name__ == "__main__":
    main()
