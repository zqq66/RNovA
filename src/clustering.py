# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from glob import glob
from numba import njit, prange
from numba.typed import List
from tqdm import tqdm
from .null_background_ngram import build_null_distribution

# --------------------------------------------------------
# 1. Numba：mass-tag aware NW 核心（块数最大化）
# -------lAIUHG -------------------------------------------------
MATCH = 2
MISMATCH = -1
GAP_OPEN = -4
GAP_EXTEND = -1


@njit(fastmath=True)
def nw_masstag_numba(mass_seq_a, mass_seq_b, mass_threshold=0.02, max_block_mass=600.0):
    """
    mass_seq_a, mass_seq_b: 1D float32/float64 array

    优化版：
    - 不再预计算 preA / preB
    - 在 (i,j) 处只向后枚举 “总质量 < max_block_mass” 的有限窗口
      p 从 i-1 向前累加，超过 max_block_mass 立即 break
      q 从 j-1 向前累加，超过 max_block_mass 立即 break
    复杂度 ~ O(na * nb * C^2)，C 为每个 block 的最大长度（通常很小）。
    """

    na = mass_seq_a.shape[0]
    nb = mass_seq_b.shape[0]

    NEG_INF = -(10**9)
    H = np.zeros((na + 1, nb + 1), dtype=np.int32)
    E = np.full((na + 1, nb + 1), NEG_INF, dtype=np.int32)
    F = np.full((na + 1, nb + 1), NEG_INF, dtype=np.int32)

    H_match = np.zeros((na + 1, nb + 1), dtype=np.int32)
    E_match = np.zeros((na + 1, nb + 1), dtype=np.int32)
    F_match = np.zeros((na + 1, nb + 1), dtype=np.int32)

    # (0,0)
    H[0, 0] = 0
    E[0, 0] = NEG_INF
    F[0, 0] = NEG_INF

    # 第一行（A 空 → B 有 j 个 gap）
    for j in range(1, nb + 1):
        gap_cost = GAP_OPEN + (j - 1) * GAP_EXTEND
        H[0, j] = gap_cost
        E[0, j] = gap_cost
        F[0, j] = NEG_INF

    # 第一列（B 空 → A 有 i 个 gap）
    for i in range(1, na + 1):
        gap_cost = GAP_OPEN + (i - 1) * GAP_EXTEND
        H[i, 0] = gap_cost
        F[i, 0] = gap_cost
        E[i, 0] = NEG_INF

    # 主循环
    for i in range(1, na + 1):
        for j in range(1, nb + 1):

            # 1) gap 继承：E / F
            # E: 从左边来
            open_E = H[i, j - 1] + GAP_OPEN + GAP_EXTEND
            ext_E = E[i, j - 1] + GAP_EXTEND
            if open_E > ext_E:
                E[i, j] = open_E
                E_match[i, j] = H_match[i, j - 1]
            else:
                E[i, j] = ext_E
                E_match[i, j] = E_match[i, j - 1]

            # F: 从上方来
            open_F = H[i - 1, j] + GAP_OPEN + GAP_EXTEND
            ext_F = F[i - 1, j] + GAP_EXTEND
            if open_F > ext_F:
                F[i, j] = open_F
                F_match[i, j] = H_match[i - 1, j]
            else:
                F[i, j] = ext_F
                F_match[i, j] = F_match[i - 1, j]

            # 当前 best 先从 gap 状态里选
            best = E[i, j]
            best_match = E_match[i, j]
            if F[i, j] > best:
                best = F[i, j]
                best_match = F_match[i, j]

            # 2) block：只枚举总质量 < max_block_mass 的窗口
            # A 方向：以 i-1 结尾，向前累加
            sumA = 0.0
            lenA = 0
            for p in range(i - 1, -1, -1):
                sumA += mass_seq_a[p]
                lenA += 1
                if sumA >= max_block_mass:
                    break  # 再往前只会更大，可以直接停止

                # B 方向：以 j-1 结尾，向前累加
                sumB = 0.0
                lenB = 0
                for q in range(j - 1, -1, -1):
                    sumB += mass_seq_b[q]
                    lenB += 1
                    if sumB >= max_block_mass:
                        break

                    # 现在 block 是 A[p..i-1], B[q..j-1]
                    # 计算长度与质量差
                    common_len = lenA if na > nb else lenB
                    if na == nb:
                        common_len = lenA if lenA < lenB else lenB

                    diff = sumA - sumB
                    if diff < 0.0:
                        diff = -diff

                    if diff <= mass_threshold:
                        score_block = MATCH
                        match_block = common_len
                    else:
                        score_block = MISMATCH * common_len
                        match_block = 0

                    cand = H[p, q] + score_block
                    cand_match = H_match[p, q] + match_block
                    if cand > best or (cand == best and cand_match > best_match):
                        best = cand
                        best_match = cand_match

            H[i, j] = best
            H_match[i, j] = best_match

    # 半全局尾端 gap free：
    best = H[na, nb]
    best_match = H_match[na, nb]
    return int(best), int(best_match)


@njit
def similarity_masstag(mass_seq_a, mass_seq_b, best_match):
    """
    相似度定义：block 数 / min(len(a), len(b))
    """
    na = mass_seq_a.shape[0]
    nb = mass_seq_b.shape[0]
    denom = na if na > nb else nb
    if denom <= 0:
        return 0.0
    return best_match / float(denom)


# --------------------------------------------------------
# 2. Numba 并行：某条序列 vs 所有代表 → 找 best rep
# --------------------------------------------------------


@njit(parallel=True)
def best_rep_for_seq_masstag(
    seq_idx,
    reps_idx,  # 1D int32 数组，存代表的全局 index
    nodes_mass,  # numba.typed.List[np.ndarray[float32/64]]
    mass_threshold: float,
):
    """
    对于第 seq_idx 条 mass 序列，找到在 reps_idx 中相似度最高的代表
    """
    a = nodes_mass[seq_idx]
    na = a.shape[0]

    n_rep = reps_idx.shape[0]
    sims = np.empty(n_rep, dtype=np.float32)
    scores = np.empty(n_rep, dtype=np.float32)

    for k in prange(n_rep):
        rep_i = reps_idx[k]
        b = nodes_mass[rep_i]

        s, best_match = nw_masstag_numba(a, b, mass_threshold)
        sim = similarity_masstag(a, b, best_match)
        sims[k] = sim
        scores[k] = s

    return sims, scores


@njit
def choose_cluster_for_seq(
    len_idx: int,
    reps_idx_np,  # 1D int32，全局 index
    sims,  # 1D float32，对 reps 的 sim
    scores,  # 1D float32，对 reps 的 alignment score
    nodes_len,  # 1D int32，所有序列长度
    p_thresh: float,
    sim_threshold: float,
    length_to_bin,
    null_scores,
    null_counts,
) -> int:
    """
    返回：
      - >=0: 该序列应该并入的 cluster 下标（就是 reps 的下标）
      - -1 : 不合并，需新建 cluster
    """

    n_rep = reps_idx_np.shape[0]
    if n_rep == 0:
        return -1

    # 按 sims 升序排序，然后从后往前遍历 = 降序
    order = np.argsort(sims)

    for t in range(n_rep - 1, -1, -1):
        k = order[t]  # 当前考虑的代表在 reps 里的下标
        sim = sims[k]

        # 先用 cheap 的 sim 过滤
        if sim < sim_threshold:
            continue

        rep_idx_global = reps_idx_np[k]
        len_rep = nodes_len[rep_idx_global]

        # Numba 版 p-value
        p = p_value_for_score_numba(
            scores[k], len_idx, len_rep, length_to_bin, null_scores, null_counts
        )

        if p <= p_thresh:
            return k  # 直接返回 cluster index（和 reps 顺序一致）

    # 没有任何一个代表通过筛选
    return -1


@njit
def p_value_for_score_numba(
    score_obs: float,
    len_a: int,
    len_b: int,
    length_to_bin,  # 1D int32
    null_scores,  # 2D float32 [n_bins, max_N]
    null_counts,
):  # 1D int32
    """
    Numba-friendly 版本：
      - length_to_bin[L_eff] 给出 row index
      - null_scores[row, :null_counts[row]] 是该 L 对应的已排序 score
    p = (# {S >= score_obs} + 1) / (N + 1)
    """
    # 1) L_eff
    L_eff = len_a if len_a < len_b else len_b
    if L_eff < 0:
        L_eff = 0
    max_L = length_to_bin.shape[0] - 1
    if L_eff > max_L:
        L_eff = max_L

    row = length_to_bin[L_eff]
    N = null_counts[row]
    arr = null_scores[row]

    # 2) 手写 searchsorted(arr[:N], score_obs, side="left")
    lo = 0
    hi = N
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < score_obs:
            lo = mid + 1
        else:
            hi = mid
    idx = lo

    num_ge = N - idx
    p = (num_ge + 1.0) / (N + 1.0)
    return p


# --------------------------------------------------------
# 3. CD-HIT 风格聚类（外层 Python，内层 Numba 并行）
# --------------------------------------------------------


def pack_null_model_for_numba(null_model):
    scores_dict = null_model["scores_per_L"]  # dict: L -> 1D np.array(sorted)

    # 1) 所有有数据的 L（升序）
    L_bins = np.array(sorted(scores_dict.keys()), dtype=np.int32)
    n_bins = L_bins.size

    # 2) 找所有 L 里最大的 sample 数，作为 padding 长度
    max_N = 0
    for L in scores_dict:
        arr = scores_dict[L]
        if arr.size > max_N:
            max_N = arr.size

    # 3) 建 2D scores 和 1D counts
    null_scores = np.empty((n_bins, max_N), dtype=np.float32)
    null_counts = np.zeros(n_bins, dtype=np.int32)

    for i, L in enumerate(L_bins):
        arr = np.asarray(scores_dict[L], dtype=np.float32)
        arr.sort()  # 确保升序
        n = arr.size
        null_scores[i, :n] = arr
        if n < max_N:
            # 后面的值不重要，反正不会读到
            null_scores[i, n:] = arr[-1]
        null_counts[i] = n

    # 4) 构造 length_to_bin: 任意 L → 最近的 L_bins 索引
    L_max = int(L_bins.max())
    length_to_bin = np.empty(L_max + 1, dtype=np.int32)

    for L in range(L_max + 1):
        # 找到 |L - L_bins| 最小的下标
        best_i = 0
        best_dist = abs(L - int(L_bins[0]))
        for i in range(1, n_bins):
            d = abs(L - int(L_bins[i]))
            if d < best_dist:
                best_dist = d
                best_i = i
        length_to_bin[L] = best_i

    return L_bins, length_to_bin, null_scores, null_counts


def cdhit_style_cluster_numba_masstag(
    nodes_mass,
    null_model,
    p_thresh=1e-3,
    sim_threshold=0.6,
    L_min_use=5,
    mass_threshold=0.02,
):

    n = len(nodes_mass)

    def eff_len_py(i):
        arr = nodes_mass[i]
        return int(arr.size)

    order = sorted(range(n), key=eff_len_py, reverse=True)

    clusters = []  # Python list-of-lists
    reps = []  # 代表的全局 index

    # --- Numba 需要的辅助数组 ---
    nodes_len = np.array([len(arr) for arr in nodes_mass], dtype=np.int32)
    L_bins, length_to_bin, null_scores, null_counts = pack_null_model_for_numba(
        null_model
    )

    # 预热一下 best_rep_for_seq_masstag（你原来的）
    if n > 1:
        dummy_reps = np.array([0], dtype=np.int32)
        _ = best_rep_for_seq_masstag(0, dummy_reps, nodes_mass, mass_threshold)

    for idx in tqdm(order, total=n, desc="CD-HIT style clustering (p-value, mass-tag)"):
        len_idx = len(nodes_mass[idx])
        if len_idx < L_min_use:
            continue

        if not reps:
            clusters.append([idx])
            reps.append(idx)
            continue

        reps_idx_np = np.array(reps, dtype=np.int32)

        sims, scores = best_rep_for_seq_masstag(
            idx, reps_idx_np, nodes_mass, mass_threshold
        )

        # 用 Numba 决定该放到哪个 cluster（或新建）
        cluster_choice = choose_cluster_for_seq(
            len_idx,
            reps_idx_np,
            sims.astype(np.float32),
            scores.astype(np.float32),
            nodes_len,
            float(p_thresh),
            float(sim_threshold),
            length_to_bin,
            null_scores,
            null_counts,
        )

        if cluster_choice >= 0:
            # 直接根据 reps 的下标 append
            clusters[cluster_choice].append(idx)
        else:
            clusters.append([idx])
            reps.append(idx)

    return clusters, reps


# --------------------------------------------------------
# 4. main: 读文件 → 构造 mass 序列 → 聚类
# --------------------------------------------------------
if __name__ == "__main__":
    # 1) 读 peptide list（假设 node_mass 是类似 "m0;m1;..." 的字符串）
    pep_path = "../data/YBC*csv"
    temp = []
    for f in glob(pep_path):
        temp_psm = pd.read_csv(f)
        temp_psm["file_name"] = os.path.basename(f)
        temp.append(temp_psm)
    temp = pd.concat(temp, ignore_index=True)

    # 2) 构造 numba.typed.List，每条是一个 mass difference 序列
    nodes_mass = List()
    for s in temp["node_mass"]:
        # 假设 s 形如 "m0;m1;...;mk"
        t = np.fromstring(s, sep=";", dtype=np.float32)
        # t[1:] - t[:-1] 是相邻 mass 的差，按你原先的写法保留
        diffs = t[1:] - t[:-1]
        nodes_mass.append(diffs.astype(np.float32))

    print("Total peptides:", len(nodes_mass))

    # 3) CD-HIT 风格聚类（Numba 内核，无多进程）
    null_model = build_null_distribution(
        nodes_mass,
        score_fn=nw_masstag_numba,
        min_ngram=6,
        max_ngram=15,
        target_per_L=50_000,
    )
    clusters, reps = cdhit_style_cluster_numba_masstag(
        nodes_mass, null_model, p_thresh=1e-4, sim_threshold=0.7, L_min_use=8
    )

    # 4) 简单看一下聚类结果（只保留 size>1 的簇）
    cluster_sizes = [len(c) for c in clusters]

    clusters_filt = [c for c in clusters if len(c) > 1]
    reps_filt = [r for r, c in zip(reps, clusters) if len(c) > 1]
    cluster_sizes_np = np.array([len(c) for c in clusters_filt], dtype=int)

    print("Num clusters (size>1):", len(clusters_filt))
    if cluster_sizes_np.size > 0:
        print(
            "Cluster size stats: min={}, max={}, median={}, sum={}".format(
                cluster_sizes_np.min(),
                cluster_sizes_np.max(),
                np.median(cluster_sizes_np),
                cluster_sizes_np.sum(),
            )
        )
    else:
        print("All clusters are singletons.")
