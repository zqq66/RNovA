#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
null_background.py

功能：
- 从 nodes_mass (mass diff 序列列表) 构建全局经验分布 all_diffs
- 按真实长度分布随机生成序列对 (a', b')
- 在给定 score_fn(a, b) 下，用 Monte Carlo 建立 null 分布
- 提供基于 null 分布的 p-value / Z-score 计算

使用方式（示例见文件最后的 main_demo）：
1. 准备 nodes_mass: List[np.ndarray]
2. 定义 score_fn(a, b) -> float
3. null_model = build_null_distribution(nodes_mass, score_fn, n_samples=100_000)
4. p = p_value_for_score(score_obs, len_a, len_b, null_model)
"""

import numpy as np
from typing import Callable, Dict, Any, Tuple, List

# ============================================================
# 1. 从 nodes_mass 构建经验分布
# ============================================================

def build_ngram_pools(nodes_mass,
                      min_ngram: int = 4,
                      max_ngram: int = 4) -> Dict[int, List[np.ndarray]]:
    """
    从所有 nodes_mass 中提取 1..max_ngram 的 n-gram 片段。
    
    返回:
        ngrams_by_len: dict
            key = k (1..max_ngram)
            value = [np.ndarray(shape=(k,)), ...]
    """
    assert min_ngram<=max_ngram
    ngrams_by_len: Dict[int, List[np.ndarray]] = {k: [] for k in range(min_ngram, max_ngram + 1)}

    for arr in nodes_mass:
        arr = np.asarray(arr, dtype=np.float32)
        L = arr.shape[0]
        for k in range(min_ngram, max_ngram + 1):
            if L < k: continue
            # 所有长度为 k 的连续子段
            for i in range(L - k + 1):
                gram = arr[i:i + k]
                # 拷贝一份，避免后面 arr 变化影响
                ngrams_by_len[k].append(gram.copy())

    # 打印一下简单统计
    print("=== N-gram pools built ===")
    for k in range(min_ngram, max_ngram + 1):
        print(f"  k={k}: {len(ngrams_by_len[k])} grams")
    return ngrams_by_len



# ============================================================
# 2. 从 all_diffs bootstrap 抽样随机 mass 序列
# ============================================================

def sample_mass_seq(L: int,
                    all_diffs: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """
    从经验分布 all_diffs 里均匀抽样 L 个 mass，组成一条随机 mass diff 序列

    参数:
        L: 序列长度
        all_diffs: shape (N_total,)
        rng: numpy.random.Generator
    """
    N = all_diffs.shape[0]
    idx = rng.integers(0, N, size=L)
    return all_diffs[idx].astype(np.float32)

def _random_length_decomposition(L, min_ng, max_ng, rng):
    """
    把整数 L 分解为若干段长度 l_i，满足：
        min_ng <= l_i <= max_ng
        sum l_i = L

    做法：
      - 先随机选一个段数 m
      - 再依次随机出 l_1, ..., l_{m-1}
      - 最后一段 l_m = 剩余长度
    一定存在解，不会死循环。
    """
    # 能用多少段？ m 满足：
    #   m * min_ng <= L <= m * max_ng
    m_min = (L + max_ng - 1) // max_ng   # ceil(L / max_ng)
    m_max = L // min_ng                  # floor(L / min_ng)
    if m_min > m_max:
        raise ValueError(f"Cannot decompose L={L} with blocks in [{min_ng}, {max_ng}]")

    # 随机选一个 m
    m = int(rng.integers(m_min, m_max + 1))

    lengths = []
    remaining = L

    for i in range(m - 1):
        remaining_segments = m - i

        # 对当前 l_i 的约束：
        # 1) 至少 min_ng
        # 2) 至多 max_ng
        # 3) 后面的 (remaining_segments - 1) 段仍然可行：
        #       (remaining - l_i) 在 [ (remaining_segments-1)*min_ng, (remaining_segments-1)*max_ng ]
        low = max(min_ng, remaining - (remaining_segments - 1) * max_ng)
        high = min(max_ng, remaining - (remaining_segments - 1) * min_ng)
        if low > high:
            # 理论上不会发生，如果发生说明上面 m 选取逻辑写错了
            raise RuntimeError("Invalid bounds in length decomposition")

        l_i = int(rng.integers(low, high + 1))
        lengths.append(l_i)
        remaining -= l_i

    # 最后一段长度确定
    lengths.append(remaining)

    # 最后一段一定落在 [min_ng, max_ng]
    return lengths

def sample_seq_from_ngrams(L,
                           ngrams_by_len,
                           rng,
                           min_ng=8,
                           max_ng=15):
    """
    使用 n-gram 池 + 精确长度分解，生成长度恰好为 L 的 synthetic 序列。
    不用重试，不用 fallback，不会出现“剩余长度 < min_ng”的情况。

    要求：
      - 对 [min_ng, max_ng] 内每个 k，ngrams_by_len[k] 至少有 1 个元素；
        否则需要手动处理（比如调小 min_ng/max_ng 或给某些 k 做特殊生成）。
    """
    # 判断一下每个 k 是否都有池子
    for k in range(min_ng, max_ng + 1):
        if k not in ngrams_by_len or len(ngrams_by_len[k]) == 0:
            raise RuntimeError(f"No n-grams available for k={k}; "
                               f"please adjust min_ng/max_ng or add fallback logic.")

    # 先把 L 分解成一组块长
    lengths = _random_length_decomposition(L, min_ng, max_ng, rng)

    # 对每个块长，从对应池子抽一个 gram 拼在一起
    parts = []
    for k in lengths:
        pool = ngrams_by_len[k]
        idx = rng.integers(0, len(pool))
        gram = pool[idx]
        parts.append(gram)

    seq = np.concatenate(parts).astype(np.float32)
    # 防守式 assert
    if seq.shape[0] != L:
        raise RuntimeError(f"sample_mass_seq_ngram_exact: got length {seq.shape[0]} != {L}")
    return seq


'''def sample_seq_from_ngrams(L,
                           ngrams_by_len,
                           rng,
                           min_ng=8,
                           max_ng=15,
                           max_retry=50):
    """
    用 n-gram 拼接生成长度为 L 的序列。
    - 只使用长度在 [min_ng, max_ng] 的 n-grams
    - 每一步从允许的 k 中按 (k * pool_size(k)) 加权采样
    - 如果剩余长度 < min_ng，重试（最多 max_retry 次）
    """

    # 预先统计每个 k 的池子大小
    pool_sizes = {k: len(ngrams_by_len[k]) for k in range(min_ng, max_ng + 1)}
    for k in range(min_ng, max_ng + 1):
        if pool_sizes[k] == 0:
            raise RuntimeError(f"No {k}-gram available in ngrams_by_len for k={k}")

    for _ in range(max_retry):
        remaining = L
        seq_parts = []

        while remaining > 0:
            # 当前剩余长度允许的 k
            allowed = [k for k in range(min_ng, max_ng + 1)
                       if (k <= remaining and pool_sizes[k] > 0)]

            if not allowed:
                # 剩余长度 < min_ng，说明这轮拼接方案不行，重新来过
                break

            # ---- 关键修改：按长度 * 数量 加权 ----
            # weight_k = k * pool_size(k)
            weights = np.array(
                [k * pool_sizes[k] for k in allowed],
                dtype=np.float64
            )
            probs = weights / weights.sum()

            # 抽一个 k
            k = int(rng.choice(allowed, p=probs))

            # 从对应 k-gram 池中随机取一个
            pool = ngrams_by_len[k]
            idx = rng.integers(0, len(pool))
            gram = pool[idx]

            # 理论上 gram 长度一定是 k，不会 > remaining，这里只是防御性代码
            if gram.shape[0] > remaining:
                gram = gram[:remaining]

            seq_parts.append(gram)
            remaining -= gram.shape[0]

        if remaining == 0:
            # exact fit，返回拼好的序列
            return np.concatenate(seq_parts).astype(np.float32)

    # 多次尝试都失败，fallback：随便拼一点（不会经常发生）
    #print(f"[WARN] sample_mass_seq_ngram: cannot exactly match L={L} after {max_retry} retries, fallback.")
    # 简单兜底：把所有 n-gram 展平成一个大池，从里面 i.i.d 抽 L 个元素
    all_vals = np.concatenate(
        [np.concatenate(ngrams_by_len[k]) for k in range(min_ng, max_ng + 1)]
    )
    return rng.choice(all_vals, size=L).astype(np.float32)'''

# ============================================================
# 3. Monte Carlo 建立 null 分布
# ============================================================

def build_null_distribution(
        nodes_mass,
        score_fn: Callable[[np.ndarray, np.ndarray], float],
        min_ngram: int = 6,
        max_ngram: int = 15,
        target_per_L: int = 20_000,
        random_state: int = 42) -> Dict[str, Any]:
    """
    使用 n-gram 生成的随机序列 + 真实长度分布 lens，构建 null 分布。
    
    对每个 L_eff:
      - 用 n-gram 生成 a（长度 = L_eff）
      - 从 lens>=L_eff 中抽 len_b，用 n-gram 生成 b（长度 = len_b）
      - 计算 score_fn(a, b) 以及 score_fn(b, a)（防止 score_fn 不对称）
    
    参数:
        nodes_mass: list-like of 1D np.ndarray(float32)
        score_fn:   (a, b) -> float/int 的打分函数
                    —— 建议返回你用于 p-value 的那个原始 score（例如 best_match）
        max_ngram:  最大 n-gram 长度
        target_per_L: 每个 L_eff 需要的样本数的一半（因为我们对称算了两次）
        random_state: RNG seed

    返回:
        null_model: dict
            - "scores_per_L": {L_eff: sorted np.array(scores)}
            - "mean": {L_eff: mean_score}
            - "std":  {L_eff: std_score}
            - "L_min", "L_max": 有效 L_eff 范围
            - "target_per_L": target_per_L
            - "lens_empirical": 原始长度数组
    """
    rng = np.random.default_rng(random_state) 
    # 1) 构建 n-gram 池 & 长度分布
    ngrams_by_len = build_ngram_pools(nodes_mass, min_ngram=min_ngram, max_ngram=max_ngram)
    lens = np.array([len(arr) for arr in nodes_mass], dtype=np.int32)

    L_min = int(max(lens.min(), min_ngram))
    L_max = int(lens.max())

    scores_per_L = {L: [] for L in range(L_min, L_max + 1)}

    print(f"[build_null_distribution_ngram] L_eff range = {L_min} ~ {L_max}")
    print(f"[build_null_distribution_ngram] target_per_L = {target_per_L}")
    print(f"[build_null_distribution_ngram] max_ngram = {max_ngram}")
    print(f"[build_null_distribution_ngram] min_ngram = {min_ngram}")

    for L_eff in range(L_min, L_max + 1):
        # 只考虑 len >= L_eff 的长度，用于 len_b 分布
        temp_lens = lens[lens >= L_eff]
        if temp_lens.size == 0:
            print(f"  WARNING: no sequences with len >= {L_eff}, skip.")
            continue

        print(f"  Sampling for L_eff = {L_eff} ... ", end="", flush=True)
        for _ in range(target_per_L):
            # len_a = L_eff（固定）
            len_a = L_eff
            # len_b 按真实长度分布（条件在 >=L_eff）抽
            len_b = int(temp_lens[rng.integers(0, temp_lens.size)])

            # 生成 a,b
            a = sample_seq_from_ngrams(len_a, ngrams_by_len, rng, min_ng=min_ngram, max_ng=max_ngram)
            b = sample_seq_from_ngrams(len_b, ngrams_by_len, rng, min_ng=min_ngram, max_ng=max_ngram)

            # score(a,b)
            s, _ = score_fn(a, b)
            scores_per_L[L_eff].append(float(s))

        print(f"done, collected {len(scores_per_L[L_eff])} scores.")

    # 2) 整理结果
    scores_per_L_sorted = {}
    mean_dict = {}
    std_dict = {}

    for L_eff, lst in scores_per_L.items():
        if not lst:
            continue
        arr = np.sort(np.array(lst, dtype=np.float64))
        scores_per_L_sorted[L_eff] = arr
        mean_dict[L_eff] = float(arr.mean())
        std_dict[L_eff] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

    null_model = {
        "scores_per_L": scores_per_L_sorted,
        "mean": mean_dict,
        "std": std_dict,
        "L_min": L_min,
        "L_max": L_max,
        "target_per_L": target_per_L,
        "lens_empirical": lens,
        "max_ngram": max_ngram,
        "min_ngram": min_ngram,
    }

    print("[build_null_distribution_ngram] Done.")
    return null_model

# ============================================================
# 4. 利用 null_model 计算 p-value / Z-score
# ============================================================

def _find_effective_L(len_a: int, len_b: int, null_model: Dict[str, Any]) -> int:
    """
    根据 (len_a, len_b) 计算 L_eff = min(len_a, len_b)，并在
    [L_min, L_max] 范围内裁剪，以避免超出训练范围。
    """
    L_eff = min(len_a, len_b)
    L_eff = max(null_model["L_min"], min(null_model["L_max"], L_eff))
    return L_eff


def p_value_for_score(score_obs: float,
                      len_a: int,
                      len_b: int,
                      null_model: Dict[str, Any]) -> float:
    """
    基于经验分布（scores_per_L），计算给定 score_obs 的右尾 p-value：

        p = P_null(S >= score_obs | L_eff)

    用的是经验 CDF + (1 加一) 平滑：
        p = ( # {S >= score_obs} + 1 ) / (N + 1)

    参数:
        score_obs: 观测到的得分
        len_a, len_b: 实际这对序列的长度
        null_model: build_null_distribution 返回的模型

    返回:
        p-value (float)
    """
    L_eff = _find_effective_L(len_a, len_b, null_model)
    scores_dict = null_model["scores_per_L"]

    if L_eff not in scores_dict:
        # 没有对应 L_eff 的数据，退而求其次：用最近的有数据的 L
        available_L = sorted(scores_dict.keys())
        # 找距离最近的 L
        nearest_L = min(available_L, key=lambda x: abs(x - L_eff))
        L_eff = nearest_L

    arr = scores_dict[L_eff]  # sorted array
    N = arr.size

    # 找到第一个 >= score_obs 的位置
    idx = np.searchsorted(arr, score_obs, side="left")
    num_ge = N - idx

    # 加一平滑，避免 p=0
    p = (num_ge + 1.0) / (N + 1.0)
    return float(p)


def z_score_for_score(score_obs: float,
                      len_a: int,
                      len_b: int,
                      null_model: Dict[str, Any]) -> float:
    """
    基于 null_model 存的 mean/std，计算 Z-score:

        Z = (score_obs - mean) / std

    如果 std=0（极端情况），则返回 0.0。
    """
    L_eff = _find_effective_L(len_a, len_b, null_model)
    mean_dict = null_model["mean"]
    std_dict = null_model["std"]

    if L_eff not in mean_dict:
        # 同样用最近的有数据的 L
        available_L = sorted(mean_dict.keys())
        nearest_L = min(available_L, key=lambda x: abs(x - L_eff))
        L_eff = nearest_L

    mu = mean_dict[L_eff]
    sigma = std_dict[L_eff]

    if sigma <= 0:
        return 0.0

    z = (score_obs - mu) / sigma
    return float(z)


# ============================================================
# 5. 示意用法（你可以删掉或改成自己的）
# ============================================================

if __name__ == "__main__":
    # 这里只给一个简单的 demo，你可以把这段删掉或改成自己的脚本逻辑。
    # 假设你已经有 nodes_mass（这里随便造一点假数据）
    rng = np.random.default_rng(0)

    # 假造 nodes_mass：1000 条序列，每条长度 5~30，不同长度
    nodes_mass = []
    for _ in range(1000):
        L = rng.integers(5, 31)
        # 随便从几个“典型 mass”里抽，加一点噪声
        base_masses = np.array([71.0371, 57.0215, 87.0320, 99.0684, 129.0426])
        base = rng.choice(base_masses, size=L)
        noise = rng.normal(0.0, 0.01, size=L)
        nodes_mass.append((base + noise).astype(np.float32))

    # 你需要提供自己的 score_fn，这里用一个玩具版：
    def score_fn_demo(a: np.ndarray, b: np.ndarray) -> float:
        """
        玩具 score：只看长度的负绝对差。
        真实情况你会用 numba 实现的 NW / mass-tag 对齐得分。
        """
        return -abs(a.shape[0] - b.shape[0])

    # 构建 null 分布（真实使用时你会把 score_fn_demo 换成自己的）
    null_model = build_null_distribution(
        nodes_mass,
        score_fn=score_fn_demo,
        n_samples=50_000,      # 真实使用可以调大，比如 1e5 或 2e5
        random_state=123,
    )

    # 假设你对一对真实序列算出了 score_obs:
    len_a, len_b = 20, 18
    score_obs = -1.0  # 只是示例

    p = p_value_for_score(score_obs, len_a, len_b, null_model)
    z = z_score_for_score(score_obs, len_a, len_b, null_model)

    print(f"Example: score={score_obs}, len_a={len_a}, len_b={len_b}")
    print(f"  p-value ≈ {p:.3e}")
    print(f"  Z-score ≈ {z:.3f}")

