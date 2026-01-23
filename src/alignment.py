import math
from collections import Counter
import numpy as np
# Monoisotopic masses (you can tweak to match your search engine)

aa_list = [
    "A",
    "M|UniMod:35",  # oxidized M
    "C|UniMod:4",  # carbamidomethyl C
    "D",
    "E",
    "F",
    "G",
    "H",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

BASE_AA_MASS = {
    "A": 71.03711,
    "R": 156.10111,
    "N": 114.04293,
    "D": 115.02694,
    "C": 103.00919,
    "E": 129.04259,
    "Q": 128.05858,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "L": 113.08406,
    "K": 128.09496,
    "M": 131.04049,
    "F": 147.06841,
    "P": 97.05276,
    "S": 87.03203,
    "T": 101.04768,
    "W": 186.07931,
    "Y": 163.06333,
    "V": 99.06841,
}

# Unimod mass shifts for the ones you used
UNIMOD_MASS = {
    "UniMod:35": 15.994915,  # Oxidation
    "UniMod:4": 57.021464,  # Carbamidomethyl
}


def build_token_mass_table(aa_list):
    """
    aa_list like:
      ['A','M|UniMod:35','C|UniMod:4','D','E',...]
    Returns:
      dict: token -> mass
    """
    token_mass = {}

    for tok in aa_list:
        if "|" in tok:
            base, mod = tok.split("|", 1)
            if base not in BASE_AA_MASS:
                raise ValueError(f"Unknown base AA in token {tok}")
            if mod not in UNIMOD_MASS:
                raise ValueError(f"Unknown Unimod ID in token {tok}")
            token_mass[tok] = BASE_AA_MASS[base] + UNIMOD_MASS[mod]
        else:
            if tok not in BASE_AA_MASS:
                raise ValueError(f"Unknown AA token {tok}")
            token_mass[tok] = BASE_AA_MASS[tok]

    return token_mass


def masses_to_tokens_with_list(
    mass_array, aa_list, aa_tol=0.3, fallback_to_raw_mass=True
):
    """
    Map each residue mass to the closest token in aa_list.

    Parameters
    ----------
    mass_array : list[float] or np.ndarray
        Per-residue masses: [m1, m2, ...].
    aa_list : list[str]
        Allowed tokens, e.g. ['A','M|UniMod:35','C|UniMod:4',...].
    aa_tol : float
        Max |mass - token_mass| allowed to assign a token.
    fallback_to_raw_mass : bool
        If no token is within aa_tol:
          True  -> use '(mass)' as an ambiguous mass-only token.
          False -> use 'X' as an unknown token.

    Returns
    -------
    tokens : list[str]
        Same length as mass_array, each is one of aa_list or an ambiguous token.
    """
    token_mass = build_token_mass_table(aa_list)
    tokens = []

    for m in mass_array:
        m = float(m)
        best_tok = None
        best_diff = math.inf

        for tok, tmass in token_mass.items():
            diff = abs(m - tmass)
            if diff < best_diff:
                best_diff = diff
                best_tok = tok

        if best_tok is not None and best_diff <= aa_tol:
            tokens.append(best_tok)
        else:
            # ambiguous: no allowed token close enough
            if fallback_to_raw_mass:
                tokens.append(f"({m:.3f})")
            else:
                tokens.append("X")

    return tokens


def is_mass_only(token):
    return token.startswith("(") and token.endswith(")")


def parse_mass_only(token):
    """
    '(123.456)' -> 123.456 (float)
    """
    try:
        return float(token.strip("()"))
    except Exception:
        return None


def get_base_aa(token):
    """
    Extract base AA from token:
      'A'              -> 'A'
      'M|UniMod:35'    -> 'M'
      'I(+15.99)'      -> 'I'
      '(123.456)'      -> None
      'X' / '?' / '-'  -> None
    """
    if token in {"-", "X", "?"}:
        return None
    if is_mass_only(token):
        return None

    # Unimod form: 'M|UniMod:35'
    if "|" in token:
        base, _ = token.split("|", 1)
        if base and base[0].isalpha():
            return base[0]

    # Parenthesis form: 'I(+15.99)'
    if "(" in token and ")" in token:
        base = token.split("(", 1)[0]
        if base and base[0].isalpha():
            return base[0]

    # Plain AA
    if token and token[0].isalpha():
        return token[0]

    return None


def get_mod_descriptor(token):
    """
    For comparison of "same AA but different mods":
      'M|UniMod:35'   -> 'UniMod:35'
      'I(+15.99)'     -> '(+15.99)'
      'A'             -> ''
      '(123.4)'       -> ''
    """
    if "|" in token:
        _, mod = token.split("|", 1)
        return mod
    if "(" in token and ")" in token and not is_mass_only(token):
        s = token.find("(")
        e = token.rfind(")")
        if e > s:
            return token[s : e + 1]
    return ""


def get_mass_for_base(base_aa):
    """
    Monoisotopic mass for unmodified base AA.
    """
    return BASE_AA_MASS.get(base_aa, None)


def token_mass_value(token, token_mass_table):
    """
    Get numeric mass of any token:
      - AA or AA|UniMod:xx   -> from token_mass_table
      - AA(+delta)           -> BASE_AA_MASS[AA] + delta
      - (mass-only)          -> that mass
      - X,?, -               -> None
    """
    if token in {"-", "X", "?"}:
        return None

    # mass-only
    if token.startswith("(") and token.endswith(")"):
        return parse_mass_only(token)

    # from aa_list
    if token in token_mass_table:
        return token_mass_table[token]

    # AA(+delta) form
    if "(" in token and ")" in token:
        base = token.split("(", 1)[0]
        try:
            delta = float(token[token.find("(") + 1 : token.rfind(")")])
        except Exception:
            return None
        if base in BASE_AA_MASS:
            return BASE_AA_MASS[base] + delta
        return None

    # bare AA
    if token in BASE_AA_MASS:
        return BASE_AA_MASS[token]

    return None


def build_mod_token(base_aa, delta_mass, zero_tol=0.02):
    """
    Build a token for consensus-filled residue with mass info:
      If |delta_mass| <= zero_tol -> just 'A'
      Else                        -> 'A(+1.234)'
    """
    if base_aa is None:
        return "X"
    if abs(delta_mass) <= zero_tol:
        return base_aa
    return f"{base_aa}({delta_mass:+.3f})"


MATCH_SCORE = 3
MOD_MATCH_SCORE = 2  # same AA, different mod (Unimod or delta)
UNKNOWN_MATCH = 1  # X/mass-only vs known AA
MISMATCH_SCORE = -1

GAP_OPEN = -5
GAP_EXTEND = -1


def token_match_score(a, b):
    """
    Score for aligning token a vs token b.
    """
    if a == "-" or b == "-":
        return MISMATCH_SCORE

    base_a = get_base_aa(a)
    base_b = get_base_aa(b)

    # both "unknown"
    if base_a is None and base_b is None:
        # e.g. X vs X, '(mass)' vs '(mass)'
        return UNKNOWN_MATCH

    # one unknown, one known
    if base_a is None and base_b is not None:
        return UNKNOWN_MATCH
    if base_b is None and base_a is not None:
        return UNKNOWN_MATCH

    # both have base AAs
    if base_a == base_b:
        mod_a = get_mod_descriptor(a)
        mod_b = get_mod_descriptor(b)
        if mod_a == mod_b:
            return MATCH_SCORE
        else:
            return MOD_MATCH_SCORE
    else:
        return MISMATCH_SCORE


def nw_tokens(seq_a_tokens, seq_b_tokens, gap_open=GAP_OPEN, gap_extend=GAP_EXTEND):
    """
    Needleman–Wunsch with affine gaps for peptide tokens.
    Returns:
        aln_a, aln_b  (aligned token lists)
    """

    na = len(seq_a_tokens)
    nb = len(seq_b_tokens)

    NEG_INF = -(10**9)

    H = np.full((na + 1, nb + 1), NEG_INF, dtype=np.int32)  # best
    E = np.full((na + 1, nb + 1), NEG_INF, dtype=np.int32)  # gap in A (horizontal)
    F = np.full((na + 1, nb + 1), NEG_INF, dtype=np.int32)  # gap in B (vertical)

    TB_H = np.zeros((na + 1, nb + 1), dtype=np.int8)  # 0=diag,1=E,2=F
    TB_E = np.zeros((na + 1, nb + 1), dtype=np.int8)  # 0=from H,1=from E
    TB_F = np.zeros((na + 1, nb + 1), dtype=np.int8)  # 0=from H,1=from F

    # init
    H[0, 0] = 0
    E[0, 0] = F[0, 0] = NEG_INF

    # first row: gaps in A
    for j in range(1, nb + 1):
        gap_cost = gap_open + (j - 1) * gap_extend
        H[0, j] = gap_cost
        E[0, j] = gap_cost
        F[0, j] = NEG_INF
        TB_H[0, j] = 1
        TB_E[0, j] = 1

    # first column: gaps in B
    for i in range(1, na + 1):
        gap_cost = gap_open + (i - 1) * gap_extend
        H[i, 0] = gap_cost
        F[i, 0] = gap_cost
        E[i, 0] = NEG_INF
        TB_H[i, 0] = 2
        TB_F[i, 0] = 1

    # fill
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            # E: gap in A
            open_E = H[i, j - 1] + gap_open + gap_extend
            ext_E = E[i, j - 1] + gap_extend
            if open_E >= ext_E:
                E[i, j] = open_E
                TB_E[i, j] = 0
            else:
                E[i, j] = ext_E
                TB_E[i, j] = 1

            # F: gap in B
            open_F = H[i - 1, j] + gap_open + gap_extend
            ext_F = F[i - 1, j] + gap_extend
            if open_F >= ext_F:
                F[i, j] = open_F
                TB_F[i, j] = 0
            else:
                F[i, j] = ext_F
                TB_F[i, j] = 1

            # diag match/mismatch
            s_mm = H[i - 1, j - 1] + token_match_score(
                seq_a_tokens[i - 1], seq_b_tokens[j - 1]
            )

            best = s_mm
            tb = 0  # diag
            if E[i, j] > best:
                best = E[i, j]
                tb = 1
            if F[i, j] > best:
                best = F[i, j]
                tb = 2

            H[i, j] = best
            TB_H[i, j] = tb

    # traceback
    i, j = na, nb
    aln_a, aln_b = [], []

    while i > 0 or j > 0:
        tb = TB_H[i, j]
        if tb == 0:
            aln_a.append(seq_a_tokens[i - 1] if i > 0 else "-")
            aln_b.append(seq_b_tokens[j - 1] if j > 0 else "-")
            i -= 1
            j -= 1
        elif tb == 1:
            # from E: gap in A
            aln_a.append("-")
            aln_b.append(seq_b_tokens[j - 1])
            j -= 1
            # follow TB_E if you want finer control; here 1-step is enough
        elif tb == 2:
            # from F: gap in B
            aln_a.append(seq_a_tokens[i - 1])
            aln_b.append("-")
            i -= 1

    aln_a.reverse()
    aln_b.reverse()
    return aln_a, aln_b


from collections import Counter


def build_consensus_bases(aligned_token_seqs):
    """
    aligned_token_seqs: list of [tok1,...,tokL], all same length
    Returns: [base_aa or None] per column
    """
    if not aligned_token_seqs:
        return []

    L = len(aligned_token_seqs[0])
    consensus_bases = []

    for col in range(L):
        bases = []
        for seq in aligned_token_seqs:
            base = get_base_aa(seq[col])
            if base is not None:
                bases.append(base)

        if bases:
            base_cons = Counter(bases).most_common(1)[0][0]
        else:
            base_cons = None

        consensus_bases.append(base_cons)

    return consensus_bases


def get_columnwise_alignment(filled_token_seqs):
    """
    filled_token_seqs: list of token lists, all same length.
        e.g. [
          ['N','T','V','V','T','G','R'],
          ['N','T','V(+15.99)','V','T','G','R'],
          ...
        ]

    Returns:
        columns: list of columns, each column is a list of tokens across sequences.
        columns[c][s] = token at column c for sequence s.
    """
    if not filled_token_seqs:
        return []

    # transpose: rows -> sequences, cols -> alignment positions
    columns = list(map(list, zip(*filled_token_seqs)))
    return columns


def single_aa_fill(
    aligned_token_seqs,
    consensus_bases,
    mass_zero_tol=0.02,
    single_aa_mass_tol=0.5,
    max_abs_delta=150.0,
):
    """
    Conservative single-AA fill:

    Only convert '(M)' -> B(+delta) when:
      - Column consensus base is B
      - This sequence has '(M)' at that column
      - Both left and right neighbors in this sequence are NOT gaps
      - |M - mass(B)| <= single_aa_mass_tol
      - |delta| = |M - mass(B)| <= max_abs_delta

    This avoids:
      - huge modifications from big mass tags
      - positions that are in the middle of long gap/misalignment blocks
    """
    n_seq = len(aligned_token_seqs)
    if n_seq == 0:
        return aligned_token_seqs
    L = len(aligned_token_seqs[0])

    filled = [list(seq) for seq in aligned_token_seqs]
    disallowed_bases = {"V", "L", "I", "P"}

    for j in range(L):
        cons_base = consensus_bases[j]
        if cons_base is None:
            continue

        if cons_base in disallowed_bases:
            continue

        base_mass = BASE_AA_MASS.get(cons_base, None)
        if base_mass is None:
            continue

        for s in range(n_seq):
            tok = filled[s][j]

            # already has base AA or gap -> skip
            if tok == "-" or get_base_aa(tok) is not None:
                continue

            # only consider mass-only '(M)'
            if not (tok.startswith("(") and tok.endswith(")")):
                continue

            # --- 1) local alignment check: neighbors must not be gaps ---
            left_gap = j > 0 and filled[s][j - 1] == "-"
            right_gap = j < L - 1 and filled[s][j + 1] == "-"
            if left_gap or right_gap:
                # looks like part of an indel/misalignment block -> let
                # collapsed_block_fill handle this later
                continue

            # --- 2) delta size check ---
            raw = parse_mass_only(tok)
            if raw is None:
                continue

            delta = raw - base_mass

            # must be close enough to base mass
            if abs(delta) < single_aa_mass_tol:
                continue

            # and must not be a huge "modification"
            if abs(delta) > max_abs_delta:
                continue

            # OK: treat as B(+delta)
            filled[s][j] = build_mod_token(cons_base, delta, zero_tol=mass_zero_tol)

    return filled


def collapsed_block_fill(aligned_token_seqs, block_mass_tol=0.5):
    """
    (3-position block):

      Reference triple at (j-1, j, j+1):
          ref[j-1] = base0
          ref[j]   = base1
          ref[j+1] = (m1)  # mass-only tag

      Target triple at (j-1, j, j+1) can be:
          '-',   base1, (m2)
          base0, '-',   (m2)
          base0, base1, (m2)

      If m2 ≈ mass(base1) + m1 (within block_mass_tol),
      we copy ref[j-1:j+2] into target[j-1:j+2].

    """
    n_seq = len(aligned_token_seqs)
    if n_seq == 0:
        return aligned_token_seqs

    L = len(aligned_token_seqs[0])
    seqs = [list(seq) for seq in aligned_token_seqs]

    for r in range(n_seq):  # reference sequence index
        ref = seqs[r]

        for j in range(1, L - 1):  # center index of triple: (j-1, j, j+1)
            ref0 = ref[j - 1]
            ref1 = ref[j]
            ref2 = ref[j + 1]

            base0 = get_base_aa(ref0)
            base1 = get_base_aa(ref1)

            # need two bases then a mass-only tag
            if base0 is None or base1 is None:
                continue
            if not (ref2.startswith("(") and ref2.endswith(")")):
                continue

            m1 = parse_mass_only(ref2)
            if m1 is None:
                continue

            base1_mass = BASE_AA_MASS.get(base1)
            if base1_mass is None:
                continue

            # predicted collapsed A(m2) mass for the 2nd residue
            predicted_m2 = base1_mass + m1

            # now try to fix other sequences at this triple
            for s in range(n_seq):
                if s == r:
                    continue
                tgt = seqs[s]
                t0 = tgt[j - 1]
                t1 = tgt[j]
                t2 = tgt[j + 1]

                # we always need a mass-only at t2
                if not (t2.startswith("(") and t2.endswith(")")):
                    continue
                m2 = parse_mass_only(t2)
                if m2 is None:
                    continue

                if abs(m2 - predicted_m2) > block_mass_tol:
                    continue

                # --- pattern A: '-', base1, (m2) ---
                if t0 == "-" and get_base_aa(t1) == base1:
                    tgt[j - 1], tgt[j], tgt[j + 1] = ref0, ref1, ref2
                    continue

                # --- pattern B: base0, '-', (m2) ---
                if get_base_aa(t0) == base0 and t1 == "-":
                    tgt[j - 1], tgt[j], tgt[j + 1] = ref0, ref1, ref2
                    continue

                # --- pattern C: base0, base1, (m2) ---
                if get_base_aa(t0) == base0 and get_base_aa(t1) == base1:
                    tgt[j - 1], tgt[j], tgt[j + 1] = ref0, ref1, ref2
                    continue

    return seqs


def pretty_print_alignment(aligned_token_seqs):
    """
    aligned_token_seqs: list of sequences, each a list of tokens (strings)

    Prints a readable alignment with per-column padding.
    Works even if sequences have different lengths.
    """
    n = len(aligned_token_seqs)
    if n == 0:
        print("(empty)")
        return

    # max alignment length across sequences
    L = max(len(seq) for seq in aligned_token_seqs)

    # determine max token width for each column
    widths = [0] * L
    for j in range(L):
        for s in range(n):
            if j < len(aligned_token_seqs[s]):
                tok = aligned_token_seqs[s][j]
            else:
                tok = ""  # no token here
            if len(tok) > widths[j]:
                widths[j] = len(tok)

    # header: column indices (1-based)
    col_line = []
    for j in range(L):
        col_line.append(str(j + 1).rjust(widths[j]))
    print("Col: ", " ".join(col_line))

    # each sequence
    for idx, seq in enumerate(aligned_token_seqs):
        row = []
        for j in range(L):
            if j < len(seq):
                tok = seq[j]
            else:
                tok = ""  # pad missing positions
            row.append(tok.rjust(widths[j]))
        print(f"Seq{idx+1}:", " ".join(row))
