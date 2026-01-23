import pandas as pd
import numpy as np
import pickle
import xml.etree.ElementTree as ET
import requests
from io import BytesIO
from collections import defaultdict
import re
from typing import List, Dict, Any, Optional
import bisect


def load_unimod():
    url = "https://www.unimod.org/xml/unimod.xml"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    mods = []
    for _, elem in ET.iterparse(BytesIO(r.content), events=("end",)):
        # elem.tag will look like "{namespace}mod", not "mod"
        if elem.tag.endswith("mod"):
            delta = None
            for child in list(elem):
                if child.tag.endswith("delta"):
                    delta = child
                    break
            if delta is None:
                elem.clear()
                continue

            mono = delta.attrib.get("mono_mass")
            avge = delta.attrib.get("avge_mass")

            specs = []
            for s in elem:
                if s.tag.endswith("specificity"):
                    specs.append((s.attrib.get("site"), s.attrib.get("position")))

            mods.append(
                {
                    "title": elem.attrib.get("title"),
                    "unimod_id": int(elem.attrib.get("record_id")),
                    "mono_mass": float(mono),
                    "avg_mass": float(avge) if avge is not None else None,
                    "specificity": specs,
                }
            )

            elem.clear()

    return mods


class UniModMassIndex:
    def __init__(self, mods):
        self.mods = sorted(mods, key=lambda x: x["mono_mass"])
        self.masses = [m["mono_mass"] for m in self.mods]

    def query(self, mass, tol=0.01):
        """
        Find UniMod PTMs within ± tol (Da)
        """
        left = bisect.bisect_left(self.masses, mass - tol)
        right = bisect.bisect_right(self.masses, mass + tol)
        return self.mods[left:right]

    def query_ppm(self, mass, ppm=10):
        """
        Find UniMod PTMs within ± ppm
        """
        tol = mass * ppm * 1e-6
        left = bisect.bisect_left(self.masses, mass - tol)
        right = bisect.bisect_right(self.masses, mass + tol)
        return self.mods[left:right]


def mass_key(mass, tol=1e-6):
    return round(mass / tol)


def query_collapsed(index, collapsed, delta_mass, tol=0.01):
    hits = index.query(delta_mass, tol)

    collapsed_hits = {}
    for h in hits:
        key = mass_key(h["mono_mass"])
        collapsed_hits[key] = collapsed[key]

    return collapsed_hits


def collapse_unimod_by_mass(mods, tol=1e-6):
    groups = defaultdict(list)

    for m in mods:
        key = mass_key(m["mono_mass"], tol)
        groups[key].append(m)

    return groups


def residue_allowed(unimod_entry, residue: str) -> bool:
    # UniMod uses site like 'N', 'S', 'K', plus sometimes 'N-term', etc.
    for site, _pos in unimod_entry.get("specificity", []):

        if site == residue:
            return True
    return False


def delta_to_unimod_candidates(
    index, residue: str, delta_mass: float, tol: float = 0.01
):
    hits = index.query(delta_mass, tol=tol)
    hits = [h for h in hits if residue_allowed(h, residue)]
    hits.sort(key=lambda h: abs(h["mono_mass"] - delta_mass))
    return hits


# Only delta-mass with + sign: N(+70.042)
RE_PLUS = re.compile(r"([A-Z])\(\+([\d.]+)\)")

# Explicit unimod ID after a residue: M|UniMod:35
RE_UNIMOD = re.compile(r"([A-Z])\|UniMod:(\d+)")

RE_UNIMOD_AFTER_BAR = re.compile(r"\|UniMod:(\d+)")
RE_PLUS_PAREN = re.compile(r"\(\+([\d.]+)\)")


def parse_peptide_mods(seq: str) -> List[Dict[str, Any]]:
    mods: List[Dict[str, Any]] = []
    residues: List[str] = []

    i = 0
    n = len(seq)

    while i < n:
        ch = seq[i]

        # Skip any parenthetical block that is NOT parsed as a (+delta) right after a residue.
        # This correctly ignores things like "(216.074)" before a residue or "(210.136)" after a residue.
        if ch == "(":
            j = seq.find(")", i + 1)
            if j == -1:
                break
            i = j + 1
            continue

        # Backbone residue letter
        if "A" <= ch <= "Z":
            aa = ch
            idx = len(residues)  # 0-based index in plain sequence
            residues.append(aa)
            i += 1

            # Case 1) Known PTM: AA|UniMod:ID
            if i < n and seq[i] == "|":
                m_u = RE_UNIMOD_AFTER_BAR.match(seq, i)
                if m_u:
                    mods.append(
                        {
                            "index": idx,
                            "residue": aa,
                            "unimod_id": int(m_u.group(1)),
                            "delta_mass": None,
                        }
                    )
                    i = m_u.end()
                    continue  # done with this residue annotation

            # Case 2) Delta PTM: AA(+mass)  (ONLY when there's a + sign)
            if i < n and seq[i] == "(":
                m_p = RE_PLUS_PAREN.match(seq, i)
                if m_p:
                    mods.append(
                        {
                            "index": idx,
                            "residue": aa,
                            "unimod_id": None,
                            "delta_mass": float(m_p.group(1)),
                        }
                    )
                    i = m_p.end()
                else:
                    # Ignore non-(+...) parentheses after a residue (e.g. T(781.404))
                    j = seq.find(")", i + 1)
                    i = (j + 1) if j != -1 else n

            continue

        # everything else
        i += 1

    # Terminal filtering:
    # - keep terminal mods if unimod_id is known
    # - drop terminal mods if they are delta_mass-only
    L = len(residues)
    if L == 0:
        return []

    def keep(m: Dict[str, Any]) -> bool:
        is_terminal = (m["index"] == 0) or (m["index"] == L - 1)
        if not is_terminal:
            return True
        return m["unimod_id"] is not None  # keep known, drop delta

    return [m for m in mods if keep(m)]


def parse_peptide_mods_filter_terminals(seq: str) -> List[Dict[str, Any]]:
    """
    Parse peptide annotations and return internal (non-terminal) residue mods only.

    Supported:
      - C|UniMod:4  -> unimod_id=4 on residue C
      - G(+128.060) -> delta_mass=128.060 on residue G   (ONLY if + sign)
    Ignored:
      - (216.074)C ... prefix mass before a residue
      - T(781.404) ... non-delta parentheses after residue
      - ...any other parentheses not matching (+mass)
    """
    mods: List[Dict[str, Any]] = []
    residues: List[str] = []

    i = 0
    n = len(seq)

    while i < n:
        ch = seq[i]

        # Skip any parenthetical block that is NOT a (+delta) immediately after a residue
        if ch == "(":
            j = seq.find(")", i + 1)
            if j == -1:
                break
            # Just skip it (prefix mass or random annotation)
            i = j + 1
            continue

        # Backbone residue
        if "A" <= ch <= "Z":
            aa = ch
            idx = len(residues)  # 0-based index in plain sequence
            residues.append(aa)
            i += 1

            # Optional explicit UniMod immediately after residue: |UniMod:NNN
            m_u = RE_UNIMOD.match(seq, i) if i < n and seq[i] == "|" else None
            if m_u:
                unimod_id = int(m_u.group(1))
                mods.append(
                    {
                        "index": idx,
                        "residue": aa,
                        "unimod_id": unimod_id,
                        "delta_mass": None,
                    }
                )
                i = m_u.end()

            # Optional delta immediately after residue: (+NNN.NNN)
            elif i < n and seq[i] == "(":
                m_p = RE_PLUS.match(seq, i)
                if m_p:
                    dm = float(m_p.group(1))
                    mods.append(
                        {
                            "index": idx,
                            "residue": aa,
                            "unimod_id": None,
                            "delta_mass": dm,
                        }
                    )
                    i = m_p.end()
                else:
                    # some other (...) after residue, ignore
                    j = seq.find(")", i + 1)
                    i = (j + 1) if j != -1 else n

            continue

        # everything else
        i += 1

    # filter out first and last residue mods
    plain_len = len(residues)
    if plain_len == 0:
        return []

    def keep(m):
        is_terminal = (m["index"] == 0) or (m["index"] == plain_len - 1)
        if not is_terminal:
            return True
        # terminal: keep if it's a known UniMod assignment, drop if it's delta mass
        return m["unimod_id"] is not None

    return [m for m in mods if keep(m)]


def fill_delta_with_unimod(pep, index, tol=0.01):
    mods_residue = []
    mods_name = []

    for m in parse_peptide_mods(pep):
        if m["unimod_id"] is None:
            cands = delta_to_unimod_candidates(
                index, m["residue"], m["delta_mass"], tol=tol
            )
            # print(f"Site {m['residue']}{m['index']+1}, Δ={m['delta_mass']}:")
            if len(cands) == 0:
                continue
            for c in cands[:1]:
                mods_residue.append(f"{m['residue']}{m['index']+1}")
                mods_name.append(f"{c['title']}|UniMod:{c['unimod_id']}")
                # err_ppm = (c["mono_mass"] - m["delta_mass"]) / m["delta_mass"] * 1e6
                # print(
                # f"  {c['title']} (UNIMOD:{c['unimod_id']}), {c['mono_mass']:.6f}, err={err_ppm:+.2f} ppm"
                # )
    if len(mods_residue) > 0:
        return mods_residue, mods_name
    else:
        return None, None
