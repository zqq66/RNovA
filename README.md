# RNovA End-to-End Workflow

This repository bundles the RNovA PathSearcher and SeqFiller inference modules plus post-processing (FDR and clustering) to run an open-PTM *de novo* workflow. The top-level entry point is `RNovA.sh`.

## Contents

- `RNovA_PathSearcher_Inference/`: PathSearcher inference (stage 1, backbone path discovery). Developed by Zeping Mao (zeping.mao@uwaterloo.ca)
- `RNovA_SeqFiller_Inference/`: SeqFiller inference (stage 2, sequence completion). Developed by Zeping Mao (zeping.mao@uwaterloo.ca)
- `workflow.py`: clustering + alignment to derive top-k PTMs from PathSearcher results. Developed by Yonghan Yu
- `FDR_stage1.py`: node/path-level FDR filtering on PathSearcher outputs. Developed by Qianqiu Zhang
- `FDR-stage2.py`: sequence-level FDR filtering on SeqFiller outputs. Developed by Qianqiu Zhang
- `decoy_spectrum_generator.py`: decoy MGF generator (source: http://github.com/nh2tran/NovoBoard)

For module-specific details, see:
- `RNovA_PathSearcher_Inference/README.md`
- `RNovA_SeqFiller_Inference/README.md`

## Dependencies

Module-specific dependencies are listed in each module's `requirements.uv`. In addition, the top-level workflow scripts require:

- Python 3.10+
- `numpy`, `pandas`, `numba`, `tqdm`, `requests`

Notes:
- `workflow.py --use-unimod` enables UniMod annotation and uses `requests` to fetch UniMod data.
- `FDR-stage2.py` imports `RNovA_SeqFiller_Inference.utils.BasicClass`, so SeqFiller must be installed/built (see its README for `python setup.py build_ext --inplace`).
- Checkpoint: https://zenodo.org/records/18352464

## Clone With Submodules

This repository uses git submodules for the two inference modules. Clone with submodules so the folders are populated:

```bash
git clone --recurse-submodules https://github.com/zqq66/RNovA.git
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Quick Start (End-to-End)

```bash
./RNovA.sh <in_dir> <useunimod:true|false> <k>
```

Arguments:
- `in_dir`: directory containing input `.mgf` files.
- `useunimod`: `true` to pass `--use-unimod` to `workflow.py` (enables UniMod annotation); `false` skips UniMod annotation.
- `k`: number of PTMs to keep in the top-k list.

Example:

```bash
./RNovA.sh ./data true 10
```

`RNovA.sh` runs:
1) decoy MGF generation
2) PathSearcher inference
3) FDR stage 1 filtering
4) clustering + alignment to get top-k PTMs
5) SeqFiller inference
6) FDR stage 2 filtering
