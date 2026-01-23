#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./RNovA.sh <in_dir> <useunimod:true|false> <k>

Arguments:
  in_dir     Directory containing input .mgf files
  useunimod  Whether to use UniMod annotation: true or false
  k          Number of PTMs to consider for top-k selection

Options:
  -h, --help Show this help and exit
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 3 ]]; then
  usage
  exit 1
fi

in_dir="$1"
useunimod="$2"
k="$3"

if [[ "$useunimod" != "true" && "$useunimod" != "false" ]]; then
  echo "error: useunimod must be true or false" >&2
  exit 1
fi

decoy_dir="$in_dir/decoy_mgf"
mkdir -p "$decoy_dir"

echo "Step 1/6: Generating decoy MGF files..."
python decoy_spectrum_generator.py "$in_dir" "$decoy_dir"

echo "Step 2/6: Running PathSearcher inference..."
pushd RNovA_PathSearcher_Inference >/dev/null
python Inference_Node.py "$in_dir"/*.mgf "$decoy_dir"/*.mgf
popd >/dev/null

echo "Step 3/6: Running FDR stage 1..."
python FDR_stage1.py "$in_dir" "$decoy_dir"

workflow_args=(--topk-ptm "$k")
if [[ "$useunimod" == "true" ]]; then
  workflow_args=(--use-unimod --topk-ptm "$k")
fi
echo "Step 4/6: Running clustering and alignment to get top-k PTMs..."
topk_ptm=$(python workflow.py "$in_dir"/*_rnova_denovo_path_node_path001fdr.csv "${workflow_args[@]}" \
  | tee /dev/stderr \
  | awk -F': ' '/^topk_ptm_annotation: /{print $2}')

echo "Step 5/6: Running SeqFiller inference... topk_ptm=${topk_ptm}"
pushd RNovA_SeqFiller_Inference >/dev/null
python Inference_Sequence.py "$in_dir"/*.mgf "$decoy_dir"/*.mgf \
  "A;C|UniMod:4;D;E;F;G;H;K;L;M;N;P;Q;R;S;T;V;W;Y;${topk_ptm}"
popd >/dev/null
echo "Step 6/6: Running FDR stage 2..."
python FDR-stage2.py "$in_dir" "$decoy_dir"
