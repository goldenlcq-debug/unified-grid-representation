#!/usr/bin/env bash
# 潮流推理器 N-0/N-1/N-2 配对评估：三座 E13 + 三座 E10 + 零训练提示；结果写 nk_results/。
# 用法（Git Bash，本目录）：bash run_pf_nk_eval.sh [families]   例：bash run_pf_nk_eval.sh "mp_case118,mp_case300,mp_activsg2000"
set -u
HERE="E:/统一计算大模型-迁移包/增量-20260902/潮流线-同拓扑留出补测-20260904"
GF="/e/潮流线优化包-20260821/grid-foundation"
REL="../../统一计算大模型-迁移包/增量-20260902/潮流线-同拓扑留出补测-20260904/gridfm-data-nk"
FAMS="${1:-}"
TAG="${2:-all}"
E13=R1dcL_edge0_wrap_L_vh_rf4qv_aux1_hd0.2_rd0.2
E10=R1dcL_edge0_wrap_L_vh
mkdir -p "$HERE/nk_results"
CK=""
for s in 0 1 2; do CK="$CK --ckpt runs/pf_reasoner_5060_${E13}_s$s/ckpt/best.pt"; done
for s in 0 1 2; do CK="$CK --ckpt runs/pf_reasoner_5060_${E10}_s$s/ckpt/best.pt"; done
for k in 0 1 2; do
  tag="N$k"
  cd "$GF" || exit 1
  FA=""; [ -n "$FAMS" ] && FA="--families $FAMS"
  python -u scripts/eval_by_family_5060.py $CK --set "配对·$tag:$REL/$tag:val" $FA --dc-ref lossy --bs 8 --device cuda \
      --out "$HERE/nk_results/byfam_pf_${TAG}_${tag}.json" > "$HERE/nk_logs/eval_pf_${TAG}_${tag}.log" 2>&1
  echo "eval $tag exit $?" >> "$HERE/nk_logs/eval_pf_${TAG}_${tag}.log"
  cd "$HERE" || exit 1
  HF=""; [ -n "$FAMS" ] && HF="|${FAMS%%,*}"
  for fam in $(echo "$FAMS" | tr "," " "); do python hint_ref_eval.py --set "配对·$tag|$HERE/gridfm-data-nk/$tag|val|$fam" --out "nk_results/hint_pf_${TAG}_${tag}_${fam}.json" >> "$HERE/nk_logs/eval_pf_${TAG}_${tag}.log" 2>&1; done
done
echo "all done"
