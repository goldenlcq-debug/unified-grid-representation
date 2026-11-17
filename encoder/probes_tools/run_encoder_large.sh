#!/usr/bin/env bash
# 编码器大网表征探针（任务书-编码器大网表征探针补测-给长卿-20260905）：P1 六张网 N-0；P2 25k/70k 的 N-0/N-1/N-2 配对集。
set -u
HERE="E:/统一计算大模型-迁移包/增量-20260902/潮流线-同拓扑留出补测-20260904"
L="E:/潮流线优化包-20260821/大规模算例语料交付-20260821/大规模算例语料交付-20260821"
LIBS="E:/统一计算大模型-迁移包/增量-20260902/大算例样本线-20260820/libs"
cd "$HERE" || exit 1
mkdir -p large_results
IDS2000=$(python -c "import json;print(','.join(json.load(open(r'$L/val留出清单-mp_activsg2000.json',encoding='utf-8'))['val_holdout_sample_ids']))")
run() { # grid root extra...
  g="$1"; r="$2"; shift 2
  python encoder_probe_large.py --grid "$g" --root "$r" "$@" --label P1 --out-json "large_results/probe_large_${g}.json" > "nk_logs/probe_large_${g}.log" 2>&1
  echo "$g exit $?" >> "nk_logs/probe_large_summary.log"
}
run ACTIVSg2000 "$L/gridfm-data-large/families/train/mp_activsg2000" --ids "$IDS2000"
run case3120sp "$LIBS/mp_case3120sp_fix"
run PEGASE1354 "$LIBS/mp_case1354pegase_pool"
run PEGASE2869 "$LIBS/mp_case2869pegase_pool"
run ACTIVSg25k "$L/gridfm-data-large/families/transfer/mp_activsg25k"
run ACTIVSg70k "$L/gridfm-data-large/families/transfer/mp_activsg70k" --limit 40
# P2：25k / 70k 配对 N-0/N-1/N-2
for k in 0 1 2; do
  python encoder_probe_large.py --grid "ACTIVSg25k_N$k" --root "$HERE/gridfm-data-nk/N$k/families/val/mp_activsg25k" --label P2 --out-json "large_results/probe_large_ACTIVSg25k_N$k.json" > "nk_logs/probe_large_ACTIVSg25k_N$k.log" 2>&1; echo "25k N$k exit $?" >> nk_logs/probe_large_summary.log
  python encoder_probe_large.py --grid "ACTIVSg70k_N$k" --root "$HERE/gridfm-data-nk/N$k/families/val/mp_activsg70k" --label P2 --out-json "large_results/probe_large_ACTIVSg70k_N$k.json" > "nk_logs/probe_large_ACTIVSg70k_N$k.log" 2>&1; echo "70k N$k exit $?" >> nk_logs/probe_large_summary.log
done
echo "all done" >> nk_logs/probe_large_summary.log
