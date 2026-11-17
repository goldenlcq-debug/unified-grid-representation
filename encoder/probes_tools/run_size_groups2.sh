#!/usr/bin/env bash
# 任务书 2026-09-06 A/B 分组探针，两路并行版（源网 src 已由 run_size_groups.sh 跑完）。用法：bash run_size_groups2.sh A|B
set -u
PART="${1:?A or B}"
HERE="E:/统一计算大模型-迁移包/增量-20260902/潮流线-同拓扑留出补测-20260904"
LEG="E:/潮流线优化包-20260821/gridfm-data-legacy-holdout"
cd "$HERE" || exit 1
mkdir -p size_results nk_logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
run() { # label in_root out_root extra...
  lab="$1"; inr="$2"; outr="$3"; shift 3
  python encoder_probe_nk.py --in-root "$inr" --in-split val --out-root "$outr" --out-split transfer --device cpu \
      --label "$lab" --out-json "size_results/$lab.json" "$@" > "nk_logs/$lab.log" 2>&1
  echo "$lab exit $? $(date +%H:%M:%S)" >> nk_logs/size_groups_summary.log
}
echo "part $PART start $(date +%H:%M:%S)" >> nk_logs/size_groups_summary.log
if [ "$PART" = "A" ]; then
  for g in 400 700 1000; do run "probe_legacy_by_size_$g" "$LEG" "$LEG" --out-families "synth${g}_" --probes-on out; done
  R="$HERE/gridfm-data-nk-enc/N0"
  run "probe_nk_by_size_N0_src" "$R" "$R"
  for g in 400 700 1000; do run "probe_nk_by_size_N0_$g" "$R" "$R" --out-families "synth${g}_" --probes-on out; done
else
  for k in 1 2; do
    R="$HERE/gridfm-data-nk-enc/N$k"
    run "probe_nk_by_size_N${k}_src" "$R" "$R"
    for g in 400 700 1000; do run "probe_nk_by_size_N${k}_$g" "$R" "$R" --out-families "synth${g}_" --probes-on out; done
  done
fi
echo "part $PART done $(date +%H:%M:%S)" >> nk_logs/size_groups_summary.log
