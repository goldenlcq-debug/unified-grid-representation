#!/usr/bin/env bash
# 任务书-编码器探针按规模分组（2026-09-06）：A 原始探针集分组 + B 配对集分组。全部 CPU（本机 GPU 被另一会话占用 6.2/8.1 GB，避免挤爆）。
set -u
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
echo "start $(date +%H:%M:%S)" >> nk_logs/size_groups_summary.log
# A：原始探针集（legacy holdout：val 8 族 = 源网；transfer 12 族按 synth400_/700_/1000_ 分组）
run probe_legacy_by_size_src "$LEG" "$LEG"
for g in 400 700 1000; do run "probe_legacy_by_size_$g" "$LEG" "$LEG" --out-families "synth${g}_" --probes-on out; done
# B：配对集 N-0/N-1/N-2 同样四组
for k in 0 1 2; do
  R="$HERE/gridfm-data-nk-enc/N$k"
  run "probe_nk_by_size_N${k}_src" "$R" "$R"
  for g in 400 700 1000; do run "probe_nk_by_size_N${k}_$g" "$R" "$R" --out-families "synth${g}_" --probes-on out; done
done
echo "all done $(date +%H:%M:%S)" >> nk_logs/size_groups_summary.log
