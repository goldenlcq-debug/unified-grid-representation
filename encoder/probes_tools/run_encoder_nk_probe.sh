#!/usr/bin/env bash
# 编码器 §1.1/§1.2 探针在配对 N-0/N-1/N-2 新方式上的复跑（seed0 编码器 v1）。
# 前提：已用 assemble_nk.py 把 nk_raw/encoder_val → gridfm-data-nk/N{k}/families/val、nk_raw/encoder_transfer → .../transfer 装配好。
set -u
HERE="E:/统一计算大模型-迁移包/增量-20260902/潮流线-同拓扑留出补测-20260904"
mkdir -p "$HERE/nk_results"
cd "$HERE" || exit 1
for k in 0 1 2; do
  python encoder_probe_nk.py --in-root "$HERE/gridfm-data-nk-enc/N$k" --in-split val --out-root "$HERE/gridfm-data-nk-enc/N$k" --out-split transfer \
      --label "paired_N$k" --out-json "nk_results/probe_encoder_N$k.json" > "nk_logs/probe_encoder_N$k.log" 2>&1
  echo "probe N$k exit $?" >> "nk_logs/probe_encoder_N$k.log"
done
echo "all done"
