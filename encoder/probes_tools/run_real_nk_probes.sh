#!/usr/bin/env bash
# 任务书 2026-09-06 C 项：等三张真实网（3120 / PEGASE1354 / PEGASE2869）配对造数结束 → 装配（三变体交集）→ 与全部训练语料哈希核验 → 编码器探针 N-0/N-1/N-2（CPU）。
set -u
HERE="E:/统一计算大模型-迁移包/增量-20260902/潮流线-同拓扑留出补测-20260904"
cd "$HERE" || exit 1
mkdir -p size_results nk_logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# 1) 等造数
while true; do
  n=$(grep -c "exit" nk_logs/produce_real_summary.log 2>/dev/null); n=${n:-0}
  [ "$n" -ge 3 ] && break
  sleep 60
done
echo "produce done: $(cat nk_logs/produce_real_summary.log | tr '\n' ';') $(date +%H:%M:%S)" >> nk_logs/real_nk_summary.log
# 2) 装配（只保留三变体都收敛的样本号）
python assemble_nk.py --raw "$HERE/nk_raw/real" --dst "$HERE/gridfm-data-nk-real" --split val > nk_logs/assemble_real.log 2>&1
echo "assemble exit $? $(date +%H:%M:%S)" >> nk_logs/real_nk_summary.log
# 3) 分离核验：全部训练语料（与 disjoint_check_nk_sets.json 同 6 个根，9,690 文件）
TR="E:/潮流线优化包-20260821/gridfm-data/families/train|E:/潮流线优化包-20260821/gridfm-data/lib118|E:/潮流线优化包-20260821/gridfm-data/lib_ieee39|E:/潮流线优化包-20260821/gridfm-data/lib_cepri36|E:/潮流线优化包-20260821/大规模算例语料交付-20260821/大规模算例语料交付-20260821/gridfm-data-large/families/train|E:/潮流线优化包-20260821/gridfm-data-large-ext/families/train"
NEW=()
for fam in mp_case3120sp mp_case1354pegase mp_case2869pegase; do
  for k in 0 1 2; do NEW+=(--new "real_${fam}_N$k=$HERE/gridfm-data-nk-real/N$k/families/val/$fam"); done
done
python check_disjoint_nk.py --train "$TR" "${NEW[@]}" --out size_results/disjoint_check_real_nk_sets.json > nk_logs/disjoint_real.log 2>&1
echo "disjoint exit $? $(date +%H:%M:%S)" >> nk_logs/real_nk_summary.log
# 4) 探针
for pair in "case3120sp:mp_case3120sp" "PEGASE1354:mp_case1354pegase" "PEGASE2869:mp_case2869pegase"; do
  g="${pair%%:*}"; fam="${pair##*:}"
  for k in 0 1 2; do
    python encoder_probe_large.py --grid "${g}_N$k" --root "$HERE/gridfm-data-nk-real/N$k/families/val/$fam" --device cpu --label P3 \
        --out-json "size_results/probe_real_${g}_N$k.json" > "nk_logs/probe_real_${g}_N$k.log" 2>&1
    echo "$g N$k exit $? $(date +%H:%M:%S)" >> nk_logs/real_nk_summary.log
  done
done
echo "all done $(date +%H:%M:%S)" >> nk_logs/real_nk_summary.log
