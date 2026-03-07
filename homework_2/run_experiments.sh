#!/usr/bin/env bash
# run_experiments.sh
# ──────────────────────────────────────────────────────────────────────────────
# (OPTIONAL for you to use)
# Convenience script to run the full experiment grid for the assignment.
# Results (CSV + PNG learning curves) are saved to ./results/.
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh                     # default: ogbg-molhiv
#   DATASET=MUTAG ./run_experiments.sh       # override dataset
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DATASET="${DATASET:-ogbg-molhiv}"
HIDDEN=64
LAYERS=4
DROPOUT=0.5
LR=0.001
BATCH=64
EPOCHS=200
PATIENCE=20
SEEDS="0 1 2"

echo "════════════════════════════════════════════════════════════════"
echo "  Dataset  : ${DATASET}"
echo "  Seeds    : ${SEEDS}"
echo "════════════════════════════════════════════════════════════════"

echo -e "\n[1/5] MLP Baseline | pool=mean"
python train.py \
  --dataset  "${DATASET}" \
  --model    mlp \
  --pool     mean \
  --hidden   ${HIDDEN} \
  --layers   ${LAYERS} \
  --dropout  ${DROPOUT} \
  --lr       ${LR} \
  --batch_size ${BATCH} \
  --epochs   ${EPOCHS} \
  --patience ${PATIENCE} \
  --seeds    ${SEEDS}

echo -e "\n[2/5] GIN | pool=mean"
python train.py \
  --dataset  "${DATASET}" \
  --model    gin \
  --pool     mean \
  --hidden   ${HIDDEN} \
  --layers   ${LAYERS} \
  --dropout  ${DROPOUT} \
  --lr       ${LR} \
  --batch_size ${BATCH} \
  --epochs   ${EPOCHS} \
  --patience ${PATIENCE} \
  --seeds    ${SEEDS}

echo -e "\n[3/5] GIN | pool=sum"
python train.py \
  --dataset  "${DATASET}" \
  --model    gin \
  --pool     sum \
  --hidden   ${HIDDEN} \
  --layers   ${LAYERS} \
  --dropout  ${DROPOUT} \
  --lr       ${LR} \
  --batch_size ${BATCH} \
  --epochs   ${EPOCHS} \
  --patience ${PATIENCE} \
  --seeds    ${SEEDS}

echo -e "\n[4/5] GCN | pool=mean"
python train.py \
  --dataset  "${DATASET}" \
  --model    gcn \
  --pool     mean \
  --hidden   ${HIDDEN} \
  --layers   ${LAYERS} \
  --dropout  ${DROPOUT} \
  --lr       ${LR} \
  --batch_size ${BATCH} \
  --epochs   ${EPOCHS} \
  --patience ${PATIENCE} \
  --seeds    ${SEEDS}

echo -e "\n[5/5] GCN | pool=sum"
python train.py \
  --dataset  "${DATASET}" \
  --model    gcn \
  --pool     sum \
  --hidden   ${HIDDEN} \
  --layers   ${LAYERS} \
  --dropout  ${DROPOUT} \
  --lr       ${LR} \
  --batch_size ${BATCH} \
  --epochs   ${EPOCHS} \
  --patience ${PATIENCE} \
  --seeds    ${SEEDS}

echo -e "\n[Bonus] GIN | pool=attention"
python train.py \
  --dataset  "${DATASET}" \
  --model    gin \
  --pool     attention \
  --hidden   ${HIDDEN} \
  --layers   ${LAYERS} \
  --dropout  ${DROPOUT} \
  --lr       ${LR} \
  --batch_size ${BATCH} \
  --epochs   ${EPOCHS} \
  --patience ${PATIENCE} \
  --seeds    ${SEEDS}

echo -e "\n════════════════════════════════════════════════════════════════"
echo "  All experiments finished.  Results in ./results/summary.csv"
echo "════════════════════════════════════════════════════════════════"
