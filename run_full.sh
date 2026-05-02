#!/bin/bash
# =============================================================
#  Run full evaluation pipeline: 7 modes x 3 datasets
#  Usage: bash run_full.sh [model] [workers]
#  Example: bash run_full.sh gpt-4o 15
# =============================================================

MODEL=${1:-gpt-4o}
WORKERS=${2:-15}

echo "==========================================="
echo " MODEL: $MODEL | WORKERS: $WORKERS"
echo " RUNNING: BASELINE + ENHANCED + S1-S5"
echo "==========================================="

MODES=(
    "baseline|"
    "enhanced|--enhanced"
    "s1_prompt|--strategy s1_prompt"
    "s2_compress|--strategy s2_compress"
    "s3_cot|--strategy s3_cot"
    "s4_twostage|--strategy s4_twostage"
    "s5_fewshot|--strategy s5_fewshot"
)

TOTAL=${#MODES[@]}
COUNT=0

for entry in "${MODES[@]}"; do
    IFS='|' read -r name flag <<< "$entry"
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] $name"
    echo "-------------------------------------------"
    uv run python main.py --dataset all --model "$MODEL" --workers "$WORKERS" $flag
    echo "[$COUNT/$TOTAL] $name DONE"
done

echo ""
echo "==========================================="
echo " All modes done. Rescoring..."
echo "==========================================="
uv run python rescore.py

echo ""
echo "==========================================="
echo " Generating RESULTS.md..."
echo "==========================================="
uv run python generate_results_md.py

echo ""
echo "==========================================="
echo " ALL DONE! Check results/ and RESULTS.md"
echo "==========================================="
