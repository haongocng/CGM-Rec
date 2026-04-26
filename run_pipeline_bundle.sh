#!/bin/bash
# run_pipeline_bundle.sh
# Execute the full CGM-Rec pipeline for the Amazon Bundle dataset

cd "$(dirname "$0")"

PYTHON_EXEC="cgmvenv/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Virtual environment not found. Falling back to system python3."
    PYTHON_EXEC="python3"
fi

DATASET="bundle"
OUT_DIR="outputs/${DATASET}_timelygpt"
mkdir -p "$OUT_DIR"

echo "==========================================================="
echo " Starting CGM-Rec full pipeline execution for: $DATASET"
echo "==========================================================="

echo "1. Phase 1: Data Inspection..."
$PYTHON_EXEC main.py --dataset $DATASET --view data --output-json "$OUT_DIR/phase1_data.json" > "$OUT_DIR/phase1_data.log" 2>&1
echo "   -> Saved to $OUT_DIR/phase1_data.log & .json"

echo "2. Phase 2: Seed Graph Construction..."
$PYTHON_EXEC main.py --dataset $DATASET --view seed-graph --dump-graph-json "$OUT_DIR/seed_graph.json" --output-json "$OUT_DIR/phase2_seed.json" > "$OUT_DIR/phase2_seed.log" 2>&1
echo "   -> Saved to $OUT_DIR/phase2_seed.log & seed_graph.json"

echo "3. Phase 3: Train Semantic Scorer (Dual Optimization)..."
$PYTHON_EXEC main.py --dataset $DATASET --view phase3-train --epochs 5 --model-json "$OUT_DIR/scorer_weights.json" --output-json "$OUT_DIR/phase3_train.json" > "$OUT_DIR/phase3_train.log" 2>&1
echo "   -> Saved to $OUT_DIR/phase3_train.log, .json & scorer_weights.json"

echo "4. Phase 3: Test Semantic Scorer (Static Evaluation)..."
$PYTHON_EXEC main.py --dataset $DATASET --view phase3-test --epochs 5 --output-json "$OUT_DIR/phase3_test.json" > "$OUT_DIR/phase3_test.log" 2>&1
echo "   -> Saved to $OUT_DIR/phase3_test.log & .json"

echo "5. Phase 4: Online Evaluation with Structural Edits (Pre-quential Evaluation)..."
$PYTHON_EXEC main.py --dataset $DATASET --view phase4-test-online --epochs 5 --output-json "$OUT_DIR/phase4_online.json" > "$OUT_DIR/phase4_online.log" 2>&1
echo "   -> Saved to $OUT_DIR/phase4_online.log & .json"

echo "6. Phase 5: Online Evaluation with LLM..."
$PYTHON_EXEC main.py --dataset $DATASET --view phase5-test-llm --epochs 5 --llm-provider timelygpt --max-test-samples 0 --output-json "$OUT_DIR/phase5_llm.json" > "$OUT_DIR/phase5_llm.log" 2>&1
echo "   -> Saved to $OUT_DIR/phase5_llm.log & .json"

echo "==========================================================="
echo " Pipeline complete! Check outputs/${DATASET}_timelygpt/"
echo "==========================================================="
