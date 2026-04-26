#!/bin/bash
# run_pipeline.sh
# Execute the full CGM-Rec pipeline for ml_100k and save all readable outputs
# cgm SCORER_TYPE=gat DEVICE=auto LLM_PROVIDER=mock ./run_pipeline.sh

cd "$(dirname "$0")"

# Make sure we use the correct virtual environment Python
PYTHON_EXEC="cgmvenv/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Virtual environment not found. Falling back to system python3."
    PYTHON_EXEC="python3"
fi

DATASET="${DATASET:-ml_100k}"
LLM_PROVIDER="${LLM_PROVIDER:-timelygpt}"
SCORER_TYPE="${SCORER_TYPE:-linear}"
DEVICE="${DEVICE:-auto}"

if [ -z "${OUT_DIR:-}" ]; then
    if [ "$SCORER_TYPE" = "linear" ]; then
        OUT_DIR="outputs/${DATASET}_${LLM_PROVIDER}"
    else
        OUT_DIR="outputs/${DATASET}_${LLM_PROVIDER}_${SCORER_TYPE}"
    fi
fi
mkdir -p "$OUT_DIR"

SCORER_ARGS=(--scorer-type "$SCORER_TYPE")
if [ "$SCORER_TYPE" = "gat" ]; then
    MODEL_FILE="$OUT_DIR/gat_scorer.pt"
    SCORER_ARGS+=(--device "$DEVICE")
    MODEL_SAVE_ARGS=(--model-path "$MODEL_FILE")
    MODEL_LOAD_ARGS=(--model-path "$MODEL_FILE")
else
    MODEL_FILE="$OUT_DIR/scorer_weights.json"
    MODEL_SAVE_ARGS=(--model-json "$MODEL_FILE")
    MODEL_LOAD_ARGS=(--model-json "$MODEL_FILE")
fi

echo "==========================================================="
echo " Starting CGM-Rec full pipeline execution for: $DATASET"
echo " Scorer: $SCORER_TYPE"
if [ "$SCORER_TYPE" = "gat" ]; then
    echo " Device: $DEVICE"
fi
echo "==========================================================="

echo "1. Phase 1: Data Inspection..."
$PYTHON_EXEC main.py --dataset "$DATASET" --view data --output-json "$OUT_DIR/phase1_data.json" > "$OUT_DIR/phase1_data.log" 2>&1
echo "   -> Saved to $OUT_DIR/phase1_data.log & .json"

echo "2. Phase 2: Seed Graph Construction..."
$PYTHON_EXEC main.py --dataset "$DATASET" --view seed-graph --dump-graph-json "$OUT_DIR/seed_graph.json" --output-json "$OUT_DIR/phase2_seed.json" > "$OUT_DIR/phase2_seed.log" 2>&1
echo "   -> Saved to $OUT_DIR/phase2_seed.log & seed_graph.json"

echo "3. Phase 3: Train Semantic Scorer (Dual Optimization)..."
# We save the model weights and JSON summary metrics
$PYTHON_EXEC main.py --dataset "$DATASET" --view phase3-train "${SCORER_ARGS[@]}" --epochs 5 "${MODEL_SAVE_ARGS[@]}" --output-json "$OUT_DIR/phase3_train.json" > "$OUT_DIR/phase3_train.log" 2>&1
echo "   -> Saved to $OUT_DIR/phase3_train.log, .json & $MODEL_FILE"

echo "4. Phase 3: Test Semantic Scorer (Static Evaluation)..."
$PYTHON_EXEC main.py --dataset "$DATASET" --view phase3-test "${SCORER_ARGS[@]}" --epochs 5 "${MODEL_LOAD_ARGS[@]}" --output-json "$OUT_DIR/phase3_test.json" > "$OUT_DIR/phase3_test.log" 2>&1
echo "   -> Saved to $OUT_DIR/phase3_test.log & .json"

echo "5. Phase 4: Online Evaluation with Structural Edits (Pre-quential Evaluation)..."
$PYTHON_EXEC main.py --dataset "$DATASET" --view phase4-test-online "${SCORER_ARGS[@]}" --epochs 5 "${MODEL_LOAD_ARGS[@]}" --output-json "$OUT_DIR/phase4_online.json" > "$OUT_DIR/phase4_online.log" 2>&1
echo "   -> Saved to $OUT_DIR/phase4_online.log & .json"

echo "6. Phase 5: Online Evaluation with LLM..."
# Using --max-test-samples 0 to run on the full test set. (Note: using mock provider for safety; change to openai for real API)
$PYTHON_EXEC main.py --dataset "$DATASET" --view phase5-test-llm "${SCORER_ARGS[@]}" --epochs 5 "${MODEL_LOAD_ARGS[@]}" --llm-provider "$LLM_PROVIDER" --max-test-samples 0 --output-json "$OUT_DIR/phase5_llm.json" > "$OUT_DIR/phase5_llm.log" 2>&1
echo "   -> Saved to $OUT_DIR/phase5_llm.log & .json"

echo "==========================================================="
echo " Pipeline execution complete! Check $OUT_DIR/"
echo " All NDCG and HIT metrics for ks=(1,5,10,20) have been saved in the JSON and log files."
echo "==========================================================="
