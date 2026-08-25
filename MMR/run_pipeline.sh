#!/bin/bash
# =====================================================================
# MemLoc end-to-end pipeline (MMR retrieval -> REL localization ->
# generation -> evaluation).
#
# Usage:
#   bash run_pipeline.sh <dataset> <data_path> [bge_m3_path] [output_tag]
#
#   dataset     one of: longmemeval_s | longmemeval_m | locomo | longmtbench
#   data_path   path to the raw dataset json
#   bge_m3_path (optional) BGE-M3 checkpoint (default checkpoints/bge-m3)
#   output_tag  (optional) sub-directory under outputs/ to isolate this
#               run's artifacts.  Required when the same dataset is
#               processed in several shards (e.g. the ten
#               longmemeval_m_part{N}_all.json files); without it every
#               shard writes into outputs/ and overwrites the previous one.
#
# Environment:
#   OPENAI_API_KEY / OPENAI_BASE_URL  OpenAI-compatible endpoint used by
#                                      the extraction / generation / judge
#                                      scripts (paper: GPT-4o / GPT-4o mini)
#   EXTRACT_MODEL   default: gpt-4o-mini
#   GENERATOR_MODEL default: gpt-4o-mini
#   JUDGE_MODEL     default: gpt-4o
#   LOCATOR_MODEL   served locator model name (vLLM); default "judge"
#                   (same convention as REL/rel_generate.py).  Set
#                   LOCATOR_MODEL="" to skip the REL localization stage.
#   LOCATOR_BASE_URL  locator endpoint (optional).  Defaults to
#                   OPENAI_BASE_URL when set, otherwise the local vLLM
#                   default http://localhost:8000/v1.  Useful when the
#                   locator is served by a vLLM instance separate from the
#                   GPT extraction/generation endpoint.
#
# Example:
#   bash run_pipeline.sh longmemeval_s data_example/LongMemEval_S_example.json
#   bash run_pipeline.sh longmemeval_m data/longmemeval_m_part1_all.json checkpoints/bge-m3 part1
# =====================================================================
set -e

DATASET=${1:?"Usage: bash run_pipeline.sh <dataset> <data_path> [bge_m3_path] [output_tag]"}
DATA_PATH=${2:?"Usage: bash run_pipeline.sh <dataset> <data_path> [bge_m3_path] [output_tag]"}
BGE_PATH=${3:-checkpoints/bge-m3}
OUTPUT_TAG=${4:-}
WORKERS=${WORKERS:-64}
ENCODE_BATCH=${ENCODE_BATCH:-64}
PYTHON=${PYTHON:-python}

EXTRACT_MODEL=${EXTRACT_MODEL:-gpt-4o-mini}
GENERATOR_MODEL=${GENERATOR_MODEL:-gpt-4o-mini}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o}
LOCATOR_MODEL=${LOCATOR_MODEL:-judge}

# Isolate per-run artifacts under outputs/<output_tag>/ when set, so the
# ten longmemeval_m shards can be processed one by one without clobbering
# each other's results.  Without an output_tag the historical outputs/
# layout is kept.
if [ -n "${OUTPUT_TAG}" ]; then
    OUT=outputs/${OUTPUT_TAG}
else
    OUT=outputs
fi
mkdir -p ${OUT}

echo "=========================================================="
echo "[0/8] Dataset: ${DATASET}  Data: ${DATA_PATH}"
echo "=========================================================="

# ---------------------------------------------------------------------
# 1+2. Event summary + summary/keyword extraction.  Both scripts resume
#      internally (processed questions / content-addressed session cache),
#      so they ALWAYS run -- a [ ! -f ] guard would wrongly skip an
#      interrupted, partially-written output.  They are independent and
#      run concurrently, WORKERS LLM threads each.
# ---------------------------------------------------------------------
echo "[1/8]+[2/8] Event extraction + summary/keyword extraction (parallel)"
EVENTS=${OUT}/events_${DATASET}.json
STEP1_PID=""
"${PYTHON}" event_summary.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --output_path ${OUT}/events_${DATASET}.json \
    --model ${EXTRACT_MODEL} \
    --workers ${WORKERS} &
STEP1_PID=$!

"${PYTHON}" keyword_summary.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --output_path ${OUT}/memory_units_${DATASET}.json \
    --model ${EXTRACT_MODEL} \
    --workers ${WORKERS}
wait "${STEP1_PID}"

# ---------------------------------------------------------------------
# 3. Multi-granularity memory construction (BGE-M3 embeddings, memory
#    graph with semantic + temporal edges)
# ---------------------------------------------------------------------
echo "[3/8] Memory construction"
EMB=${OUT}/memory_embeddings_${DATASET}.pt
GRAPHS=${OUT}/memory_graphs_${DATASET}.pt
if [ ! -f "${EMB}" ]; then
    python Memory_Construct.py \
        --dataset ${DATASET} \
        --data_path ${DATA_PATH} \
        --event_path ${EVENTS} \
        --encoder_path ${BGE_PATH} \
        --output_dir ${OUT} \
        --encode_batch_size ${ENCODE_BATCH}
fi

# ---------------------------------------------------------------------
# 4. MMR retrieval (inner-memory routing + cross-memory propagation)
# ---------------------------------------------------------------------
echo "[4/8] MMR retrieval"
RETRIEVAL=${OUT}/retrieval_${DATASET}.jsonl
python Dynamic_Router.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --embedding_path ${EMB} \
    --graph_path ${GRAPHS} \
    --output_path ${RETRIEVAL}

# ---------------------------------------------------------------------
# 5. Retrieval evaluation (Long-MT-Bench+ has no retrieval GT and is
#    skipped inside the script)
# ---------------------------------------------------------------------
echo "[5/8] Retrieval evaluation"
python evaluate_retrieval.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --retrieval_path ${RETRIEVAL} \
    --ks 3,5,10 || true

# ---------------------------------------------------------------------
# 6. REL localization (optional; needs the trained locator served
#    behind an OpenAI-compatible endpoint)
# ---------------------------------------------------------------------
LOCALIZATION=""
if [ -n "${LOCATOR_MODEL}" ]; then
    echo "[6/8] REL evidence localization (locator: ${LOCATOR_MODEL} @ ${LOCATOR_BASE_URL:-${OPENAI_BASE_URL:-http://localhost:8000/v1}})"
    LOCALIZATION=${OUT}/localization_${DATASET}.jsonl
    python locator_inference.py \
        --dataset ${DATASET} \
        --data_path ${DATA_PATH} \
        --retrieval_path ${RETRIEVAL} \
        --units_path ${OUT}/memory_units_full_${DATASET}.json \
        --model ${LOCATOR_MODEL} \
        --workers ${WORKERS} \
        --output_path ${LOCALIZATION}
else
    echo "[6/8] REL evidence localization skipped (LOCATOR_MODEL not set)"
fi

# ---------------------------------------------------------------------
# 7. Cue-guidance generation (Multi-Granular Reasoning prompt)
# ---------------------------------------------------------------------
echo "[7/8] Answer generation"
GENERATION=${OUT}/generation_${DATASET}.jsonl
python generation.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --retrieval_path ${RETRIEVAL} \
    --units_path ${OUT}/memory_units_full_${DATASET}.json \
    $( [ -n "${LOCALIZATION}" ] && echo --localization_path ${LOCALIZATION} ) \
    --topk 3 \
    --model ${GENERATOR_MODEL} \
    --output_path ${GENERATION}

# ---------------------------------------------------------------------
# 8. LLM-as-a-judge evaluation (Answer Verification prompt)
# ---------------------------------------------------------------------
echo "[8/8] Answer verification"
python generation_judge.py \
    --input_path ${GENERATION} \
    --output_path ${OUT}/generation_${DATASET}_judge.jsonl \
    --model ${JUDGE_MODEL}

echo "=========================================================="
echo "Pipeline finished. Results in ${OUT}/"
echo "=========================================================="
