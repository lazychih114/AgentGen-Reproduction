#!/bin/bash

# Add the src directory to PYTHONPATH
# export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Base directories
DATA_DIR="data"
RESULTS_DIR="results"
ITERATION="it1"  # Current iteration number
PREV_ITERATION="it0"  # Previous iteration number

# Input data paths
CONTEXT_PATH="${DATA_DIR}/inspiration_corpus/lima_example.json"  # Please change this to your own context path if needed. Current lima_example.json is for testing.
SEED_ENV_PATH="${DATA_DIR}/${PREV_ITERATION}/merged_envs_${PREV_ITERATION}.json"

# Output paths for current iteration
EVOLVED_DESC="${RESULTS_DIR}/${ITERATION}/environment_specification.json"
GENERATED_DOMAIN="${RESULTS_DIR}/${ITERATION}/generated_domain.json"
DOMAIN_WITH_NL="${RESULTS_DIR}/${ITERATION}/generated_domain_wnl.json"
DOMAIN_WITH_PROB="${RESULTS_DIR}/${ITERATION}/generated_domain_wprob.json"
EVOLVE_GPT="${RESULTS_DIR}/${ITERATION}/evolve_gpt.json"

# Merged data paths
SOLVABLE_ENV="${DATA_DIR}/${ITERATION}/solvable_envs.json"
MERGED_ENV="${DATA_DIR}/${ITERATION}/merged_envs_${ITERATION}.json"
MERGED_GPT="${RESULTS_DIR}/${ITERATION}/merged_gpt_${ITERATION}.json"

# Create necessary directories
mkdir -p "${RESULTS_DIR}/${ITERATION}"
mkdir -p "${DATA_DIR}/${ITERATION}"

# Generate Environment Specification
python scripts/1_environment_specification.py \
    --context_path "${CONTEXT_PATH}" \
    --data_path "${SEED_ENV_PATH}" \
    --output_path "${EVOLVED_DESC}" \
    --api_type "openai" \
    --model "gpt-4o" \
    --n_process 5

python scripts/2_generate_domain.py \
    --data_path "${EVOLVED_DESC}" \
    --output_path "${GENERATED_DOMAIN}" \
    --prompt_file "prompt/desc2domain" \
    --model "gpt-4o" \
    --n_process 5

python scripts/3_generate_nl_interface.py \
    --data_path "${GENERATED_DOMAIN}" \
    --output_path "${DOMAIN_WITH_NL}" \
    --api_type "openai" \
    --model "gpt-4o" \
    --n_process 5

python scripts/4_generate_problems.py \
    --data_path "${DOMAIN_WITH_NL}" \
    --output_path "${DOMAIN_WITH_PROB}" \
    --model "gpt-4o" \
    --n_process 5 \
    --prob_num 10   \
    --api_type "openai"

# The script will download and compile FD in the first-time run.
python scripts/5_generate_gpt_data.py \
    --data_path "${DOMAIN_WITH_PROB}" \
    --output_path "${EVOLVE_GPT}" \
    --solvable_path "${SOLVABLE_ENV}" 

# # Merge current iteration data
python scripts/6_merge_json.py \
    --d1_path "${SOLVABLE_ENV}" \
    --d2_path "${DATA_DIR}/${PREV_ITERATION}/merged_envs_${PREV_ITERATION}.json" \
    --output_path "${MERGED_ENV}"

python scripts/6_merge_json.py \
    --d1_path "${RESULTS_DIR}/${PREV_ITERATION}/merged_gpt_${PREV_ITERATION}.json" \
    --d2_path "${EVOLVE_GPT}" \
    --output_path "${MERGED_GPT}"