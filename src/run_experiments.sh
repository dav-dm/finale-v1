#!/bin/bash
# run_experiments.sh
# This script runs experiments in a combinatorial fashion.
# It accepts the following parameters:
#   --datasets (-d)     : space-separated list of datasets
#   --seed (-s)         : comma-separated list of seeds or an interval (e.g., 0-100)
#   --k-seed (-ks)      : comma-separated list of k-seeds or an interval (e.g., 0-100) [optional]
#   --approach (-a)     : comma-separated list of approaches
#   --is-flat (-f)      : flag indicating flat structure (if present, add to command)
#   --cpu (-c)          : integer number of cores to use
#   --extra-args (-e)   : extra arguments to pass directly to "python main.py"
#   --log-keyword (-lk) : additional string for log_dir

usage() {
    echo "Usage: $0 --datasets <\"sd1 sd2 ...\"> --seed <s1,s2,... or X-Y>"
    echo "           --approach <appr1,appr2,...> [--is-flat] --cpu <num_cores>"
    echo "           [--log-keyword <keyword>] [--extra-args \"<args>\"]"
    exit 1
}

# Default values
DATASETS=""
SEEDS=""
K_SEEDS=""
APPROACHES=""
IS_FLAT=0
CPU_CORES=""
EXTRA_ARGS=""
LOG_KEYWORD=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--datasets)
            DATASETS="$2"
            shift
            ;;
        -s|--seed)
            SEEDS="$2"
            shift
            ;;
        -ks|--k-seed)
            K_SEEDS="$2"
            shift
            ;;
        -a|--approach)
            APPROACHES="$2"
            shift
            ;;
        -f|--is-flat)
            IS_FLAT=1
            ;;
        -c|--cpu)
            CPU_CORES="$2"
            shift
            ;;
        -e|--extra-args)
            EXTRA_ARGS="$2"
            shift
            ;;
        -lk|--log-keyword)
            LOG_KEYWORD="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
    shift
done

# Verify required parameters
if [[ -z "$DATASETS" || -z "$SEEDS" || -z "$APPROACHES" || -z "$CPU_CORES" ]]; then
    usage
fi

# Set flag variable for --is-flat
if [ "$IS_FLAT" -eq 1 ]; then
    FLAT_FLAG=" --is-flat"
else
    FLAT_FLAG=""
fi

# Split comma-separated lists into arrays
IFS=',' read -ra APPROACH_ARR <<< "$APPROACHES"

# Process the --seed parameter:
# If it matches the pattern X-Y, expand the range; otherwise, split by commas.
SEED_ARR=()
if [[ "$SEEDS" =~ ^[0-9]+-[0-9]+$ ]]; then
    IFS='-' read -r start end <<< "$SEEDS"
    for ((i=start; i<=end; i++)); do
        SEED_ARR+=("$i")
    done
else
    IFS=',' read -ra SEED_ARR <<< "$SEEDS"
fi

# Process the --k-seed parameter (optional)
if [[ -n "$K_SEEDS" ]]; then
    K_SEED_ARR=()
    if [[ "$K_SEEDS" =~ ^[0-9]+-[0-9]+$ ]]; then
        IFS='-' read -r kstart kend <<< "$K_SEEDS"
        for ((k=kstart; k<=kend; k++)); do
            K_SEED_ARR+=("$k")
        done
    else
        IFS=',' read -ra K_SEED_ARR <<< "$K_SEEDS"
    fi
fi

# Split datasets into array
read -ra DATASET_ARR <<< "$DATASETS"
# Create a unique tag for the dataset combination
DATASET_TAG=""
for idx in "${!DATASET_ARR[@]}"; do
    d="${DATASET_ARR[$idx]}"
    DATASET_TAG+="d$((idx+1))_${d}_"
done
DATASET_TAG="${DATASET_TAG%_}"  # remove trailing underscore

# Array to store commands
commands=()

# Create experiments for each combination
for approach in "${APPROACH_ARR[@]}"; do

    # Build LOG_DIR once per approach
    LOG_DIR="../results_${DATASET_TAG}_${approach}"

    if [[ -n "$LOG_KEYWORD" ]]; then
        LOG_DIR+="_${LOG_KEYWORD}"
    fi

    LOG_DIR+="_6f_20p"

    mkdir -p "${LOG_DIR}"

    for seed in "${SEED_ARR[@]}"; do

        # If --k-seed is provided, run an additional loop
        if [[ -n "$K_SEEDS" ]]; then

            for kseed in "${K_SEED_ARR[@]}"; do

                CMD="python main.py --datasets ${DATASETS} --approach ${approach} --seed ${seed} --k-seed ${kseed} --log-dir ${LOG_DIR}${FLAT_FLAG}"
                CMD+=" ${EXTRA_ARGS}"

                if [[ -n "$GPU_ID" ]]; then
                    CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} ${CMD}"
                fi

                FULL_CMD="{ echo \"$(printf '=%.0s' {1..100})\" ; \
                    echo \"Command: ${CMD}\" ; \
                    echo \"$(printf '=%.0s' {1..100})\" >&2 ; \
                    echo \"Command: ${CMD}\" >&2 ; \
                    ${CMD} ; \
                } > \"${LOG_DIR}/output.log\" 2>> \"${LOG_DIR}/errors.log\""
                commands+=("$FULL_CMD")
            done

        else

            CMD="python main.py --datasets ${DATASETS} --approach ${approach} --seed ${seed} --log-dir ${LOG_DIR}${FLAT_FLAG}"
            CMD+=" ${EXTRA_ARGS}"

            if [[ -n "$GPU_ID" ]]; then
                CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} ${CMD}"
            fi

            FULL_CMD="{ echo \"$(printf '=%.0s' {1..100})\" ; \
                echo \"Command: ${CMD}\" ; \
                echo \"$(printf '=%.0s' {1..100})\" >&2 ; \
                echo \"Command: ${CMD}\" >&2 ; \
                ${CMD} ; \
            } > \"${LOG_DIR}/output.log\" 2>> \"${LOG_DIR}/errors.log\""
            commands+=("$FULL_CMD")
        fi
    done
done

TOTAL=${#commands[@]}
echo "Total experiments: $TOTAL"

# Run experiments in parallel with progress reporting if GNU parallel is available.
if command -v parallel > /dev/null 2>&1; then
    mkdir -p ../log_dump
    printf "%s\n" "${commands[@]}" | parallel --progress --delay 10 --joblog ../log_dump/joblog.txt -j "$CPU_CORES"
else
    echo "GNU parallel not found; using xargs without progress tracking."
    printf "%s\n" "${commands[@]}" | xargs -P "$CPU_CORES" -I {} bash -c 'sleep 10; eval "$@"' -- {}
fi

echo "All experiments completed."