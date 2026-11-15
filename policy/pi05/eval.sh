#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [--key value ...]
# bash eval.sh click_bell random checkpoints/click_bell 0
#
# <checkpoint_path>: the task's checkpoint folder, i.e. the Hugging Face folder <task_name>/ (params/, assets/),
# by default downloaded to policy/pi05/checkpoints/<task_name> (README §2). The task's openpi config is chosen
# from <task_name> (TRAIN_CONFIGS in pi_model.py).
# ----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -z "${PYTHON_BIN:-}" && -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

POLICY_NAME=pi05

USAGE="Usage: bash eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [--key value ...]"
TASK_NAME="${1:?$USAGE}"
SETTING="${2:?$USAGE}"
CHECKPOINT_PATH="${3:?$USAGE}"
shift 3
GPU_ID=0
if [[ $# -gt 0 && "$1" != --* ]]; then
    GPU_ID="$1"
    shift
fi
EXTRA_ARGS=("$@")

# A relative path is taken from the current directory, else from policy/pi05.
if [[ ! -d "$CHECKPOINT_PATH" && -d "$SCRIPT_DIR/$CHECKPOINT_PATH" ]]; then
    CHECKPOINT_PATH="$SCRIPT_DIR/$CHECKPOINT_PATH"
fi
if [[ ! -d "$CHECKPOINT_PATH/params" ]]; then
    echo "Error: no checkpoint at $CHECKPOINT_PATH (it needs params/ and assets/)." >&2
    echo "Download the Hugging Face folder $TASK_NAME/ of owen1257173018/robosyn-pi05 (README §2)." >&2
    exit 1
fi
CHECKPOINT_PATH="$(cd "$CHECKPOINT_PATH" && pwd)"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Inference peaks at about 6.7 GiB; raise the fraction on cards below 20 GB, e.g. when no simulator shares the GPU.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"


echo "========================================="
echo "  π0.5 Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  GPU:        $GPU_ID"
echo "========================================="

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: cannot find Python command: $PYTHON_BIN" >&2
    exit 1
fi

export PI0_VENV_DIR="$VENV_DIR"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$SCRIPT_DIR/src:$SCRIPT_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$EMBODICHAIN_ROOT" ]]; then
    export PYTHONPATH="$EMBODICHAIN_ROOT:$PYTHONPATH"
fi
cd "$REPO_ROOT" # move to RoboSynChallenge root

PYTHONWARNINGS=ignore::UserWarning \
EMBODICHAIN_SIM_EXIT_PROCESS=0 \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    "${EXTRA_ARGS[@]}"
