#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

device=cuda
suggest_profile=false
args=()
while (($#)); do
    case "$1" in
        --suggest-profile)
            suggest_profile=true
            shift
            ;;
        --device)
            if (($# < 2)); then
                echo "Error: --device requires cuda or mps" >&2
                exit 2
            fi
            device=$2
            shift 2
            ;;
        --device=*)
            device=${1#*=}
            shift
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

case "$device" in
    cuda|mps) ;;
    *)
        echo "Error: --device requires cuda or mps" >&2
        exit 2
        ;;
esac

if [[ "$suggest_profile" == true ]]; then
    exec uv run python -m pycoder.hardware --device "$device" "${args[@]}"
fi

config=configs/train_124m.yaml
if [[ "$device" == cuda ]]; then
    config=configs/train_124m_cuda.yaml
fi

exec uv run python scripts/train.py --model configs/model_124m.yaml \
    --config "$config" --device "$device" "${args[@]}"
