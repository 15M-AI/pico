#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pico_dir="$(cd "${script_dir}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /kaggle/input/DATASET/corpus.txt [Pico train.py options]" >&2
  exit 2
fi

dataset="$1"
shift
if [[ ! -f "${dataset}" || ! -r "${dataset}" ]]; then
  echo "Kaggle corpus is missing or unreadable: ${dataset}" >&2
  exit 2
fi

if [[ ! -d /kaggle/working ]]; then
  echo "This helper expects a Kaggle notebook environment; use run_local.sh elsewhere." >&2
  exit 2
fi

if [[ -z "${OPEN_MIND_CORE_DIR:-}" ]]; then
  export OPEN_MIND_CORE_DIR=/kaggle/working/open-mind
fi
export PICO_OUTPUT_DIR="${PICO_OUTPUT_DIR:-/kaggle/working/pico-output}"
mkdir -p "${PICO_OUTPUT_DIR}"

"${script_dir}/setup_core.sh"
manifest="${PICO_OUTPUT_DIR}/corpus-manifest.json"
candidate_manifest="$(mktemp "${manifest}.candidate.XXXXXX")"
trap 'rm -f "${candidate_manifest}"' EXIT
python "${pico_dir}/prepare_data.py" "${dataset}" --manifest "${candidate_manifest}"

checkpoint="${PICO_OUTPUT_DIR}/pico-0.1.0-v1.omf"
if [[ -f "${checkpoint}" ]]; then
  if [[ ! -f "${manifest}" ]]; then
    echo "Cannot safely resume: checkpoint exists but its corpus manifest is missing." >&2
    exit 2
  fi
  python - "${manifest}" "${candidate_manifest}" <<'PY'
import json
import sys
from pathlib import Path

previous, current = (json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:])
if previous.get("sha256") != current.get("sha256"):
    raise SystemExit(
        "Cannot resume this checkpoint with a different corpus. "
        "Restore the original corpus or remove the checkpoint explicitly."
    )
PY
fi
mv "${candidate_manifest}" "${manifest}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Enable a Kaggle GPU accelerator before training.")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
PY

output="${PICO_OUTPUT_DIR}/pico-0.1.0.omf"
args=(--device cuda --dataset "${dataset}" --output "${output}")
if [[ -f "${checkpoint}" ]]; then
  args+=(--resume-from "${checkpoint}")
fi
python "${pico_dir}/train.py" "${args[@]}" "$@"

python - <<PY
from pathlib import Path
import hashlib

for path in sorted(Path("${PICO_OUTPUT_DIR}").glob("*.omf")):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path}  {path.stat().st_size} bytes  sha256={digest}")
PY
