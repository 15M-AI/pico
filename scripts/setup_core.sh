#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pico_dir="$(cd "${script_dir}/.." && pwd)"
core_dir="${OPEN_MIND_CORE_DIR:-}"

if [[ -z "${core_dir}" ]]; then
  if [[ -d /kaggle/working ]]; then
    core_dir="/kaggle/working/open-mind"
  else
    core_dir="${HOME}/.cache/open-mind/core"
  fi
fi

if [[ ! -f "${core_dir}/Cargo.toml" || ! -d "${core_dir}/open-mind-train" ]]; then
  if [[ -e "${core_dir}" ]]; then
    echo "OPEN_MIND_CORE_DIR does not point to an Open Mind checkout: ${core_dir}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${core_dir}")"
  git clone --depth 1 https://github.com/15M-AI/open-mind.git "${core_dir}"
fi

python - <<'PY'
import sys
try:
    import numpy
    import torch
except ImportError as error:
    raise SystemExit(f"Missing training dependency {error.name}; install CUDA-enabled PyTorch and NumPy first.")

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {numpy.__version__}")
PY

python -m pip install --no-deps -e "${core_dir}/open-mind-format/python"
python -m pip install --no-deps -e "${core_dir}/open-mind-train"

echo "Pico project: ${pico_dir}"
echo "Open Mind core: ${core_dir}"
