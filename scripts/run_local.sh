#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pico_dir="$(cd "${script_dir}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/utf8-corpus.txt [Pico train.py options]" >&2
  exit 2
fi

dataset="$1"
shift
if [[ ! -f "${dataset}" || ! -r "${dataset}" ]]; then
  echo "Corpus file is missing or unreadable: ${dataset}" >&2
  exit 2
fi

"${script_dir}/setup_core.sh"
python "${pico_dir}/prepare_data.py" "${dataset}"
python "${pico_dir}/train.py" --device cpu --dataset "${dataset}" "$@"
