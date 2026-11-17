#!/usr/bin/env bash
set -euo pipefail

python -m pip install --quiet --upgrade pip
python -m pip install --quiet pytest tqdm
python -m pip install --quiet "transformers>=4.44"

python -c "from bench.env import write_env_file; print('Environment recorded at:', write_env_file())"
