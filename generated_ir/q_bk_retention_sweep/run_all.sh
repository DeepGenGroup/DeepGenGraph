#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$experiment_dir/run.sh" baseline_16words before
bash "$experiment_dir/run.sh" retained_24words first
bash "$experiment_dir/run.sh" baseline_16words between
bash "$experiment_dir/run.sh" retained_32words first
bash "$experiment_dir/run.sh" baseline_16words after
