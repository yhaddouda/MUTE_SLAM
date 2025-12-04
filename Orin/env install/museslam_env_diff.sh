#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path/to/environment.yaml> [target_env=muteslam]"
  exit 2
fi

YAML_PATH="$1"
TARGET_ENV="${2:-muteslam}"

# Run the diff with Jetson profile and mute_slam prefixes
python env_diff_from_yaml.py "$YAML_PATH" "$TARGET_ENV" --out-prefix mute_slam --profile jetson
