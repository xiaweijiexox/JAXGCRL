#!/bin/bash
# run_tqc.sh — Run TD3 with TQC quantile critics.
# Usage: bash scripts/run_tqc.sh <env_name> [extra_args...]
# Example: bash scripts/run_tqc.sh arm_push_hard
# Example: bash scripts/run_tqc.sh arm_reach --n_critics=3

set -e

ENV=${1:?Usage: bash scripts/run_tqc.sh <env_name> [extra_args...]}
shift

# Paths
JAXGCRL_ROOT="${JAXGCRL_ROOT:-/workspace/jaxgcrl}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_DIR="${JAXGCRL_ROOT}/jaxgcrl/agents/td3"

echo "=== TQC TD3 on ${ENV} ==="

# Backup originals
cp "${AGENT_DIR}/td3.py"       "${AGENT_DIR}/td3.py.bak"
cp "${AGENT_DIR}/losses.py"    "${AGENT_DIR}/losses.py.bak"
cp "${AGENT_DIR}/networks.py"  "${AGENT_DIR}/networks.py.bak"

# Copy TQC variants
cp "${SCRIPT_DIR}/td3_variants/td3_tqc.py"       "${AGENT_DIR}/td3.py"
cp "${SCRIPT_DIR}/td3_variants/losses_tqc.py"     "${AGENT_DIR}/losses.py"
cp "${SCRIPT_DIR}/td3_variants/networks_tqc.py"   "${AGENT_DIR}/networks.py"

# Run (default: 5 critics, 25 quantiles, drop=0)
cd "${JAXGCRL_ROOT}"
python run.py td3 --env="${ENV}" "$@"

# Restore originals
mv "${AGENT_DIR}/td3.py.bak"       "${AGENT_DIR}/td3.py"
mv "${AGENT_DIR}/losses.py.bak"    "${AGENT_DIR}/losses.py"
mv "${AGENT_DIR}/networks.py.bak"  "${AGENT_DIR}/networks.py"

echo "=== Done. Originals restored. ==="
