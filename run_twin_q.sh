#!/bin/bash
# run_twin_q.sh — Run CRL with Twin-Q critics.
# Usage: bash scripts/run_twin_q.sh <env_name> [extra_args...]
# Example: bash scripts/run_twin_q.sh ant_soccer --total_env_steps=50000000

set -e

ENV=${1:?Usage: bash scripts/run_twin_q.sh <env_name> [extra_args...]}
shift

# Paths
JAXGCRL_ROOT="${JAXGCRL_ROOT:-/workspace/jaxgcrl}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_DIR="${JAXGCRL_ROOT}/jaxgcrl/agents/crl"

echo "=== Twin-Q CRL on ${ENV} ==="

# Backup originals
cp "${AGENT_DIR}/crl.py"     "${AGENT_DIR}/crl.py.bak"
cp "${AGENT_DIR}/losses.py"  "${AGENT_DIR}/losses.py.bak"

# Copy Twin-Q variants
cp "${SCRIPT_DIR}/crl_variants/crl_2q.py"      "${AGENT_DIR}/crl.py"
cp "${SCRIPT_DIR}/crl_variants/losses_2q.py"    "${AGENT_DIR}/losses.py"

# Run
cd "${JAXGCRL_ROOT}"
python run.py crl --env="${ENV}" "$@"

# Restore originals
mv "${AGENT_DIR}/crl.py.bak"     "${AGENT_DIR}/crl.py"
mv "${AGENT_DIR}/losses.py.bak"  "${AGENT_DIR}/losses.py"

echo "=== Done. Originals restored. ==="
