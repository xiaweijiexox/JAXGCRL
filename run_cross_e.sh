#!/bin/bash
# run_cross_e.sh — Run CRL with Twin-Q + Cross-Episode HER.
# Usage: bash scripts/run_cross_e.sh <env_name> [cross_episode_ratio] [extra_args...]
# Example: bash scripts/run_cross_e.sh ant_u_maze 0.1
# Example: bash scripts/run_cross_e.sh ant_soccer 0.001 --total_env_steps=50000000

set -e

ENV=${1:?Usage: bash scripts/run_cross_e.sh <env_name> [cross_episode_ratio] [extra_args...]}
RATIO=${2:-0.1}
shift 2 2>/dev/null || shift 1

# Paths
JAXGCRL_ROOT="${JAXGCRL_ROOT:-/workspace/jaxgcrl}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_DIR="${JAXGCRL_ROOT}/jaxgcrl/agents/crl"

echo "=== CrossE CRL on ${ENV} (ratio=${RATIO}) ==="

# Backup originals
cp "${AGENT_DIR}/crl.py"     "${AGENT_DIR}/crl.py.bak"
cp "${AGENT_DIR}/losses.py"  "${AGENT_DIR}/losses.py.bak"

# Copy CrossE variants (uses losses_2q for Twin-Q losses)
cp "${SCRIPT_DIR}/crl_variants/crl_Markov_crossE.py"  "${AGENT_DIR}/crl.py"
cp "${SCRIPT_DIR}/crl_variants/losses_2q.py"           "${AGENT_DIR}/losses.py"

# Run
cd "${JAXGCRL_ROOT}"
python run.py crl --env="${ENV}" --cross_episode_ratio="${RATIO}" "$@"

# Restore originals
mv "${AGENT_DIR}/crl.py.bak"     "${AGENT_DIR}/crl.py"
mv "${AGENT_DIR}/losses.py.bak"  "${AGENT_DIR}/losses.py"

echo "=== Done. Originals restored. ==="
