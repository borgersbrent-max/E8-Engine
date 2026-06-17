#!/bin/bash
# Run the clean hierarchical experiment and capture all output

cd "$(dirname "$0")" || exit 1

echo "Starting E8 Hierarchical Clean Experiment..."
echo "Timestamp: $(date)"
echo ""

python e8_hierarchical_clean.py 2>&1 | tee e8_hierarchical_run_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Experiment complete. Log files generated."
ls -lh e8_hierarchical_*.log e8_ablation_*.log 2>/dev/null || echo "No log files found yet."
