#!/bin/bash
cd "$(dirname "$0")" || exit 1
python3 e8_hierarchical_debug.py 2>&1 | tee debug_run_$(date +%s).log
echo "Done. Output saved to debug_run_*.log"
