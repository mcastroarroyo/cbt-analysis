# CBT Analysis

Utilities for analyzing TinyLlama + LoRA CBT training runs:
- Head entropy / concentration metrics
- Head ablations (Δloss)
- Heatmap exports
- Enterprise-oriented prompt probes

## Layout
- scripts/analyze_heads.py — main entry point
- data/ — place inputs or adapter weights if needed
- outputs/ — reports & heatmaps
