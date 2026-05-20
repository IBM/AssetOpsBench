"""Smart Grid 7th-domain MCP servers.

This namespace hosts four MCP servers that surface a Smart Grid transformer
operations dataset to AOB agents:

- :mod:`servers.smart_grid.iot` — sensor telemetry and asset metadata
- :mod:`servers.smart_grid.fmsr` — failure-mode-to-sensor relations + DGA analysis
- :mod:`servers.smart_grid.tsfm` — time-series forecasting (RUL + anomaly detection)
- :mod:`servers.smart_grid.wo` — work-order creation and management

The shared CSV-loading helpers live in :mod:`servers.smart_grid.base`. Set
``SG_DATA_DIR`` in the environment to point at the directory holding the
processed CSVs (defaults to ``data/sg_processed/`` relative to cwd).

Originally developed in the SmartGridBench source project
(``HPML6998-S26-Team13/hpml-assetopsbench-smart-grid-mcp``) and extracted into
AOB to make Smart Grid Bench a first-class AOB domain.
"""
