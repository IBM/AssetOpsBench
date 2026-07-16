"""TSFM MCP server — time-series AI tasks on the sktime substrate.

Layered package:
  core/       store, schemas, tasks          — the data model + task contracts
  substrate/  resolver                       — sktime as the execution substrate
  stores/     model_store, feature_store,    — catalog = pointer index (models/features as data)
              results
  reasoning/  profile, param_space,          — evidence the agent reasons from
              feature_selection (FLOps)
  engine/     composition, plan,             — the recipe engine (the pipeline is composed)
              feature_runner, evolve
  eval/       forecast_eval                  - GIFT-Eval-style scoring + leaderboard
  io/         refs (file pointers), window   — data I/O
  main.py     the MCP tool surface           — config.py  the env knobs

Design: models & features are DATA (catalog cards), not tools; the agent composes recipes and
reasons every parameter; the server provides evidence. See docs/STRUCTURE.md.
"""

__all__ = ["core", "substrate", "stores", "reasoning", "engine", "eval", "io", "config"]
__version__ = "0.2.0"