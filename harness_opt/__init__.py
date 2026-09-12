"""Harness optimization for AssetOpsBench.

An outer loop searches client-side adapter configurations while the model
weights, the MCP servers and the scenario data all stay fixed. The design
follows three papers and is documented in ``docs/harness_optimization_plan.md``:

* AIDE (arXiv 2502.13138) contributes the journal, the three operators and the
  search policy, with its greedy fitness moved off the batch and onto a
  held-out group, because a harness must generalize and a solution script need
  not.
* AutoSaddler (arXiv 2608.23041) contributes the mini-batch loop, the patch
  taxonomy, the phased schedule and the reflection buckets.
* StarHarness (arXiv 2608.24804) contributes the frozen environment, the
  forbidden-change classes and the group-wise holdout opened once.

Everything in this package is stdlib-only so the loop is testable without a
model, a container or a live MCP server.
"""
