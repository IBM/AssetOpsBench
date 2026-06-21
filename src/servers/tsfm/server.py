"""server.py — the TSFM MCP tool surface (current, recipe-based).

Design rule (HuggingGPT): models & features are DATA in the catalog, not tools. The agent gets
a small, stable tool set and composes the large catalog through recipes. Adding a model/feature
is a catalog card, never a new tool.

Groups: discover · evidence/learn · compose+run · write-back · results. Import is guarded so the
package stays testable without `mcp` installed; run with `python -m tsfm.server`.
The pre-recipe surface is kept at legacy/server_legacy.py for reference.
"""

from __future__ import annotations

from typing import List, Optional

from tsfm.bootstrap import fresh_store
from tsfm.config import RUNS_COLLECTION, PLANS_COLLECTION
from tsfm.core import tasks as task_spec
from tsfm.stores import model_store, feature_store, results
from tsfm.engine import composition, plan
from tsfm.eval import gifteval
from tsfm.reasoning import param_space, profile

_STORE = fresh_store()  # MemoryStore + seeds by default; TSFM_STORE=couch for CouchDB


def build_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "tsfm",
        instructions=(
            "Time-series AI over an sktime substrate. Discover components (models/features are "
            "catalog data), read evidence (profile_series), compose a recipe (transforms + single/"
            "ensemble + conformal), run it (run_recipe / run_tabular_recipe / run_plan), score it "
            "(evaluate, GIFT-Eval). You reason every parameter; the server gives evidence + grades. "
            "Zero-shot is the default; fine-tune is optional. Anomaly/forecast results hand off "
            "downstream — no alerts here."
        ),
    )
    S = _STORE

    # ------------------------------------------------------------------ discover
    @mcp.tool(title="List Tasks")
    def list_tasks():
        return {"tasks": task_spec.list_tasks()}

    @mcp.tool(title="Discover Components")
    def discover_components(task: str = "tsfm_forecasting"):
        return composition.discover_components(S, task=task)

    @mcp.tool(title="Describe Candidates")
    def describe_candidates(task_id: str, top_k: int = 5, domain: Optional[str] = None):
        return {
            "candidates": model_store.describe_candidates(
                S, task_id, top_k=top_k, domain=domain
            )
        }

    @mcp.tool(title="Find Models")
    def find_models(
        task_id: str,
        min_context_length: Optional[int] = None,
        prediction_length: Optional[int] = None,
        domain: Optional[str] = None,
        top_k: int = 5,
    ):
        return {
            "models": model_store.find_models(
                S,
                task_id,
                min_context_length=min_context_length,
                prediction_length=prediction_length,
                domain=domain,
                top_k=top_k,
            )
        }

    @mcp.tool(title="Find Features")
    def find_features(
        scenario_category: Optional[str] = None,
        target_task: Optional[str] = None,
        target_model: Optional[str] = None,
    ):
        return {
            "features": feature_store.find_features(
                S,
                category=scenario_category,
                target_task=target_task,
                target_model=target_model,
            )
        }

    @mcp.tool(title="Get Component")
    def get_component(component_id: str):
        """A model or feature card + (for models) its reasoned parameter schema."""
        card = model_store.get_model(S, component_id)
        if card:
            try:
                card = {**card, "param_schema": param_space.param_schema(card)}
            except Exception as e:
                card = {**card, "param_schema_error": str(e)[:120]}
            return card
        return feature_store.get_feature(S, component_id) or {"error": "not found"}

    # ------------------------------------------------------------- evidence/learn
    @mcp.tool(title="Profile Series")
    def profile_series(asset_id: str, channels: Optional[List[str]] = None):
        return profile.profile_series(S, asset_id, channels=channels)

    @mcp.tool(title="Select Features (FLOps)")
    def select_features(
        series: List[float],
        reference_feature: str = "mean",
        cd_margin: float = 0.05,
        scorers: Optional[List[str]] = None,
    ):
        return feature_store.select_features(
            series, reference_feature=reference_feature, cd_margin=cd_margin
        )

    # --------------------------------------------------------------- compose+run
    @mcp.tool(title="Run Recipe")
    def run_recipe(
        series: List[float],
        recipe: dict,
        asset_id: str = "asset",
        parent_run_id: Optional[str] = None,
    ):
        return composition.run_recipe(
            S, series, recipe, asset_id=asset_id, parent_run_id=parent_run_id
        )

    @mcp.tool(title="Run Tabular Recipe")
    def run_tabular_recipe(
        X: List[List[float]],
        recipe: dict,
        y: Optional[list] = None,
        asset_id: str = "asset",
    ):
        return composition.run_tabular_recipe(S, X, recipe, y=y, asset_id=asset_id)

    @mcp.tool(title="Run Plan")
    def run_plan(
        plan_spec: dict, asset_id: str = "asset", scenario_id: Optional[str] = None
    ):
        return plan.run_plan(S, plan_spec, asset_id=asset_id, scenario_id=scenario_id)

    @mcp.tool(title="Evaluate (GIFT-Eval)")
    def evaluate(recipe: dict, configs: List[dict]):
        return gifteval.evaluate_recipe(S, recipe, configs)

    @mcp.tool(title="Finetune")
    def finetune(asset_id: str, base_model_id: str):
        """Fine-tune a base model on a data_ref → returns a checkpoint pointer (registers
        nothing; the agent decides whether to register_model). Pending wiring to the sktime
        fine-tune path — see docs/FLOPS_PLAN.md / training_regime."""
        return {
            "status": "not_implemented",
            "note": "zero-shot is the default path; fine-tune wiring is the next build",
            "base_model_id": base_model_id,
            "asset_id": asset_id,
        }

    # ----------------------------------------------------------------- write-back
    @mcp.tool(title="Register Model")
    def register_model(model: dict):
        return model_store.register_model(S, model, overwrite=True)

    @mcp.tool(title="Register Feature")
    def register_feature(feature: dict):
        return feature_store.register_feature(S, feature, overwrite=True)

    # -------------------------------------------------------------------- results
    @mcp.tool(title="Get Result")
    def get_result(task_type: str, result_id: str):
        return results.get_result(S, task_type, result_id) or {"error": "not found"}

    @mcp.tool(title="List Results")
    def list_results(
        task_type: str,
        asset_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
    ):
        return {
            "results": results.list_results(
                S, task_type, asset_id=asset_id, scenario_id=scenario_id
            )
        }

    @mcp.tool(title="Get Run")
    def get_run(run_id: str):
        return S.get(RUNS_COLLECTION, run_id) or {"error": "not found"}

    @mcp.tool(title="List Runs")
    def list_runs(asset_id: Optional[str] = None):
        sel = {"asset_id": asset_id} if asset_id else {}
        runs = S.find(RUNS_COLLECTION, sel)
        plans = S.find(PLANS_COLLECTION, sel)
        return {"runs": runs, "plans": plans}

    return mcp


def main():
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
