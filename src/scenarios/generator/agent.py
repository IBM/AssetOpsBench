"""Scenario generation orchestration and validation loop."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agent.cli import _DEFAULT_MODEL
from agent.plan_execute.executor import Executor
from llm.litellm import LiteLLMBackend

from ..constraints import (
    FOCUS_ORDER,
    ScenarioValidationFailure,
    failure_payload,
    format_accepted_scenarios_for_prompt,
    format_categories_for_prompt,
    format_forbidden_patterns_for_prompt,
    format_hardness_guidance_for_prompt,
    format_mode_requirements,
    format_requirements_for_prompt,
    validate_negative_scenario_batch,
    validate_scenario_batch,
)
from ..grounding import discover_grounding
from ..models import (
    AssetProfile,
    EvidenceCandidate,
    GroundingBundle,
    KeyDescription,
    RetrieverMode,
    Scenario,
    ScenarioGenerationResult,
    ScenarioBudget,
    SensorNameDescription,
)
from ..prompts import (
    BUDGET_ALLOCATOR_PROMPT,
    MULTIAGENT_COMBINER_PROMPT,
    NEGATIVE_SCENARIO_GENERATOR_PROMPT,
    NEGATIVE_VALIDATE_REPAIR_PROMPT,
    PROFILE_BUILDER_PROMPT,
    SCENARIO_GENERATOR_PROMPT,
    VALIDATE_REPAIR_PROMPT,
)
from ..retrieval import retrieve_asset_evidence, synthesize_research_digest
from ..text import slugify_asset_name, truncate_title_one_line
from ..utils import fetch_hf_fewshot, parse_llm_json
from .prompt_helpers import (
    _BUDGET_MAX_TOKENS,
    _MULTIAGENT_MAX_TOKENS,
    _MAX_FEWSHOT_EXAMPLES,
    _MAX_SCENARIO_ATTEMPTS,
    _PROFILE_MAX_TOKENS,
    _SCENARIO_LLM_MAX_TOKENS,
    _asset_profile_json,
    _few_shot_examples_section,
    _grounding_summary_for_prompt,
    hard_scenario_target,
    _invert_failure_mapping,
    _label_desc_dict_from_list,
    _multiagent_budget_cap,
    _normalize_failure_sensor_mapping,
    _normalize_string_list,
    _ordered_descriptions,
    _print_live_step,
    _print_section,
    _print_step,
    _redact_logged_prompt,
    _require_nonempty_list,
    _require_nonempty_str,
    _quiet_litellm_logging,
    _tool_summary_for_prompt,
    _validation_tool_names_by_focus,
)

_log = logging.getLogger(__name__)


def _scenario_from_llm_row(
    row: dict,
    *,
    scenario_type: str,
    generation_mode: str,
) -> Scenario:
    payload = dict(row)
    payload.pop("type", None)
    return Scenario(**payload, type=scenario_type, generation_mode=generation_mode)


class ScenarioGeneratorAgent:
    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        show_workflow: bool = False,
        log_dir: str | None = None,
        *,
        retriever: RetrieverMode = "arxiv",
        research_digest_path: str | None = None,
    ) -> None:
        _quiet_litellm_logging()
        self.llm = LiteLLMBackend(model_id=model_id)
        self.executor = Executor(llm=self.llm)
        self.show_workflow = show_workflow
        self.log_dir = log_dir
        self.retriever = retriever
        self.research_digest_path = research_digest_path
        self._log_steps_by_subdir: dict[str, int] = {}

    def _log_path(self, name: str, *, step: int) -> Path:
        if not self.log_dir:
            raise RuntimeError("log_dir must be configured before writing logs")
        relative = Path(name)
        suffix = relative.suffix or ".txt"
        stem = relative.stem if relative.suffix else relative.name
        filename = f"{step:02d}_{stem}{suffix}"
        return Path(self.log_dir) / relative.parent / filename

    def _write_log(self, name: str, content: str) -> None:
        if not self.log_dir:
            return
        relative = Path(name)
        dir_key = relative.parent.as_posix() if relative.parent != Path(".") else ""
        step = self._log_steps_by_subdir.get(dir_key, 0) + 1
        path = self._log_path(name, step=step)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            handle.write(content)
        self._log_steps_by_subdir[dir_key] = step

    def _write_json_log(self, name: str, payload: object) -> None:
        log_name = name if name.endswith(".json") else f"{name}.json"
        self._write_log(log_name, json.dumps(payload, indent=2))

    async def run(
        self,
        asset_name: str,
        num_scenarios: int = 50,
        data_in_couchdb: bool = False,
        num_negative_scenarios: int = 2,
    ) -> ScenarioGenerationResult:
        asset_name = asset_name.strip()
        if not asset_name:
            raise ValueError("Asset class name is empty after stripping whitespace.")
        _log.debug("Starting scenario generation for asset: %s", asset_name)
        server_desc = await self.executor.get_server_descriptions()

        if self.show_workflow:
            _print_section("Phase 1: Asset Profile Construction")
        asset_profile = await self.build_asset_profile(
            asset_name=asset_name,
            server_desc=server_desc,
            requested_open_form=data_in_couchdb,
        )

        if self.show_workflow:
            _print_section("Phase 2: Scenario Budget Allocation")
        budget = await self.allocate_budget(asset_profile, total=num_scenarios)

        if self.show_workflow:
            _print_section("Phase 3: Focused Generation & Validation")

        all_scenarios: list[Scenario] = []
        validation_tool_names = _validation_tool_names_by_focus(server_desc)
        for focus, count in budget.allocation.items():
            if count == 0 or focus == "multiagent":
                continue

            if self.show_workflow:
                _print_section(f"{focus.upper()} Focus")
                _print_step(
                    f"generate_{focus}",
                    f"Generating {count} {focus} scenarios in {asset_profile.generation_mode}",
                )

            valid_scenarios = self.generate_validated_scenarios(
                focus=focus,
                count=count,
                profile=asset_profile,
                server_desc=server_desc,
                accepted_scenarios=[scenario.to_dict() for scenario in all_scenarios],
                validation_tool_names=validation_tool_names,
            )

            for scenario_data in valid_scenarios:
                scenario_data["id"] = f"{slugify_asset_name(asset_name)}_scenario_{len(all_scenarios)+1:02d}"
                all_scenarios.append(
                    _scenario_from_llm_row(
                        scenario_data,
                        scenario_type=focus,
                        generation_mode=asset_profile.generation_mode,
                    )
                )

            if self.show_workflow:
                _print_step(
                    f"validate_{focus}",
                    f"Validated {len(valid_scenarios)} {focus} scenarios",
                )

        multiagent_count = budget.allocation.get("multiagent", 0)
        if multiagent_count > 0:
            if self.show_workflow:
                _print_section("Phase 4: Multi-Agent Scenario Construction")
            valid_multi = self.generate_validated_scenarios(
                focus="multiagent",
                count=multiagent_count,
                profile=asset_profile,
                server_desc=server_desc,
                accepted_scenarios=[scenario.to_dict() for scenario in all_scenarios],
                single_agents=[scenario.to_dict() for scenario in all_scenarios],
                validation_tool_names=validation_tool_names,
            )

            for scenario_data in valid_multi:
                scenario_data["id"] = f"{slugify_asset_name(asset_name)}_scenario_{len(all_scenarios)+1:02d}"
                all_scenarios.append(
                    _scenario_from_llm_row(
                        scenario_data,
                        scenario_type="multiagent",
                        generation_mode=asset_profile.generation_mode,
                    )
                )

            if self.show_workflow:
                _print_step("validate_multiagent", f"Validated {len(valid_multi)} multiagent scenarios")

        if self.show_workflow:
            _print_section("Summary")
            print(
                f"Successfully generated {len(all_scenarios)} total scenarios for asset: {asset_name} "
                f"in {asset_profile.generation_mode} mode.\n"
            )

        negative_scenarios: list[Scenario] = []
        if num_negative_scenarios > 0:
            if self.show_workflow:
                _print_section("Phase 5: Negative Scenario Construction")
            negative_scenarios = self.generate_negative_scenarios(
                count=num_negative_scenarios,
                profile=asset_profile,
                server_desc=server_desc,
                accepted_scenarios=[scenario.to_dict() for scenario in all_scenarios],
                validation_tool_names=validation_tool_names,
            )

        return ScenarioGenerationResult(
            scenarios=all_scenarios,
            negative_scenarios=negative_scenarios,
        )

    async def build_asset_profile(
        self,
        asset_name: str,
        server_desc: dict,
        requested_open_form: bool = False,
    ) -> AssetProfile:
        asset_name = asset_name.strip()
        if not asset_name:
            raise ValueError("Asset class name is empty after stripping whitespace.")
        if self.show_workflow:
            _print_step(
                "grounding_start",
                f"Starting grounded discovery for {asset_name}...",
                details="Inspecting IoT coverage, vibration coverage, and bounded FMSR grounding.",
            )
        grounding = discover_grounding(asset_name, requested_open_form=requested_open_form)
        self._write_json_log("01_grounding/discovery.json", grounding.model_dump())
        generation_mode = "open_form" if grounding.open_form_eligible else "closed_form"

        if self.show_workflow:
            details = (
                f"Requested open form: {requested_open_form}\n"
                f"Resolved generation mode: {generation_mode}\n"
                f"Grounded instances: {len(grounding.asset_instances)}\n"
            )
            _print_step("grounding", "Grounded discovery complete", details=details)

        preload = self.research_digest_path
        if preload:
            digest_path = Path(preload)
            if not digest_path.is_file():
                raise FileNotFoundError(
                    f"Research digest file not found: {digest_path}. "
                    "Omit --research-digest or provide a valid path to skip retrieval."
                )
            research_digest = digest_path.read_text(encoding="utf-8")
            if self.log_dir:
                self._write_log(
                    "02_retrieval/paper_digest/merged.txt",
                    research_digest.strip(),
                )
            if self.show_workflow:
                _print_step(
                    "research_digest",
                    "Loaded precomputed research digest (retrieval and digest LLM steps skipped)",
                    details=str(digest_path.resolve()),
                )
        else:
            if self.show_workflow:
                _print_step(
                    "researcher_queries",
                    f"Planning bounded academic literature retrieval for {asset_name}...",
                )

            on_pdf_workflow = None
            on_academic_query = None
            if self.show_workflow:
                _pdf_ok_logged: set[str] = set()

                def _on_academic_query(query: str, n_results: int) -> None:
                    _print_step(
                        "academic_query",
                        f"Query: {query!r}",
                        details=f"Results: {n_results} paper(s) from metadata search",
                    )

                def _on_pdf_workflow(
                    candidate: EvidenceCandidate, phase: str, outcome: str
                ) -> None:
                    title = truncate_title_one_line(candidate.title)
                    pid = candidate.paper_id
                    if outcome == "ok":
                        if pid in _pdf_ok_logged:
                            return
                        _pdf_ok_logged.add(pid)
                        _print_step(
                            "pdf_fetch",
                            f'Paper "{title}" — PDF text loaded',
                            details=f"paper_id={pid}\nphase={phase}",
                        )
                        return
                    label = {
                        "no_pdf_url": "No PDF URL (no open-access direct link resolved from metadata)",
                        "fetch_failed": "PDF fetch failed or URL not a usable PDF stream",
                        "empty_text": "PDF downloaded but extractable text is empty",
                    }.get(outcome, outcome)
                    _print_step(
                        "pdf_fetch",
                        f'Paper "{title}" — {label}',
                        details=f"paper_id={pid}\nphase={phase}\noutcome={outcome}",
                    )

                on_pdf_workflow = _on_pdf_workflow
                on_academic_query = _on_academic_query

            evidence_bundle = retrieve_asset_evidence(
                asset_name=asset_name,
                server_desc=server_desc,
                llm=self.llm,
                log_writer=self._write_log,
                retriever=self.retriever,
                on_pdf_workflow=on_pdf_workflow,
                on_academic_query=on_academic_query,
            )

            if self.show_workflow:
                top_titles = "\n".join(
                    f" - {candidate.title} (judge_score={candidate.judge_score}/10)"
                    for candidate in evidence_bundle.candidates[:3]
                ) or " - No ranked candidates"
                details = (
                    f"Canonical asset: {evidence_bundle.canonical_asset_name}\n"
                    f"Queries:\n"
                    + "\n".join(f" - {query}" for query in evidence_bundle.query_history)
                    + "\nTop Evidence:\n"
                    + top_titles
                )
                _print_step(
                    "evidence_retrieval",
                    "Academic search engine evidence retrieval complete",
                    details=details,
                )

            research_digest = synthesize_research_digest(
                evidence_bundle,
                self.llm,
                retriever=self.retriever,
                log_writer=self._write_log,
                on_pdf_workflow=on_pdf_workflow,
            )
            if self.show_workflow:
                _print_step(
                    "research_digest",
                    "Research digest synthesis complete (per-paper extraction + merge)",
                )

        prompt = PROFILE_BUILDER_PROMPT.format(
            asset_name=asset_name,
            generation_mode=generation_mode,
            grounding_summary_json=_grounding_summary_for_prompt(grounding),
            research_digest=research_digest,
            tool_descriptions=_tool_summary_for_prompt(server_desc),
        )
        self._write_log("03_asset_profile/prompt.txt", prompt)

        response = self.llm.generate_with_usage(
            prompt, max_tokens=_PROFILE_MAX_TOKENS
        )
        parsed, parse_err = parse_llm_json(response.text)
        if not parsed or not isinstance(parsed, dict):
            raise RuntimeError(
                f"Failed to parse asset profile JSON from LLM: {parse_err or 'response is not a JSON object'}"
            )
        self._write_json_log("03_asset_profile/response.json", parsed)

        profile = self._merge_profile(parsed, grounding, generation_mode)
        self._write_json_log("03_asset_profile/final_asset_profile.json", profile.model_dump())
        if self.show_workflow:
            details: str | None = None
            if self.log_dir:
                details = "Full profile JSON is saved under 03_asset_profile/ in the log directory."
            _print_step(
                "build_profile",
                f"Successfully generated Asset Profile for '{asset_name}'.",
                details=details,
            )
        return profile

    def _merge_profile(
        self,
        parsed_profile: dict,
        grounding: GroundingBundle,
        generation_mode: str,
    ) -> AssetProfile:
        merged = dict(parsed_profile)
        merged["asset_name"] = grounding.asset_name
        merged["generation_mode"] = generation_mode
        merged["asset_instances"] = [instance.model_dump() for instance in grounding.asset_instances]

        merged_failure_mapping = _normalize_failure_sensor_mapping(merged.get("failure_sensor_mapping", {}))
        if grounding.failure_sensor_mapping:
            merged["failure_sensor_mapping"] = grounding.failure_sensor_mapping
            merged_failure_mapping = grounding.failure_sensor_mapping
        elif merged_failure_mapping:
            merged["failure_sensor_mapping"] = merged_failure_mapping
        else:
            merged["failure_sensor_mapping"] = {}

        grounded_failure_modes = _normalize_string_list(grounding.failure_modes)
        from_list_fm = _label_desc_dict_from_list(
            merged.get("failure_modes"),
            primary="key",
            alternate="name",
        )
        all_failure_mode_keys = list(
            dict.fromkeys(
                [
                    *grounded_failure_modes,
                    *merged_failure_mapping.keys(),
                    *from_list_fm.keys(),
                ]
            )
        )

        if grounding.sensor_failure_mapping:
            merged["sensor_failure_mapping"] = grounding.sensor_failure_mapping
        elif merged_failure_mapping:
            merged["sensor_failure_mapping"] = _invert_failure_mapping(merged_failure_mapping)
        else:
            merged["sensor_failure_mapping"] = {}

        merged["description"] = _require_nonempty_str(merged.get("description"), field="description")

        req_iot = sorted(dict.fromkeys(str(s).strip() for s in grounding.iot_sensors if str(s).strip()))
        req_vib = sorted(dict.fromkeys(str(s).strip() for s in grounding.vibration_sensors if str(s).strip()))

        iot_from_llm = _label_desc_dict_from_list(merged.get("iot_sensors"), primary="name", alternate="key")
        vib_from_llm = _label_desc_dict_from_list(merged.get("vibration_sensors"), primary="name", alternate="key")

        merged["iot_sensors"] = _ordered_descriptions(
            iot_from_llm,
            req_iot,
            field="iot_sensors",
            build=lambda n, d: SensorNameDescription(name=n, description=d),
            sort_when_ordered_empty=True,
        )
        merged["vibration_sensors"] = _ordered_descriptions(
            vib_from_llm,
            req_vib,
            field="vibration_sensors",
            build=lambda n, d: SensorNameDescription(name=n, description=d),
            sort_when_ordered_empty=True,
        )

        if not all_failure_mode_keys:
            merged["failure_modes"] = [
                KeyDescription(key=k, description=from_list_fm[k])
                for k in sorted(from_list_fm.keys())
            ]
        else:
            merged["failure_modes"] = _ordered_descriptions(
                from_list_fm,
                all_failure_mode_keys,
                field="failure_modes",
                build=lambda k, d: KeyDescription(key=k, description=d),
            )

        if not isinstance(merged.get("relevant_tools"), dict):
            raise ValueError("Asset profile field 'relevant_tools' must be an object keyed by focus")
        tool_focuses = tuple(f for f in FOCUS_ORDER if f != "multiagent")
        normalized_relevant: dict[str, list] = {}
        for focus in tool_focuses:
            tools = merged["relevant_tools"].get(focus)
            if tools is None:
                normalized_relevant[focus] = []
                continue
            if not isinstance(tools, list):
                raise ValueError(
                    f"Asset profile field 'relevant_tools.{focus}' must be a list "
                    f"(use [] when this subagent is not applicable)"
                )
            normalized_relevant[focus] = tools
        merged["relevant_tools"] = normalized_relevant
        merged["operator_tasks"] = _require_nonempty_list(merged.get("operator_tasks"), field="operator_tasks")
        merged["manager_tasks"] = _require_nonempty_list(merged.get("manager_tasks"), field="manager_tasks")
        return AssetProfile(**merged)

    async def allocate_budget(self, profile: AssetProfile, total: int = 50) -> ScenarioBudget:
        prompt = BUDGET_ALLOCATOR_PROMPT.format(
            total_scenarios=total,
            asset_profile_json=profile.model_dump_json(indent=2),
        )
        self._write_log("04_budget/prompt.txt", prompt)

        response = self.llm.generate_with_usage(
            prompt, max_tokens=_BUDGET_MAX_TOKENS
        )
        parsed, _ = parse_llm_json(response.text)
        if not parsed or not isinstance(parsed, dict):
            parsed = {}
        self._write_json_log("04_budget/response.json", parsed)

        allocation = self._normalize_allocation(parsed.get("allocation", {}), total)
        budget = ScenarioBudget(
            total_scenarios=total,
            allocation=allocation,
            reasoning=str(parsed.get("reasoning", "")).strip(),
        )

        if self.show_workflow:
            details = f"Reasoning: {budget.reasoning or '(not provided)'}\n\nAllocation:\n"
            details += "\n".join(f" - {focus}: {count}" for focus, count in budget.allocation.items())
            _print_step("allocate_budget", f"Successfully allocated {total} scenarios across focuses.", details=details)

        return budget

    def _normalize_allocation(self, raw_allocation: dict, total: int) -> dict[str, int]:
        allocation = {
            focus: max(0, int(raw_allocation.get(focus, 0) or 0))
            for focus in FOCUS_ORDER
        }
        if not any(allocation.values()):
            raise RuntimeError("Budget allocator returned an empty allocation; refusing to default.")

        allocation["multiagent"] = min(
            allocation["multiagent"], _multiagent_budget_cap(total)
        )
        current_total = sum(allocation.values())
        if current_total == 0:
            raise RuntimeError("Budget allocator produced a zero-sum allocation; refusing to default.")

        if current_total > total:
            overflow = current_total - total
            for focus in ("multiagent", "wo", "vibration", "tsfm", "fmsr", "iot"):
                if overflow <= 0:
                    break
                reducible = min(allocation[focus], overflow)
                allocation[focus] -= reducible
                overflow -= reducible
        elif current_total < total:
            deficit = total - current_total
            allowed = list(FOCUS_ORDER)
            while deficit > 0 and allowed:
                for focus in allowed:
                    if deficit <= 0:
                        break
                    if focus == "multiagent" and allocation["multiagent"] >= _multiagent_budget_cap(
                        total
                    ):
                        continue
                    allocation[focus] += 1
                    deficit -= 1
        return allocation

    def generate_single_focus_scenarios(
        self,
        focus: str,
        count: int,
        profile: AssetProfile,
        server_desc: dict,
        accepted_scenarios: list[dict] | None = None,
    ) -> list[dict]:
        few_shots = fetch_hf_fewshot(
            focus=focus,
            max_examples=_MAX_FEWSHOT_EXAMPLES,
        )
        profile_json = _asset_profile_json(profile)
        prompt = SCENARIO_GENERATOR_PROMPT.format(
            count=count,
            subagent_name=focus,
            asset_name=profile.asset_name,
            generation_mode=profile.generation_mode,
            asset_profile_json=profile_json,
            tool_definitions=json.dumps(server_desc.get(focus, {}), indent=2),
            few_shot_examples_section=_few_shot_examples_section(few_shots),
            category_options=format_categories_for_prompt(focus),
            specialization_requirements=format_requirements_for_prompt(focus),
            hard_target_count=hard_scenario_target(count),
            hardness_guidance=format_hardness_guidance_for_prompt(focus),
            forbidden_patterns=format_forbidden_patterns_for_prompt(focus),
            mode_requirements=format_mode_requirements(profile, focus, profile.generation_mode),
            accepted_scenario_texts=format_accepted_scenarios_for_prompt(accepted_scenarios or []),
        )
        self._write_log(
            f"05_generation/{focus}/generation_prompt.txt",
            _redact_logged_prompt(prompt, profile_json),
        )

        response = self.llm.generate_with_usage(
            prompt, max_tokens=_SCENARIO_LLM_MAX_TOKENS
        )
        parsed, _ = parse_llm_json(response.text)
        if isinstance(parsed, list):
            self._write_json_log(f"05_generation/{focus}/generation_response.json", parsed)
            return parsed
        self._write_json_log(f"05_generation/{focus}/generation_response.json", [])
        return []

    def validate_and_repair(
        self,
        focus: str,
        scenarios: list[dict],
        profile: AssetProfile,
        accepted_scenarios: list[dict] | None = None,
        failures: list[ScenarioValidationFailure] | None = None,
        *,
        negative: bool = False,
    ) -> list[dict]:
        if not scenarios:
            return []

        profile_json = _asset_profile_json(profile)
        prompt_template = NEGATIVE_VALIDATE_REPAIR_PROMPT if negative else VALIDATE_REPAIR_PROMPT
        prompt = prompt_template.format(
            subagent_name=focus,
            category_options=format_categories_for_prompt(focus),
            specialization_requirements=format_requirements_for_prompt(focus),
            hard_target_count=hard_scenario_target(len(scenarios)),
            batch_size=len(scenarios),
            hardness_guidance=format_hardness_guidance_for_prompt(focus),
            forbidden_patterns=format_forbidden_patterns_for_prompt(focus),
            mode_requirements=format_mode_requirements(profile, focus, profile.generation_mode),
            asset_profile_json=profile_json,
            input_scenarios_json=json.dumps(scenarios, indent=2),
            accepted_scenario_texts=format_accepted_scenarios_for_prompt(accepted_scenarios or []),
            validation_failures_json=failure_payload(failures or []),
        )
        self._write_log(
            f"{'06_negative_generation' if negative else '05_generation'}/{focus}/validate_repair_prompt.txt",
            _redact_logged_prompt(prompt, profile_json),
        )

        response = self.llm.generate_with_usage(
            prompt, max_tokens=_SCENARIO_LLM_MAX_TOKENS
        )
        parsed, _ = parse_llm_json(response.text)
        if isinstance(parsed, list):
            repaired = parsed[: len(scenarios)]
            self._write_json_log(
                f"{'06_negative_generation' if negative else '05_generation'}/{focus}/validate_repair_response.json",
                repaired,
            )
            return repaired
        self._write_json_log(
            f"{'06_negative_generation' if negative else '05_generation'}/{focus}/validate_repair_response.json",
            scenarios,
        )
        return scenarios

    def construct_multiagent_scenarios(
        self,
        count: int,
        single_agents: list[dict],
        profile: AssetProfile,
        server_desc: dict,
        accepted_scenarios: list[dict] | None = None,
    ) -> list[dict]:
        profile_json = _asset_profile_json(profile)
        prompt = MULTIAGENT_COMBINER_PROMPT.format(
            count=count,
            asset_name=profile.asset_name,
            generation_mode=profile.generation_mode,
            asset_profile_json=profile_json,
            mcp_function_definitions=json.dumps(server_desc, indent=2),
            single_agent_scenarios_json=json.dumps(single_agents[:10], indent=2),
            accepted_scenario_texts=format_accepted_scenarios_for_prompt(accepted_scenarios or []),
            hard_target_count=hard_scenario_target(count),
            hardness_guidance=format_hardness_guidance_for_prompt("multiagent"),
            mode_requirements=format_mode_requirements(profile, "multiagent", profile.generation_mode),
            forbidden_patterns=format_forbidden_patterns_for_prompt("multiagent"),
        )
        self._write_log(
            "05_generation/multiagent/generation_prompt.txt",
            _redact_logged_prompt(prompt, profile_json),
        )

        response = self.llm.generate_with_usage(
            prompt, max_tokens=_MULTIAGENT_MAX_TOKENS
        )
        parsed, _ = parse_llm_json(response.text)
        if isinstance(parsed, list):
            self._write_json_log("05_generation/multiagent/generation_response.json", parsed)
            return parsed
        self._write_json_log("05_generation/multiagent/generation_response.json", [])
        return []

    def generate_validated_scenarios(
        self,
        focus: str,
        count: int,
        profile: AssetProfile,
        server_desc: dict,
        accepted_scenarios: list[dict] | None = None,
        single_agents: list[dict] | None = None,
        validation_tool_names: dict[str, tuple[str, ...]] | None = None,
    ) -> list[dict]:
        accepted_scenarios = list(accepted_scenarios or [])
        valid_batch: list[dict] = []
        failure_notes: list[str] = []

        for attempt in range(1, _MAX_SCENARIO_ATTEMPTS + 1):
            remaining = count - len(valid_batch)
            if remaining <= 0:
                break

            baseline = accepted_scenarios + valid_batch
            generated = self._generate_attempt_batch(
                focus=focus,
                count=remaining,
                profile=profile,
                server_desc=server_desc,
                accepted_scenarios=baseline,
                single_agents=single_agents,
            )
            if not generated:
                failure_notes.append(f"attempt {attempt}: generator returned no parseable scenarios")
                continue

            generated = self.validate_and_repair(
                focus=focus,
                scenarios=generated,
                profile=profile,
                accepted_scenarios=baseline,
            )
            valid_now, failures = validate_scenario_batch(
                focus=focus,
                scenarios=generated,
                accepted_scenarios=baseline,
                profile=profile,
                generation_mode=profile.generation_mode,
                tool_names_by_focus=validation_tool_names,
            )
            valid_batch.extend(valid_now)

            if failures:
                self._write_log(
                    f"05_generation/{focus}/deterministic_failures_attempt_{attempt:02d}.json",
                    failure_payload(failures),
                )
                if self.show_workflow:
                    _print_live_step(
                        f"repair_{focus}",
                        f"Repairing {len(failures)} scenario(s)",
                    )
                repaired_invalids = self.validate_and_repair(
                    focus=focus,
                    scenarios=[failure.scenario for failure in failures],
                    profile=profile,
                    accepted_scenarios=accepted_scenarios + valid_batch,
                    failures=failures,
                )
                repaired_valid, remaining_failures = validate_scenario_batch(
                    focus=focus,
                    scenarios=repaired_invalids,
                    accepted_scenarios=accepted_scenarios + valid_batch,
                    profile=profile,
                    generation_mode=profile.generation_mode,
                    tool_names_by_focus=validation_tool_names,
                )
                valid_batch.extend(repaired_valid)
                if remaining_failures:
                    failure_notes.append(
                        f"attempt {attempt}: {len(remaining_failures)} scenario(s) still invalid after repair"
                    )
                    self._write_log(
                        f"05_generation/{focus}/remaining_failures_attempt_{attempt:02d}.json",
                        failure_payload(remaining_failures),
                    )

        if len(valid_batch) < count:
            shortage = count - len(valid_batch)
            summary = "; ".join(failure_notes) if failure_notes else "no detailed failure summary available"
            raise RuntimeError(
                f"Failed to generate {count} valid {focus} scenarios after {_MAX_SCENARIO_ATTEMPTS} attempts. "
                f"Still missing {shortage}. Details: {summary}"
            )
        final_for_focus = valid_batch[:count]
        self._write_json_log(f"05_generation/{focus}/final_scenarios.json", final_for_focus)
        return final_for_focus

    def generate_negative_focus_scenarios(
        self,
        focus: str,
        count: int,
        profile: AssetProfile,
        server_desc: dict,
        accepted_scenarios: list[dict] | None = None,
    ) -> list[dict]:
        profile_json = _asset_profile_json(profile)
        prompt = NEGATIVE_SCENARIO_GENERATOR_PROMPT.format(
            count=count,
            subagent_name=focus,
            asset_name=profile.asset_name,
            generation_mode=profile.generation_mode,
            asset_profile_json=profile_json,
            tool_definitions=json.dumps(server_desc.get(focus, {}), indent=2),
            category_options=format_categories_for_prompt(focus),
            specialization_requirements=format_requirements_for_prompt(focus),
            mode_requirements=format_mode_requirements(profile, focus, profile.generation_mode),
            accepted_scenario_texts=format_accepted_scenarios_for_prompt(accepted_scenarios or []),
        )
        self._write_log(
            f"06_negative_generation/{focus}/generation_prompt.txt",
            _redact_logged_prompt(prompt, profile_json),
        )

        response = self.llm.generate_with_usage(
            prompt, max_tokens=_SCENARIO_LLM_MAX_TOKENS
        )
        parsed, _ = parse_llm_json(response.text)
        if isinstance(parsed, list):
            self._write_json_log(f"06_negative_generation/{focus}/generation_response.json", parsed)
            return parsed
        self._write_json_log(f"06_negative_generation/{focus}/generation_response.json", [])
        return []

    def generate_validated_negative_scenarios(
        self,
        focus: str,
        count: int,
        profile: AssetProfile,
        server_desc: dict,
        accepted_scenarios: list[dict] | None = None,
        validation_tool_names: dict[str, tuple[str, ...]] | None = None,
    ) -> list[dict]:
        accepted_scenarios = list(accepted_scenarios or [])
        valid_batch: list[dict] = []
        failure_notes: list[str] = []

        for attempt in range(1, _MAX_SCENARIO_ATTEMPTS + 1):
            remaining = count - len(valid_batch)
            if remaining <= 0:
                break

            baseline = accepted_scenarios + valid_batch
            generated = self.generate_negative_focus_scenarios(
                focus=focus,
                count=remaining,
                profile=profile,
                server_desc=server_desc,
                accepted_scenarios=baseline,
            )
            if not generated:
                failure_notes.append(f"attempt {attempt}: negative generator returned no parseable scenarios")
                continue

            generated = self.validate_and_repair(
                focus=focus,
                scenarios=generated,
                profile=profile,
                accepted_scenarios=baseline,
                negative=True,
            )
            valid_now, failures = validate_negative_scenario_batch(
                focus=focus,
                scenarios=generated,
                accepted_scenarios=baseline,
                profile=profile,
                generation_mode=profile.generation_mode,
                tool_names_by_focus=validation_tool_names,
            )
            valid_batch.extend(valid_now)

            if failures:
                self._write_log(
                    f"06_negative_generation/{focus}/deterministic_failures_attempt_{attempt:02d}.json",
                    failure_payload(failures),
                )
                repaired_invalids = self.validate_and_repair(
                    focus=focus,
                    scenarios=[failure.scenario for failure in failures],
                    profile=profile,
                    accepted_scenarios=accepted_scenarios + valid_batch,
                    failures=failures,
                    negative=True,
                )
                repaired_valid, remaining_failures = validate_negative_scenario_batch(
                    focus=focus,
                    scenarios=repaired_invalids,
                    accepted_scenarios=accepted_scenarios + valid_batch,
                    profile=profile,
                    generation_mode=profile.generation_mode,
                    tool_names_by_focus=validation_tool_names,
                )
                valid_batch.extend(repaired_valid)
                if remaining_failures:
                    failure_notes.append(
                        f"attempt {attempt}: {len(remaining_failures)} negative scenario(s) still invalid after repair"
                    )
                    self._write_log(
                        f"06_negative_generation/{focus}/remaining_failures_attempt_{attempt:02d}.json",
                        failure_payload(remaining_failures),
                    )

        if len(valid_batch) < count:
            shortage = count - len(valid_batch)
            summary = "; ".join(failure_notes) if failure_notes else "no detailed failure summary available"
            raise RuntimeError(
                f"Failed to generate {count} negative {focus} scenarios after {_MAX_SCENARIO_ATTEMPTS} attempts. "
                f"Still missing {shortage}. Details: {summary}"
            )
        final_for_focus = valid_batch[:count]
        self._write_json_log(f"06_negative_generation/{focus}/final_scenarios.json", final_for_focus)
        return final_for_focus

    def generate_negative_scenarios(
        self,
        count: int,
        profile: AssetProfile,
        server_desc: dict,
        accepted_scenarios: list[dict] | None = None,
        validation_tool_names: dict[str, tuple[str, ...]] | None = None,
    ) -> list[Scenario]:
        accepted_scenarios = list(accepted_scenarios or [])
        if count <= 0:
            return []

        focuses = self._negative_focus_order(profile, validation_tool_names)
        negative_scenarios: list[Scenario] = []
        for index in range(count):
            focus = focuses[index % len(focuses)]
            negative_rows = self.generate_validated_negative_scenarios(
                focus=focus,
                count=1,
                profile=profile,
                server_desc=server_desc,
                accepted_scenarios=accepted_scenarios + [scenario.to_dict() for scenario in negative_scenarios],
                validation_tool_names=validation_tool_names,
            )
            for scenario_data in negative_rows:
                scenario_data["id"] = (
                    f"{slugify_asset_name(profile.asset_name)}_negative_scenario_{len(negative_scenarios)+1:02d}"
                )
                negative_scenarios.append(
                    _scenario_from_llm_row(
                        scenario_data,
                        scenario_type=focus,
                        generation_mode=profile.generation_mode,
                    )
                )

        return negative_scenarios

    def _negative_focus_order(
        self,
        profile: AssetProfile,
        validation_tool_names: dict[str, tuple[str, ...]] | None,
    ) -> list[str]:
        ordered: list[str] = []
        for focus in FOCUS_ORDER:
            if focus == "multiagent":
                continue
            if (validation_tool_names and validation_tool_names.get(focus)) or profile.relevant_tools.get(focus):
                ordered.append(focus)
        if ordered:
            return ordered
        return [focus for focus in FOCUS_ORDER if focus != "multiagent"]

    def _generate_attempt_batch(
        self,
        focus: str,
        count: int,
        profile: AssetProfile,
        server_desc: dict,
        accepted_scenarios: list[dict],
        single_agents: list[dict] | None = None,
    ) -> list[dict]:
        if focus == "multiagent":
            if single_agents is None:
                raise RuntimeError("single_agents are required when generating multiagent scenarios")
            return self.construct_multiagent_scenarios(
                count=count,
                single_agents=single_agents,
                profile=profile,
                server_desc=server_desc,
                accepted_scenarios=accepted_scenarios,
            )

        return self.generate_single_focus_scenarios(
            focus=focus,
            count=count,
            profile=profile,
            server_desc=server_desc,
            accepted_scenarios=accepted_scenarios,
        )
