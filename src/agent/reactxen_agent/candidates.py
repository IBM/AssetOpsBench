"""Append-only provenance ledger for candidate calculations, never ground truth."""

import hashlib
import json
import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata

from .evidence_paths import EvidencePaths

_ROW = re.compile(
    r"^([A-Za-z][\w.:-]*)\s+(\d+)\s+([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)$"
)
_HEADER = re.compile(r"^[A-Za-z_][\w /!<>=().:+-]{0,159}$")


def extract_calculations(text):
    """Recognize labelled entity/count/value tables and scalar JSON answers.

    These are observations for review, not a claim of semantic plausibility.
    Other formats can be retained explicitly through retain_candidate.
    """
    stdout = re.search(r"<stdout>(.*?)</stdout>", text, re.S)
    body = stdout.group(1) if stdout else text
    label, rows = "unlabelled calculation", []
    observations = []

    def emit():
        if rows:
            observations.append(
                {"kind": "table", "interpretation": label, "results": list(rows)}
            )
            rows.clear()

    for raw in body.splitlines():
        line = raw.strip()
        match = _ROW.fullmatch(line)
        if match:
            entity, count, value = match.groups()
            rows.append(
                {
                    "entity": entity,
                    "count": int(count),
                    "value": str(Decimal(value.replace(",", ""))),
                }
            )
            continue
        emit()
        if line and _HEADER.fullmatch(line):
            label = line
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if (
                    isinstance(obj, dict)
                    and 2 <= len(obj) <= 8
                    and all(
                        isinstance(v, (str, int, float, bool, type(None)))
                        for v in obj.values()
                    )
                    and any(
                        isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in obj.values()
                    )
                    and any(isinstance(v, str) for v in obj.values())
                ):
                    observations.append(
                        {
                            "kind": "answer",
                            "interpretation": label,
                            "candidate_answer": obj,
                        }
                    )
                elif (
                    isinstance(obj, dict)
                    and len(obj) >= 2
                    and all(
                        isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in obj.values()
                    )
                ):
                    observations.append(
                        {
                            "kind": "numeric_mapping",
                            "interpretation": label,
                            "results": obj,
                        }
                    )
            except (ValueError, TypeError):
                pass
    emit()
    return observations


class CandidateComparison(BaseModel):
    candidate_id: str
    status: Literal["supported", "plausible", "ruled_out", "not_answer"]
    candidate_answer: dict | str | None = None
    interpretation: str = Field(min_length=1, max_length=1200)
    supporting_evidence: str = Field(min_length=1, max_length=1500)
    counterevidence_or_gap: str = Field(min_length=1, max_length=1500)
    evidence_paths: EvidencePaths = Field(default_factory=list)


class CandidateAssessment(BaseModel):
    candidate_id: str
    decision: Literal["selected", "rejected", "unresolved", "not_answer"]
    reason: str = Field(min_length=1, max_length=1200)
    evidence_paths: EvidencePaths = Field(default_factory=list)


class RetainCandidateParams(BaseModel):
    interpretation: str = Field(min_length=1, max_length=2000)
    candidate_answer: dict | str
    evidence_paths: EvidencePaths = Field(min_length=1)


class CandidateLedger:
    """One ledger per run; IDs derive from content, records are never deleted."""

    path = "candidate-ledger/index.json"

    def __init__(self, workspace, entries=None):
        self.workspace = workspace
        self.entries = entries if entries is not None else []
        self._by_id = {e["candidate_id"]: e for e in self.entries}
        self.phase = "setup"
        self.source_count = 0

    def retain(self, observation, provenance):
        canonical = json.dumps(
            observation, sort_keys=True, ensure_ascii=False, default=str
        )
        ident = "candidate_" + hashlib.sha256(canonical.encode()).hexdigest()[:16]
        if ident not in self._by_id:
            entry = {"candidate_id": ident, **observation, "provenance": []}
            self.entries.append(entry)
            self._by_id[ident] = entry
        if provenance not in self._by_id[ident]["provenance"]:
            self._by_id[ident]["provenance"].append(provenance)
        return ident

    async def flush(self):
        await self.workspace.write_file_bytes(
            self.path, json.dumps(self.entries, ensure_ascii=False, indent=2).encode()
        )

    async def observe(self, output, *, tool, arguments=None):
        text = output if isinstance(output, str) else json.dumps(output, default=str)
        observations = extract_calculations(text)
        if not observations:
            return []
        self.source_count += 1
        source = f"candidate-ledger/sources/observation-{self.source_count:05d}.json"
        await self.workspace.write_file_bytes(
            source,
            json.dumps(
                {
                    "phase": self.phase,
                    "tool": tool,
                    "arguments": arguments,
                    "output": output,
                },
                ensure_ascii=False,
                default=str,
            ).encode(),
        )
        ids = [
            self.retain(o, {"phase": self.phase, "source": source})
            for o in observations
        ]
        await self.flush()
        return ids

    def wrap_code_tool(self, tool):
        original = tool.executor

        async def execute(params):
            result = await original(params)
            # Code results are normally XML strings; images are preserved intact.
            if isinstance(result.content, str):
                ids = await self.observe(
                    result.content, tool=tool.name, arguments=params.model_dump()
                )
                if ids:
                    note = (
                        "\nCandidate calculations retained: "
                        + ", ".join(dict.fromkeys(ids))
                        + f". Compare EVERY ledger entry at {self.path}; observations are not automatically valid answers."
                    )
                    result = result.model_copy(
                        update={"content": result.content + note}
                    )
            return result

        return tool.model_copy(update={"executor": execute})

    def require_coverage(self, assessments):
        ids = [a.candidate_id for a in assessments]
        expected = set(self._by_id)
        missing = expected - set(ids)
        extra = set(ids) - expected
        if missing or extra or len(ids) != len(set(ids)):
            # Include missing observations so a finish-only repair need not fetch files.
            details = [
                {k: v for k, v in self._by_id[i].items() if k != "provenance"}
                for i in sorted(missing)
            ]
            raise ValueError(
                "Compare every retained candidate exactly once. "
                f"Unknown IDs: {sorted(extra)}. Missing observations: "
                + json.dumps(details, ensure_ascii=False)
                + f". Ledger: {self.path}"
            )

    def registration_tool(self):
        async def execute(params: RetainCandidateParams):
            ident = self.retain(
                {
                    "kind": "registered",
                    "interpretation": params.interpretation,
                    "candidate_answer": params.candidate_answer,
                },
                {"phase": self.phase, "evidence_paths": list(params.evidence_paths)},
            )
            await self.flush()
            return ToolResult(
                content=json.dumps({"candidate_id": ident, "ledger": self.path}),
                metadata=ToolUseCountMetadata(),
            )

        return Tool(
            name="retain_candidate",
            parameters=RetainCandidateParams,
            executor=execute,
            description="Retain EVERY newly discovered plausible candidate answer with its calculation/interpretation and evidence paths, even if you do not select it. Never delete earlier candidates. Use this for results not automatically retained from code tables/JSON.",
        )
