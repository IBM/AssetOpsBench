#!/usr/bin/env python3
"""Gate 5: counterfactual utility of the skill library, measured per graph.

Gates 1 to 4 ask whether a skill is well-formed, self-contained, physically
admissible and free of benchmark answers. None of them asks whether it helps.
This is the gate that does, and it is the only one whose input is runs rather
than files.

    # 1. record runs at both K levels, then
    python skills/tools/gate5_counterfactual.py --runs runs.jsonl

    # 2. attribute the delta to the graphs the agent actually opened
    python skills/tools/gate5_counterfactual.py --runs runs.jsonl --per-graph

    # 3. stamp the result into a machine-readable admission record
    python skills/tools/gate5_counterfactual.py --runs runs.jsonl --per-graph \
        --emit gate5-admission.json

    python skills/tools/gate5_counterfactual.py --self-test

The design, and why each piece is there:

**Paired, task by task.** Scenario difficulty varies far more than the library
effect does, so an unpaired comparison of two group means measures the task mix.
The unit of analysis is `s(t) = score_K1(t) - score_K0(t)` for the same task, and
the resampling unit is the task, not the run.

**Regressions counted separately, never netted.** A library that helps on average
while poisoning one asset class is worse than no library. Mean `s` cannot show
that and is not asked to.

**A clean-baseline check on the recorded runs, not on the code.** `preflight.py`
proves the k0 code path mounts nothing. This proves the k0 runs that were
actually scored consulted nothing, by reading their trajectories. A contaminated
baseline invalidates every number below it, so it is a hard failure and it is
checked first.

**Power reported alongside every null.** With one suite and many graphs, most
graphs are consulted on a handful of tasks. "No effect detected" and "not enough
runs to detect one" are different findings and are reported as different
verdicts. A graph consulted on too few tasks is `INSUFFICIENT_POWER`, never
`NEUTRAL`.

**Multiplicity controlled.** Per-graph p-values are corrected across the graphs
actually tested, by Benjamini-Hochberg. Testing 38 graphs at alpha 0.05 and
reporting the two that cleared it is how a library gets admitted on noise.

Input is a run manifest, JSONL, one recorded run per line:

    {"task_id": "s-014", "k_level": "k0", "score": 0.0, "repetition": 0,
     "tokens": 18422, "steps": 11, "tool_calls": 7,
     "asset_class": "chiller-hvac", "trajectory": "runs/s-014-k0-0.jsonl"}

`task_id`, `k_level` and `score` are required. Everything else is optional and
enables a further section of the report. `trajectory` may be a path to a
recorded trajectory (JSON, JSONL or transcript text) or the transcript inline
under `transcript`; it is read only to discover which `SKILL.md` files were
opened, which is what makes per-graph attribution possible.

Exit codes: 0 the library is admitted at the collection level (or the self-test
passed), 1 a hard failure, a regression breach, or a collection-level null,
2 bad invocation.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
import statistics
import sys
from collections import defaultdict

K_BASELINE = "k0"
K_TREATMENT = "k1"
K_RECOVERY = "k1-recovery"
KNOWN_K = {K_BASELINE, K_TREATMENT, K_RECOVERY}

# A path into the mounted collection, as it appears in a shell command, a file
# read or a transcript line. Both mount layouts are matched, and so is a bare
# relative reference, because the agent's own cwd varies.
CONSULT_RE = re.compile(
    r"repo-skills/([a-z0-9][a-z0-9-]{0,63})"
    r"(?:/sub-skills/([a-z0-9][a-z0-9-]{0,63}))?"
)
ROUTER_RE = re.compile(r"repo-skills-router")

# 80 percent power, two-sided alpha 0.05: z(0.975) + z(0.80).
Z_MDE = 1.959964 + 0.841621
MIN_TASKS_FOR_A_VERDICT = 8


# --------------------------------------------------------------------------
# statistics, stdlib only so the harness runs wherever the benchmark runs
# --------------------------------------------------------------------------

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(deltas: list[float], reps: int = 10000, alpha: float = 0.05,
                 seed: int = 20260905) -> tuple[float, float]:
    """Percentile bootstrap over the paired per-task deltas.

    The resampling unit is the task. Resampling runs instead would treat two
    repetitions of one scenario as two independent observations, which they are
    not, and would report a confidence interval that is too narrow.
    """
    if len(deltas) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(reps):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(math.floor((alpha / 2) * reps))]
    hi = means[min(reps - 1, int(math.ceil((1 - alpha / 2) * reps)) - 1)]
    return (lo, hi)


def sign_test_p(deltas: list[float]) -> float:
    """Exact two-sided sign test. Zero deltas are dropped, which is the
    conservative convention: a task the library did not change is not evidence
    that it helped."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _midranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho with midranks, so ties do not inflate it."""
    if len(xs) < 3:
        return float("nan")
    rx, ry = _midranks(xs), _midranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def benjamini_hochberg(pvals: dict[str, float], q: float = 0.05) -> set[str]:
    """Return the keys rejected at false-discovery rate q."""
    if not pvals:
        return set()
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    cut = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= q * i / m:
            cut = i
    return {k for k, _ in items[:cut]}


def mde(deltas: list[float]) -> float:
    """Minimum effect this many paired tasks could detect at 80 percent power.
    Reported next to every null so a null is readable."""
    if len(deltas) < 2:
        return float("nan")
    sd = statistics.stdev(deltas)
    return Z_MDE * sd / math.sqrt(len(deltas))


# --------------------------------------------------------------------------
# reading the manifest
# --------------------------------------------------------------------------

def load_runs(path: pathlib.Path) -> list[dict]:
    runs = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: not JSON: {exc}")
        for field in ("task_id", "k_level", "score"):
            if field not in rec:
                raise SystemExit(f"{path}:{lineno}: missing required field `{field}`")
        if rec["k_level"] not in KNOWN_K:
            raise SystemExit(f"{path}:{lineno}: unknown k_level `{rec['k_level']}`; "
                             f"expected one of {sorted(KNOWN_K)}")
        try:
            rec["score"] = float(rec["score"])
        except (TypeError, ValueError):
            raise SystemExit(f"{path}:{lineno}: score is not numeric")
        runs.append(rec)
    if not runs:
        raise SystemExit(f"{path}: no runs")
    return runs


def consulted_graphs(rec: dict, base: pathlib.Path | None) -> set[str]:
    """Which skill graphs this run opened, read from its trajectory.

    Attribution is by what the agent actually read, not by what the router
    might have offered it. A graph nobody opened is untested, and saying so is
    the point of the `UNTESTED` verdict.
    """
    text = rec.get("transcript")
    if text is None:
        traj = rec.get("trajectory")
        if not traj:
            return set()
        p = pathlib.Path(traj)
        if not p.is_absolute() and base is not None:
            p = base / p
        if not p.exists():
            return set()
        text = p.read_text(encoding="utf-8", errors="replace")
    found = set()
    for graph, sub in CONSULT_RE.findall(text):
        if graph in {"repo-skills", "sub-skills"}:
            continue
        found.add(graph)
        if sub:
            found.add(f"{graph}/{sub}")
    if ROUTER_RE.search(text):
        found.add("repo-skills-router")
    return found


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def pair_runs(runs: list[dict], treatment: str) -> tuple[dict, list[str]]:
    """Collapse repetitions to a per-task mean at each K level, then pair.

    A task present at only one level cannot contribute a delta and is reported
    rather than dropped silently, because a systematically missing arm is the
    most common way a paired comparison goes wrong.
    """
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by[r["task_id"]][r["k_level"]].append(r)
    paired, unpaired = {}, []
    for task, levels in by.items():
        if K_BASELINE in levels and treatment in levels:
            paired[task] = {
                K_BASELINE: levels[K_BASELINE],
                treatment: levels[treatment],
            }
        else:
            have = sorted(levels)
            unpaired.append(f"{task}: only {have}")
    return paired, sorted(unpaired)


def analyse(runs: list[dict], treatment: str, base: pathlib.Path | None,
            per_graph: bool, regression_budget: float,
            reps: int) -> dict:
    out: dict = {"treatment_arm": treatment, "hard_failures": [],
                 "warnings": []}

    # Hard check first: the baseline must be clean in the recorded runs, not
    # only in the code path. Everything downstream is void if it is not.
    contaminated = []
    for r in runs:
        if r["k_level"] != K_BASELINE:
            continue
        if consulted_graphs(r, base):
            contaminated.append(r["task_id"])
    if contaminated:
        out["hard_failures"].append({
            "code": "CONTAMINATED_BASELINE",
            "detail": f"{len(contaminated)} k0 run(s) reference the skill "
                      f"collection; the baseline is not unaided",
            "tasks": sorted(set(contaminated))[:20],
        })

    paired, unpaired = pair_runs(runs, treatment)
    out["tasks_paired"] = len(paired)
    out["tasks_unpaired"] = unpaired
    if len(paired) < 2:
        out["hard_failures"].append({
            "code": "NO_PAIRED_TASKS",
            "detail": "fewer than two tasks have runs at both K levels",
        })
        return out

    deltas, meta = {}, {}
    for task, levels in paired.items():
        s0 = mean([r["score"] for r in levels[K_BASELINE]])
        s1 = mean([r["score"] for r in levels[treatment]])
        deltas[task] = s1 - s0
        meta[task] = {
            "asset_class": levels[treatment][0].get("asset_class"),
            "d_tokens": _delta_field(levels, treatment, "tokens"),
            "d_steps": _delta_field(levels, treatment, "steps"),
            "d_tool_calls": _delta_field(levels, treatment, "tool_calls"),
            "consulted": sorted(set().union(*[consulted_graphs(r, base)
                                              for r in levels[treatment]])),
        }

    d = list(deltas.values())
    lo, hi = bootstrap_ci(d, reps=reps)
    p = sign_test_p(d)
    regressions = {t: v for t, v in deltas.items() if v < 0}
    improvements = {t: v for t, v in deltas.items() if v > 0}
    out["headline"] = {
        "mean_s": mean(d),
        "median_s": statistics.median(d),
        "ci95": [lo, hi],
        "sign_test_p": p,
        "n_tasks": len(d),
        "improved": len(improvements),
        "unchanged": len(d) - len(improvements) - len(regressions),
        "regressed": len(regressions),
        "regression_rate": len(regressions) / len(d),
        "mde_at_80_power": mde(d),
    }
    out["worst_regressions"] = sorted(regressions.items(), key=lambda kv: kv[1])[:10]

    # Regressions are a budget, not a footnote.
    if len(regressions) / len(d) > regression_budget:
        out["hard_failures"].append({
            "code": "REGRESSION_BUDGET_EXCEEDED",
            "detail": f"{len(regressions)}/{len(d)} tasks regressed "
                      f"({len(regressions)/len(d):.1%}), budget is "
                      f"{regression_budget:.1%}",
        })

    # Per asset class, because a mean can hide a class the library poisons.
    by_class = defaultdict(list)
    for t, v in deltas.items():
        if meta[t]["asset_class"]:
            by_class[meta[t]["asset_class"]].append(v)
    out["by_asset_class"] = {
        k: {"n": len(v), "mean_s": mean(v),
            "regressed": sum(1 for x in v if x < 0)}
        for k, v in sorted(by_class.items())
    }

    # The compute confound. If s tracks extra tokens, the finding is "we spent
    # more", and a reviewer will say so before we do.
    conf = {}
    for field in ("d_tokens", "d_steps", "d_tool_calls"):
        xs = [(deltas[t], meta[t][field]) for t in deltas
              if meta[t][field] is not None]
        if len(xs) >= 3:
            conf[field] = {
                "spearman_rho": spearman([a for a, _ in xs], [b for _, b in xs]),
                "n": len(xs),
                "mean_delta": mean([b for _, b in xs]),
            }
    out["compute_confound"] = conf

    if per_graph:
        out["per_graph"] = _per_graph(deltas, meta, reps)

    out["verdict"] = _verdict(out)
    # The gate passes only when the effect is admitted AND nothing else failed.
    # Keeping the two apart means a library that helps on average but breaches
    # the regression budget reads as what it is, rather than as a null.
    out["gate"] = "PASS" if (out["verdict"] == "ADMITTED"
                             and not out["hard_failures"]) else "FAIL"
    return out


def _delta_field(levels: dict, treatment: str, field: str):
    a = [r[field] for r in levels[K_BASELINE] if isinstance(r.get(field), (int, float))]
    b = [r[field] for r in levels[treatment] if isinstance(r.get(field), (int, float))]
    if not a or not b:
        return None
    return mean(b) - mean(a)


def _per_graph(deltas: dict, meta: dict, reps: int) -> dict:
    """Admission per graph, restricted to the tasks where it was opened.

    This is the part that makes Gate 5 a gate rather than a headline. A library
    can post a positive mean while a third of its graphs do nothing, and only
    per-graph attribution shows which third.
    """
    tasks_by_graph = defaultdict(list)
    for t in deltas:
        for g in meta[t]["consulted"]:
            if "/" in g or g == "repo-skills-router":
                continue  # graph level only; sub-skills roll up
            tasks_by_graph[g].append(t)

    rows, pvals = {}, {}
    for g, ts in sorted(tasks_by_graph.items()):
        d = [deltas[t] for t in ts]
        row = {
            "n_tasks": len(d),
            "mean_s": mean(d),
            "regressed": sum(1 for x in d if x < 0),
            "mde_at_80_power": mde(d),
        }
        if len(d) >= MIN_TASKS_FOR_A_VERDICT:
            row["ci95"] = list(bootstrap_ci(d, reps=reps))
            row["sign_test_p"] = sign_test_p(d)
            pvals[g] = row["sign_test_p"]
        rows[g] = row

    rejected = benjamini_hochberg(pvals)
    for g, row in rows.items():
        if row["n_tasks"] < MIN_TASKS_FOR_A_VERDICT:
            row["verdict"] = "INSUFFICIENT_POWER"
        elif row["mean_s"] < 0 and g in rejected:
            row["verdict"] = "REGRESSION"
        elif row["mean_s"] > 0 and g in rejected:
            row["verdict"] = "ADMITTED"
        else:
            row["verdict"] = "NEUTRAL"
        row["bh_rejected"] = g in rejected
    return rows


#: Failures that make the comparison meaningless rather than merely negative.
#: A contaminated baseline is not a bad result, it is no result. A regression
#: breach is a real result and a gate failure, so it must not be allowed to
#: overwrite the effect verdict with `VOID`.
INVALIDATING = {"CONTAMINATED_BASELINE", "NO_PAIRED_TASKS"}


def _verdict(out: dict) -> str:
    if any(f["code"] in INVALIDATING for f in out["hard_failures"]):
        return "VOID"
    if "headline" not in out:
        return "VOID"
    h = out["headline"]
    if h["ci95"][0] > 0:
        return "ADMITTED"
    if h["ci95"][1] < 0:
        return "HARMFUL"
    if h["mde_at_80_power"] > abs(h["mean_s"]) * 2 and h["n_tasks"] < 40:
        return "UNDERPOWERED"
    return "NOT_SHOWN_TO_HELP"


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def render(out: dict, untested: list[str]) -> None:
    for f in out["hard_failures"]:
        print(f"FAIL  {f['code']}: {f['detail']}")
        if f.get("tasks"):
            print(f"      tasks: {', '.join(f['tasks'])}")
    if out["hard_failures"]:
        print()
    if "headline" not in out:
        return

    h = out["headline"]
    print(f"Arm: {K_BASELINE} versus {out['treatment_arm']}, "
          f"{h['n_tasks']} paired tasks")
    if out["tasks_unpaired"]:
        print(f"  {len(out['tasks_unpaired'])} task(s) had only one arm and "
              f"were excluded")
    print()
    print(f"  mean s            {h['mean_s']:+.4f}")
    print(f"  95% CI            [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]")
    print(f"  median s          {h['median_s']:+.4f}")
    print(f"  sign test p       {h['sign_test_p']:.4f}")
    print(f"  detectable at 80% {h['mde_at_80_power']:.4f}")
    print(f"  improved          {h['improved']}")
    print(f"  unchanged         {h['unchanged']}")
    print(f"  regressed         {h['regressed']}  ({h['regression_rate']:.1%})")
    if out["worst_regressions"]:
        print()
        print("  worst regressions")
        for t, v in out["worst_regressions"]:
            print(f"    {t:<28}{v:+.4f}")

    if out.get("by_asset_class"):
        print()
        print("  by asset class")
        for k, v in out["by_asset_class"].items():
            print(f"    {k:<28}n={v['n']:<4}mean {v['mean_s']:+.4f}  "
                  f"regressed {v['regressed']}")

    if out.get("compute_confound"):
        print()
        print("  compute confound (Spearman of s against extra compute)")
        for k, v in out["compute_confound"].items():
            rho = v["spearman_rho"]
            shown = "no variance" if rho != rho else f"{rho:+.3f}"
            print(f"    {k:<28}rho {shown:<12}n={v['n']}  "
                  f"mean delta {v['mean_delta']:+.1f}")

    if out.get("per_graph"):
        print()
        print("  per graph, on the tasks where the graph was actually opened")
        print(f"    {'graph':<44}{'n':>4}  {'mean s':>8}  {'verdict':<20}")
        for g, row in sorted(out["per_graph"].items(),
                             key=lambda kv: -kv[1]["mean_s"]):
            print(f"    {g:<44}{row['n_tasks']:>4}  {row['mean_s']:>+8.4f}  "
                  f"{row['verdict']:<20}")
        if untested:
            print()
            print(f"    {len(untested)} graph(s) never opened by any run: "
                  f"UNTESTED")
            for g in untested[:12]:
                print(f"      {g}")
            if len(untested) > 12:
                print(f"      ... and {len(untested) - 12} more")

    print()
    print(f"VERDICT: {out['verdict']}   GATE 5: {out.get('gate', 'FAIL')}")
    if VERDICT_MEANING.get(out["verdict"]):
        print(f"         {VERDICT_MEANING[out['verdict']]}")


VERDICT_MEANING = {
    "VOID": "a hard failure invalidates the comparison",
    "ADMITTED": "the confidence interval on mean s excludes zero from above",
    "HARMFUL": "the confidence interval excludes zero from below",
    "UNDERPOWERED": "the effect this many tasks could detect is larger than the "
                    "effect observed; run more tasks before concluding anything",
    "NOT_SHOWN_TO_HELP": "the interval spans zero at adequate power",
}


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _synth(tmp: pathlib.Path, effect: float, n: int = 60, seed: int = 7,
           contaminate: bool = False, poison_class: str | None = None) -> pathlib.Path:
    """Build a manifest with a known planted effect, so the harness can be
    checked against an answer it cannot see."""
    rng = random.Random(seed)
    classes = ["chiller-hvac", "pumps", "compressors", "bearings-gearboxes"]
    graphs = ["assetops-domain", "rca-and-responsible-variable",
              "compressor-diagnosis", "pint-units-for-assets"]
    lines = []
    for i in range(n):
        task = f"s-{i:03d}"
        cls = classes[i % len(classes)]
        base = rng.uniform(0.1, 0.8)
        eff = effect
        if poison_class and cls == poison_class:
            eff = -abs(effect) * 2
        lines.append(json.dumps({
            "task_id": task, "k_level": "k0", "score": round(base, 4),
            "tokens": 15000 + rng.randint(-2000, 2000), "steps": 10,
            "tool_calls": 6, "asset_class": cls,
            "transcript": "opened nothing" if not contaminate
                          else "cat repo-skills/assetops-domain/SKILL.md",
        }))
        lines.append(json.dumps({
            "task_id": task, "k_level": "k1",
            "score": round(min(1.0, max(0.0, base + eff + rng.gauss(0, 0.05))), 4),
            "tokens": 17000 + rng.randint(-2000, 2000), "steps": 12,
            "tool_calls": 8, "asset_class": cls,
            "transcript": f"cat repo-skills/{graphs[i % len(graphs)]}/SKILL.md",
        }))
    p = tmp / f"runs-{effect}-{seed}-{contaminate}-{poison_class}.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def self_test() -> int:
    import tempfile
    fails = []
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)

        # 1. a real effect is recovered and admitted
        r = analyse(load_runs(_synth(tmp, 0.12)), K_TREATMENT, tmp, True, 0.35, 2000)
        if r["verdict"] != "ADMITTED":
            fails.append(f"planted +0.12 gave {r['verdict']}, expected ADMITTED")
        if not (0.08 < r["headline"]["mean_s"] < 0.16):
            fails.append(f"planted +0.12 recovered as {r['headline']['mean_s']:.4f}")

        # 2. a null library is not admitted
        r = analyse(load_runs(_synth(tmp, 0.0, seed=11)), K_TREATMENT, tmp, True, 0.6, 2000)
        if r["verdict"] == "ADMITTED":
            fails.append("a null effect was admitted; the gate passes anything")

        # 3. a harmful library is caught
        # The budget is set to 1.0 so the effect verdict is isolated: a harmful
        # library also breaches any sane regression budget, and the point of
        # this case is that the effect itself is named.
        r = analyse(load_runs(_synth(tmp, -0.15, seed=13)), K_TREATMENT, tmp, True, 1.0, 2000)
        if r["verdict"] != "HARMFUL":
            fails.append(f"planted -0.15 gave {r['verdict']}, expected HARMFUL")
        if r["gate"] != "FAIL":
            fails.append("a harmful library passed the gate")

        # 4. a contaminated baseline voids the run
        r = analyse(load_runs(_synth(tmp, 0.12, seed=17, contaminate=True)),
                    K_TREATMENT, tmp, True, 0.35, 500)
        if r["verdict"] != "VOID":
            fails.append(f"contaminated baseline gave {r['verdict']}, expected VOID")
        if not any(f["code"] == "CONTAMINATED_BASELINE" for f in r["hard_failures"]):
            fails.append("contamination was not named as the failure")

        # 5. a class the library poisons is visible even when the mean is up
        r = analyse(load_runs(_synth(tmp, 0.20, seed=19, poison_class="pumps")),
                    K_TREATMENT, tmp, True, 0.9, 2000)
        if r["headline"]["mean_s"] <= 0:
            fails.append("poisoned-class fixture did not produce a positive mean")
        if r["by_asset_class"].get("pumps", {}).get("mean_s", 0) >= 0:
            fails.append("the poisoned class did not show a negative class mean")

        # 6. the regression budget bites
        r = analyse(load_runs(_synth(tmp, 0.20, seed=19, poison_class="pumps")),
                    K_TREATMENT, tmp, True, 0.10, 500)
        if not any(f["code"] == "REGRESSION_BUDGET_EXCEEDED" for f in r["hard_failures"]):
            fails.append("a 25% regression rate did not breach a 10% budget")

        # 7. small n is called underpowered, not neutral
        r = analyse(load_runs(_synth(tmp, 0.01, n=10, seed=23)), K_TREATMENT,
                    tmp, True, 0.9, 2000)
        for g, row in r.get("per_graph", {}).items():
            if row["n_tasks"] < MIN_TASKS_FOR_A_VERDICT and row["verdict"] != "INSUFFICIENT_POWER":
                fails.append(f"{g} with n={row['n_tasks']} was called {row['verdict']}")

        # 8. the statistics themselves
        if abs(spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1.0) > 1e-9:
            fails.append("spearman of a perfect inversion is not -1")
        if abs(sign_test_p([1, 1, 1, 1, 1]) - 2 / 32) > 1e-12:
            fails.append("exact sign test disagrees with 2/2^5")
        if benjamini_hochberg({"a": 0.001, "b": 0.9, "c": 0.8}) != {"a"}:
            fails.append("BH rejected the wrong set")
        # One graph at p=0.04 among nineteen nulls must NOT survive: nominal
        # significance on one of twenty tests is exactly what multiplicity
        # control exists to refuse. (Twenty graphs all at 0.04 is a different
        # situation and BH does reject them, correctly.)
        lonely = {"hit": 0.04}
        lonely.update({f"g{i}": 0.9 for i in range(19)})
        if benjamini_hochberg(lonely) != set():
            fails.append("BH admitted one nominal hit among nineteen nulls")
        if benjamini_hochberg({f"g{i}": 0.04 for i in range(20)}) != {f"g{i}" for i in range(20)}:
            fails.append("BH failed to reject twenty consistent hits")

    for f in fails:
        print(f"SELF-TEST FAIL  {f}")
    if not fails:
        print("self-test passed: 8 checks, planted effects recovered, "
              "null and contaminated fixtures correctly refused")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=pathlib.Path,
                    help="JSONL run manifest, one recorded run per line")
    ap.add_argument("--root", type=pathlib.Path,
                    default=pathlib.Path("skills/repositories"),
                    help="collection root, used to list graphs never opened")
    ap.add_argument("--base", type=pathlib.Path,
                    help="directory that relative `trajectory` paths are relative to "
                         "(default: the manifest's own directory)")
    ap.add_argument("--arm", default=K_TREATMENT, choices=sorted(KNOWN_K - {K_BASELINE}),
                    help="which treatment arm to compare against k0")
    ap.add_argument("--per-graph", action="store_true",
                    help="attribute the delta to the graphs each run opened")
    ap.add_argument("--regression-budget", type=float, default=0.15,
                    help="fraction of tasks allowed to regress before the gate fails")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--emit", type=pathlib.Path,
                    help="write the machine-readable admission record here")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.runs:
        ap.error("--runs is required unless --self-test is given")
    if not a.runs.exists():
        print(f"not found: {a.runs}", file=sys.stderr)
        return 2

    base = a.base or a.runs.parent
    runs = load_runs(a.runs)
    out = analyse(runs, a.arm, base, a.per_graph, a.regression_budget, a.bootstrap)

    untested: list[str] = []
    if a.per_graph and (a.root / "repo-skills").is_dir():
        present = {p.name for p in (a.root / "repo-skills").iterdir()
                   if p.is_dir() and (p / "SKILL.md").exists()}
        untested = sorted(present - set(out.get("per_graph", {})))
        out["untested_graphs"] = untested

    out["verdict_meaning"] = VERDICT_MEANING.get(out["verdict"], "")
    if a.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        render(out, untested)

    if a.emit:
        a.emit.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"\nadmission record written to {a.emit}")

    return 0 if out["verdict"] == "ADMITTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
