#!/usr/bin/env python3
"""Collect GitHub traffic locally and publish it as a stats .md with charts.

GitHub keeps traffic for 14 days only. Run this at least weekly from a clone
of the repo. It uses your existing `gh` login (needs push access to the repo).

The master archive lives outside the repo in ~/.github-traffic/<owner>/<repo>/,
so data survives even if a stats PR sits unmerged. Each run renders CSVs and a
README with Mermaid charts (rendered natively by GitHub) into docs/traffic/.

Usage (from the repo root):
  python scripts/traffic_stats.py            # update docs/traffic/ in place
  python scripts/traffic_stats.py --pr       # commit to a branch and open a PR
  python scripts/traffic_stats.py --repo IBM/AssetOpsBench --weeks 26

Weekly cron (Mon 09:00), via `crontab -e`:
  0 9 * * 1 cd /path/to/AssetOpsBench && /usr/bin/python3 scripts/traffic_stats.py --pr >> ~/traffic.log 2>&1

Requires Python 3.9+, git, and the GitHub CLI (`gh auth login`). No pip packages.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

START, END = "<!-- traffic:start -->", "<!-- traffic:end -->"
DAILY = ("views.csv", "clones.csv")
SNAPS = (("referrers.csv", "referrer"), ("paths.csv", "path"))


# ---------- shell ----------

def run(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stderr.strip()}")
    return r.stdout.strip()


def gh_api(repo, path):
    return json.loads(run("gh", "api", f"repos/{repo}/traffic/{path}"))


# ---------- csv ----------

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------- archive ----------

def update_store(store, repo_dir, repo, today):
    """Merge repo copy (seed) + fresh API data into the local store."""
    os.makedirs(store, exist_ok=True)
    api = {
        "views.csv": gh_api(repo, "views?per=day")["views"],
        "clones.csv": gh_api(repo, "clones?per=day")["clones"],
        "referrers.csv": gh_api(repo, "popular/referrers"),
        "paths.csv": gh_api(repo, "popular/paths"),
    }
    for name in DAILY:
        rows = {r["date"]: r for r in read_csv(f"{repo_dir}/{name}")}
        rows.update({r["date"]: r for r in read_csv(f"{store}/{name}")})
        for it in api[name]:
            d = it["timestamp"][:10]
            rows[d] = {"date": d, "count": it["count"], "uniques": it["uniques"]}
        write_csv(f"{store}/{name}", [rows[k] for k in sorted(rows)], ["date", "count", "uniques"])
    for name, key in SNAPS:
        seen, rows = set(), []
        for r in read_csv(f"{repo_dir}/{name}") + read_csv(f"{store}/{name}"):
            k = (r["snapshot_date"], r[key])
            if r["snapshot_date"] != today and k not in seen:
                seen.add(k)
                rows.append(r)
        rows += [{"snapshot_date": today, key: i[key], "count": i["count"],
                  "uniques": i["uniques"]} for i in api[name]]
        rows.sort(key=lambda r: r["snapshot_date"])
        write_csv(f"{store}/{name}", rows, ["snapshot_date", key, "count", "uniques"])
    return api["referrers.csv"], api["paths.csv"]


# ---------- markdown ----------

def weekly(rows, weeks):
    agg = {}
    for r in rows:
        d = dt.date.fromisoformat(r["date"])
        wk = d - dt.timedelta(days=d.weekday())
        c, u = agg.get(wk, (0, 0))
        agg[wk] = (c + int(r["count"]), u + int(r["uniques"]))
    return [(k.strftime("%m-%d"), *agg[k]) for k in sorted(agg)[-weeks:]]


def mermaid(title, labels, values, y_label, kind="bar"):
    return "\n".join([
        "```mermaid",
        "xychart-beta",
        f'    title "{title}"',
        "    x-axis [" + ", ".join(f'"{l}"' for l in labels) + "]",
        f'    y-axis "{y_label}"',
        f"    {kind} [" + ", ".join(map(str, values)) + "]",
        "```",
    ])


def stats_block(repo, views, clones, refs, paths, today, weeks):
    cut = (dt.date.fromisoformat(today) - dt.timedelta(days=14)).isoformat()
    tot = lambda rows, f="count": sum(int(r[f]) for r in rows)
    recent = lambda rows, f="count": sum(int(r[f]) for r in rows if r["date"] >= cut)
    wv, wc = weekly(views, weeks), weekly(clones, weeks)
    span = f"{views[0]['date']} to {views[-1]['date']}" if views else "n/a"

    out = [
        START,
        f"_Updated {today} for `{repo}`. Archive covers {span}. "
        "Weeks start Monday; the latest week may be partial._",
        "",
        "| Metric | Archived total | Last 14 days |",
        "|---|---:|---:|",
        f"| Views | {tot(views):,} | {recent(views):,} |",
        f"| Clones | {tot(clones):,} | {recent(clones):,} |",
        f"| Unique cloners (daily, summed) | {tot(clones, 'uniques'):,} | {recent(clones, 'uniques'):,} |",
        "",
    ]
    if wv:
        out += ["### Weekly views", "",
                mermaid("Views per week", [w[0] for w in wv], [w[1] for w in wv], "Views"), ""]
    if wc:
        out += ["### Weekly unique cloners", "",
                mermaid("Unique cloners per week", [w[0] for w in wc], [w[2] for w in wc],
                        "Cloners", "line"), "",
                "### Weekly clones", "",
                mermaid("Clones per week", [w[0] for w in wc], [w[1] for w in wc], "Clones"), ""]

    counts = [int(r["count"]) for r in clones]
    if len(counts) >= 7:
        med = statistics.median(counts)
        spikes = [r for r in clones if med and int(r["count"]) > 5 * med]
        if spikes:
            out += ["### Clone spikes", "",
                    "Days above 5x the median clone count; usually CI or mirror pulls.", ""]
            out += [f"- {r['date']}: {int(r['count']):,} clones from {r['uniques']} cloners"
                    for r in spikes[-10:]] + [""]
    for title, items, key, fmt in (("Top referrers", refs, "referrer", "{}"),
                                   ("Top pages", paths, "path", "`{}`")):
        if items:
            out += [f"### {title} (last 14 days)", "",
                    f"| {key.title()} | Views | Unique |", "|---|---:|---:|"]
            out += [f"| {fmt.format(i[key])} | {i['count']} | {i['uniques']} |" for i in items[:10]]
            out.append("")
    out.append(END)
    return "\n".join(out)


def render(store, out_dir, repo, refs, paths, today, weeks):
    os.makedirs(out_dir, exist_ok=True)
    for name in DAILY + tuple(n for n, _ in SNAPS):
        shutil.copyfile(f"{store}/{name}", f"{out_dir}/{name}")
    views, clones = read_csv(f"{store}/views.csv"), read_csv(f"{store}/clones.csv")
    block = stats_block(repo, views, clones, refs, paths, today, weeks)

    readme = f"{out_dir}/README.md"
    text = open(readme).read() if os.path.exists(readme) else (
        "# Repository traffic\n\n"
        "GitHub keeps traffic data for 14 days only; `scripts/traffic_stats.py` archives it "
        "into the CSVs in this folder. Daily unique counts do not add up to unique people "
        "over a period, and CI or mirrors inflate raw clones.\n\n")
    if START not in text:
        text = text.rstrip() + "\n\n" + START + "\n" + END + "\n"
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    with open(readme, "w") as f:
        f.write(text)


# ---------- PR ----------

def open_pr(repo, out_dir, today, do_render):
    # This script may itself be new or edited; carry it into the PR.
    root = run("git", "rev-parse", "--show-toplevel")
    me = os.path.relpath(os.path.realpath(__file__), os.path.realpath(root))
    me_src = open(__file__).read() if not me.startswith("..") else None
    dirty = run("git", "status", "--porcelain", "--", ".", f":(exclude){out_dir}",
                *([f":(exclude){me}"] if me_src else []))
    if dirty:
        sys.exit("uncommitted changes outside the stats folder; commit or stash them first:\n" + dirty)
    base = run("gh", "repo", "view", repo, "--json", "defaultBranchRef", "-q", ".defaultBranchRef.name")
    start = run("git", "rev-parse", "--abbrev-ref", "HEAD")
    branch = f"traffic/{today}"

    # Discard local edits in the stats folder; the store holds the real data.
    subprocess.run(["git", "checkout", "--", out_dir], capture_output=True)
    subprocess.run(["git", "clean", "-fdq", "--", out_dir], capture_output=True)
    me_dirty = bool(me_src) and bool(run("git", "status", "--porcelain", "--", me))
    if me_dirty:
        subprocess.run(["git", "checkout", "--", me], capture_output=True)
        if os.path.exists(me) and not run("git", "ls-files", "--", me):
            os.remove(me)  # untracked; restored on the PR branch and afterwards
    run("git", "fetch", "origin", base)
    run("git", "switch", "-C", branch, f"origin/{base}")
    try:
        do_render()
        run("git", "add", out_dir)
        if me_dirty:
            os.makedirs(os.path.dirname(me) or ".", exist_ok=True)
            with open(me, "w") as f:
                f.write(me_src)
            run("git", "add", me)
        if not run("git", "diff", "--cached", "--name-only"):
            print("no traffic changes to commit")
            return
        run("git", "commit", "-m", f"docs(traffic): update repo traffic stats ({today})")
        run("git", "push", "-u", "origin", branch, "--force-with-lease")
        existing = run("gh", "pr", "list", "--repo", repo, "--head", branch, "--json", "url", "-q", ".[0].url")
        print(existing or run("gh", "pr", "create", "--repo", repo, "--base", base, "--head", branch,
                              "--title", f"docs(traffic): update repo traffic stats ({today})",
                              "--body", f"Weekly traffic snapshot. See `{out_dir}/README.md`."))
    finally:
        subprocess.run(["git", "checkout", "--", out_dir], capture_output=True)
        subprocess.run(["git", "clean", "-fdq", "--", out_dir], capture_output=True)
        if me_dirty:
            subprocess.run(["git", "checkout", "--", me], capture_output=True)
            subprocess.run(["git", "clean", "-fq", "--", me], capture_output=True)
        run("git", "switch", start)
        if me_dirty:  # put your working copy of the script back
            os.makedirs(os.path.dirname(me) or ".", exist_ok=True)
            with open(me, "w") as f:
                f.write(me_src)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Archive GitHub traffic and render a stats .md")
    ap.add_argument("--repo", help="owner/name (default: this clone's GitHub repo)")
    ap.add_argument("--out", default="docs/traffic", help="folder in the repo for CSVs and README")
    ap.add_argument("--store", help="local archive (default: ~/.github-traffic/<owner>/<repo>)")
    ap.add_argument("--weeks", type=int, default=26, help="weeks shown in charts")
    ap.add_argument("--pr", action="store_true", help="commit to traffic/<date> and open a PR")
    a = ap.parse_args()

    repo = a.repo or run("gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner")
    store = os.path.expanduser(a.store or f"~/.github-traffic/{repo}")
    today = dt.date.today().isoformat()

    refs, paths = update_store(store, a.out, repo, today)
    n = len(read_csv(f"{store}/views.csv"))
    print(f"{repo}: {n} days archived in {store}")

    do_render = lambda: render(store, a.out, repo, refs, paths, today, a.weeks)
    if a.pr:
        open_pr(repo, a.out, today, do_render)
    else:
        do_render()
        print(f"wrote {a.out}/README.md")


if __name__ == "__main__":
    main()
