"""Compare a run of run.py with the earlier runs in the history, report it in markdown and append it to the history.

Exits with 1 when a case failed or got worse, so that the scheduled workflow fails and notifies.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

# a case is worse when it takes this much more time or memory than the median of the recent runs
LIMIT = 1.25
RECENT = 5
# differences of time below this many seconds are noise
MIN_SECONDS = 3.0


def change(value, reference):
    return f"{100 * (value / reference - 1):+.0f}%" if reference else ""


def compare_case(name, case, earlier):
    """A table row of the case and the reasons it is worse, if any."""
    problems = []
    if case["status"] != "ok":
        problems.append(case["status"])
    total = sum(case["iterations"].values())
    row = [name, case["status"], f"{case['time']:.1f}", "", str(case["memory_mb"]), "", str(total) if total else "", ""]
    if earlier and case["status"] == "ok":
        time = statistics.median(e["time"] for e in earlier)
        memory = statistics.median(e["memory_mb"] for e in earlier)
        last = sum(earlier[-1]["iterations"].values())
        row[3], row[5], row[7] = change(case["time"], time), change(case["memory_mb"], memory), change(total, last)
        if case["time"] > LIMIT * time and case["time"] - time > MIN_SECONDS:
            problems.append(f"time {row[3]}")
        if case["memory_mb"] > LIMIT * memory:
            problems.append(f"memory {row[5]}")
        # the iterations and the result of a timing-dependent case vary from run to run
        if case.get("deterministic", True):
            if total > last:
                problems.append(f"iterations {row[7]}")
            if case["result"] != earlier[-1]["result"]:
                problems.append(f"result changed from `{earlier[-1]['result']}`")
    return row, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="markdown file to append the report to")
    args = parser.parse_args()

    run = json.loads(args.results.read_text())
    history = [json.loads(line) for line in args.history.read_text().splitlines()] if args.history.exists() else []
    recent = history[-RECENT:]

    rows, worse = [], []
    for name, case in sorted(run["cases"].items()):
        earlier = [r["cases"][name] for r in recent if r["cases"].get(name, {}).get("status") == "ok"]
        row, problems = compare_case(name, case, earlier)
        rows.append(row)
        if problems:
            worse.append(f"- `{name}`: {', '.join(problems)}")

    lines = [
        f"### Performance on {run['storm']}",
        f"paynt {run['commit']}, {run['date']}, {run['cpu']}. Changes are against the median of the last {RECENT} runs, iterations against the last run.",
        "",
        "| case | status | time [s] | change | memory [MB] | change | iterations | change |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
    ]
    if not history:
        lines.append("First run, nothing to compare with.")
    elif worse:
        lines += [f"**{len(worse)} cases failed or got worse:**", *worse]
    else:
        lines.append("No case failed or got worse.")
    for name, case in sorted(run["cases"].items()):
        if case["tail"]:
            lines += ["", f"<details><summary>end of the output of {name}</summary>", "", "```", *case["tail"], "```", "</details>"]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.summary:
        with args.summary.open("a") as summary:
            summary.write(report)

    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a") as file:
        file.write(json.dumps(run) + "\n")
    sys.exit(1 if worse else 0)


if __name__ == "__main__":
    main()
