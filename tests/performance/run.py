"""Run the performance benchmarks of cases.toml and write the measurements as JSON."""

import argparse
import datetime
import fnmatch
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ROOT / "models" / "performance-test"

LOG_PREFIX = r"^(?:\S+ \S+ - \S+ - )?"
RESULT = re.compile(
    LOG_PREFIX + r"(optimum: .*|feasible: .*|satisfied \d+/\d+ members|no admissible tree found|built: .*"
    r"|the synthesized tree has value .*|final tree has value .*|robust assignment, worst case \S+)",
    re.MULTILINE,
)
ITERATIONS = re.compile(r"^(Game|MDP|DTMC) stats: .*iterations: (\d+)$", re.MULTILINE)
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def same_number(a, b):
    if a.lstrip("-").isdigit() and b.lstrip("-").isdigit():
        return a == b
    return math.isclose(float(a), float(b), rel_tol=1e-3, abs_tol=1e-6)


def same_result(expected, actual):
    if actual is None or NUMBER.sub("#", expected) != NUMBER.sub("#", actual):
        return False
    return all(same_number(a, b) for a, b in zip(NUMBER.findall(expected), NUMBER.findall(actual), strict=True))


def command(case):
    project = str(MODELS / case["project"])
    if case.get("build"):
        return [sys.executable, str(HERE / "build.py"), project, *case.get("args", [])]
    return [sys.executable, "-m", "paynt", project, *case.get("args", [])]


def run_once(case, timeout):
    paths = [str(ROOT), os.environ.get("PYTHONPATH", "")]
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=os.pathsep.join(filter(None, paths)))
    with tempfile.TemporaryFile("w+") as log:
        start = time.perf_counter()
        process = subprocess.Popen(command(case), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        timer = threading.Timer(timeout, process.kill)
        timer.start()
        _, status, usage = os.wait4(process.pid, 0)
        elapsed = time.perf_counter() - start
        timed_out = not timer.is_alive()
        timer.cancel()
        process.returncode = os.waitstatus_to_exitcode(status)
        log.seek(0)
        output = log.read()

    iterations = {}
    for kind, count in ITERATIONS.findall(output):
        iterations[kind] = iterations.get(kind, 0) + int(count)
    results = RESULT.findall(output)
    if timed_out:
        outcome = "timeout"
    elif process.returncode != 0:
        outcome = "error"
    elif "expected" in case and not same_result(case["expected"], results[-1] if results else None):
        outcome = "wrong result"
    else:
        outcome = "ok"
    return {
        "status": outcome,
        "time": round(elapsed, 2),
        "memory_mb": round(usage.ru_maxrss / 1024),
        "iterations": iterations,
        "result": results[-1] if results else None,
        "tail": output.splitlines()[-20:] if outcome != "ok" else [],
    }


def run_case(case, repeat, timeout):
    runs = [run_once(case, timeout) for _ in range(repeat)]
    failed = [run for run in runs if run["status"] != "ok"]
    first = failed[0] if failed else runs[0]
    return dict(
        first,
        time=statistics.median(run["time"] for run in runs),
        times=[run["time"] for run in runs],
        deterministic=case.get("deterministic", True),
    )


def environment():
    import stormpy
    import stormpy.info

    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), cpu)
    commit = os.environ.get("GITHUB_SHA")
    if commit is None:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        commit = git.stdout.strip() or None
    return {
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "storm": stormpy.info.storm_long_version(),
        "stormpy": stormpy.__version__,
        "python": platform.python_version(),
        "cpu": cpu,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=HERE / "cases.toml", type=Path)
    parser.add_argument("--only", help="run only the cases whose name matches this pattern, e.g. 'ar-*'")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case, the time is their median")
    parser.add_argument("--timeout", type=float, default=300, help="seconds after which a run is killed")
    parser.add_argument("--output", type=Path, default=Path("performance.json"))
    args = parser.parse_args()

    cases = tomllib.loads(args.cases.read_text())["case"]
    if args.only:
        cases = [case for case in cases if fnmatch.fnmatch(case["name"], args.only)]

    record = environment()
    record["cases"] = {}
    for case in cases:
        measured = run_case(case, args.repeat, args.timeout)
        record["cases"][case["name"]] = measured
        iterations = " ".join(f"{kind} {count}" for kind, count in measured["iterations"].items())
        print(
            f"{case['name']:40} {measured['status']:12} {measured['time']:8.1f} s {measured['memory_mb']:6} MB  {iterations:30} {measured['result']}",
            flush=True,
        )
        args.output.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
