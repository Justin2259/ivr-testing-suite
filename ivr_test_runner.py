"""
IVR Test Runner
===============
Loads a test suite YAML, starts the ivr_agent worker process,
runs all test cases in parallel (asyncio semaphore), and aggregates results.

Usage:
    py execution/ivr_test_runner.py --suite test_suites/name.yaml
    py execution/ivr_test_runner.py --suite test_suites/name.yaml --tags regression
    py execution/ivr_test_runner.py --suite test_suites/name.yaml --concurrency 3
    py execution/ivr_test_runner.py --suite test_suites/name.yaml --case "English main menu"
    py execution/ivr_test_runner.py --suite test_suites/name.yaml --dry-run
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from glob import glob
from uuid import uuid4

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [runner] %(levelname)s %(message)s"
)
log = logging.getLogger("ivr_test_runner")


def load_suite(suite_path: str) -> dict:
    """Load and validate test suite YAML."""
    if not os.path.exists(suite_path):
        print(f"Error: Test suite not found: {suite_path}")
        sys.exit(1)
    with open(suite_path) as f:
        suite = yaml.safe_load(f)
    if not suite or "test_cases" not in suite:
        print(f"Error: Invalid test suite (missing 'test_cases'): {suite_path}")
        sys.exit(1)
    return suite


def filter_cases(cases: list, tags: list = None, case_name: str = None) -> list:
    """Filter test cases by tags or name."""
    if case_name:
        filtered = [tc for tc in cases if tc.get("name", "").lower() == case_name.lower()]
        if not filtered:
            print(f"Warning: No test case found with name '{case_name}'")
        return filtered
    if tags:
        filtered = []
        for tc in cases:
            tc_tags = tc.get("tags", [])
            if any(t in tc_tags for t in tags):
                filtered.append(tc)
        return filtered
    return cases


def start_agent_worker() -> subprocess.Popen:
    """Start ivr_agent.py as a background worker process."""
    log.info("Starting ivr-test-agent worker process...")
    proc = subprocess.Popen(
        [sys.executable, "execution/ivr_agent.py", "start"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy()
    )
    # Give it a moment to register with LiveKit
    time.sleep(3)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        print(f"Error: ivr_agent.py worker failed to start:\n{stderr}")
        sys.exit(1)
    log.info(f"Agent worker started (PID {proc.pid})")
    return proc


def aggregate_results(run_id: str, suite: dict, results: list) -> dict:
    """Combine all per-call results into a suite-level summary."""
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    blocked = sum(1 for r in results if r.get("status") in ("timeout", "error"))

    return {
        "run_id": run_id,
        "suite_name": suite.get("test_suite", "unnamed"),
        "phone_number": suite.get("phone_number", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "passed": passed,
        "failed": failed,
        "blocked": blocked,
        "coverage_pct": round((passed / total * 100) if total else 0, 1),
        "test_cases": results
    }


def print_summary(aggregated: dict):
    total = aggregated["total"]
    passed = aggregated["passed"]
    failed = aggregated["failed"]
    blocked = aggregated["blocked"]

    print("\n" + "=" * 60)
    print(f"IVR Test Suite Complete: {aggregated['suite_name']}")
    print(f"Phone: {aggregated['phone_number']}")
    print(f"Run ID: {aggregated['run_id']}")
    print("=" * 60)
    print(f"Total:   {total}")
    print(f"Passed:  {passed} ({round(passed/total*100) if total else 0}%)")
    print(f"Failed:  {failed} ({round(failed/total*100) if total else 0}%)")
    print(f"Blocked: {blocked} ({round(blocked/total*100) if total else 0}%)")
    print()

    if failed:
        print("FAILURES:")
        for r in aggregated["test_cases"]:
            if r.get("status") == "failed":
                failed_steps = [s for s in r.get("steps", []) if s.get("status") == "failed"]
                step_info = ""
                if failed_steps:
                    s = failed_steps[0]
                    step_info = f" (step {s['step_number']}: expected '{s.get('expected', '?')}', got '{str(s.get('actual', ''))[:50]}')"
                print(f"  [FAIL] {r['test_case']}{step_info}")

    if blocked:
        print("\nBLOCKED:")
        for r in aggregated["test_cases"]:
            if r.get("status") in ("timeout", "error"):
                err = r.get("error", "")
                print(f"  [{r['status'].upper()}] {r['test_case']}" + (f" — {err}" if err else ""))

    print()


async def run_suite_async(suite: dict, concurrency: int, agent_proc: subprocess.Popen) -> list:
    """Run all test cases in parallel up to concurrency limit."""
    from execution.ivr_test_call import run_test_call

    phone_number = suite["phone_number"]
    test_cases = suite["test_cases"]
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_run(tc: dict) -> dict:
        async with semaphore:
            name = tc.get("name", "unnamed")
            log.info(f"Starting: {name}")
            try:
                result = await run_test_call(tc, phone_number)
            except Exception as e:
                log.error(f"Unexpected error for '{name}': {e}")
                result = {
                    "test_case": name,
                    "phone_number": phone_number,
                    "status": "error",
                    "error": str(e),
                    "steps": [],
                    "full_transcript": [],
                    "duration_seconds": 0
                }
            log.info(f"Done: {name} [{result.get('status', '?')}]")
            return result

    tasks = [bounded_run(tc) for tc in test_cases]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)


def main():
    parser = argparse.ArgumentParser(description="Run IVR test suite")
    parser.add_argument("--suite", required=True, help="Path to test suite YAML")
    parser.add_argument("--tags", help="Comma-separated tags to filter (e.g. regression,after_hours)")
    parser.add_argument("--concurrency", type=int, help="Override IVR_MAX_CONCURRENT_CALLS")
    parser.add_argument("--case", help="Run a single named test case only")
    parser.add_argument("--dry-run", action="store_true", help="Validate suite and print cases, do not call")
    parser.add_argument("--no-report", action="store_true", help="Skip auto-generating report after run")
    args = parser.parse_args()

    # Load suite
    suite = load_suite(args.suite)
    phone = suite.get("phone_number", "")
    if not phone:
        print("Error: phone_number not set in test suite YAML")
        sys.exit(1)

    # Filter cases
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
    cases = filter_cases(suite["test_cases"], tags=tags, case_name=args.case)
    suite = {**suite, "test_cases": cases}

    # Dry run
    if args.dry_run:
        print(f"\nDry run: {args.suite}")
        print(f"Phone: {phone}")
        print(f"Test cases ({len(cases)}):")
        for tc in cases:
            tags_str = ", ".join(tc.get("tags", []))
            print(f"  [{tc.get('type', '?')}] {tc['name']}" + (f" ({tags_str})" if tags_str else ""))
        concurrency = args.concurrency or suite.get("max_concurrent", int(os.environ.get("IVR_MAX_CONCURRENT_CALLS", 5)))
        avg_min = 3
        est = max(1, len(cases) * avg_min // concurrency)
        print(f"\nConcurrency: {concurrency} | Estimated time: ~{est} min")
        print("(dry-run: no calls placed)")
        return

    if not cases:
        print("No test cases match the filter. Nothing to run.")
        sys.exit(0)

    concurrency = args.concurrency or suite.get("max_concurrent", int(os.environ.get("IVR_MAX_CONCURRENT_CALLS", 5)))

    print(f"\nStarting IVR test run")
    print(f"Suite: {suite.get('test_suite', 'unnamed')}")
    print(f"Phone: {phone}")
    print(f"Cases: {len(cases)} | Concurrency: {concurrency}")
    print()

    # Ensure output dirs exist
    os.makedirs(".tmp/ivr_results", exist_ok=True)
    os.makedirs(".tmp/recordings", exist_ok=True)
    os.makedirs(".tmp/transcripts", exist_ok=True)

    # Start agent worker
    agent_proc = start_agent_worker()

    run_id = str(uuid4())
    start_wall = time.time()

    try:
        results = asyncio.run(run_suite_async(suite, concurrency, agent_proc))
    finally:
        # Stop agent worker
        agent_proc.terminate()
        try:
            agent_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent_proc.kill()
        log.info("Agent worker stopped")

    elapsed = int(time.time() - start_wall)
    log.info(f"All calls completed in {elapsed}s")

    # Aggregate results
    aggregated = aggregate_results(run_id, suite, results)
    print_summary(aggregated)

    # Write aggregated results
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = f".tmp/ivr_results_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(f"Results written: {results_path}")

    # Auto-generate report unless suppressed
    if not args.no_report:
        report_path = f".tmp/ivr_report_{ts}.html"
        print(f"\nGenerating report: {report_path}")
        ret = subprocess.run(
            [sys.executable, "execution/ivr_generate_report.py",
             "--results", results_path,
             "--output", report_path,
             "--open"],
            env=os.environ.copy()
        )
        if ret.returncode == 0:
            print(f"Report ready: {report_path}")
        else:
            print("Warning: Report generation failed. Run manually:")
            print(f"  py execution/ivr_generate_report.py --results {results_path} --output {report_path}")

    # Exit with failure code if any test failed or was blocked
    if aggregated["failed"] > 0 or aggregated["blocked"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

# revised

# rev 4

# rev 7

# rev 8
