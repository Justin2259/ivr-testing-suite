"""
IVR Test Case Generator
=======================
Generates a test suite YAML from a Genesys IVR flow (Mode A) or natural language
description (Mode B). Also validates existing YAML files.

Usage:
    # Mode A: from Genesys flow
    py execution/ivr_test_case_generator.py --flow-id <id> --phone +1... --output test_suites/name.yaml

    # Mode B: from description
    py execution/ivr_test_case_generator.py --describe "IVR description..." --phone +1... --output test_suites/name.yaml

    # Validate existing
    py execution/ivr_test_case_generator.py --validate test_suites/name.yaml
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import anthropic
import yaml
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-6"

TEST_CASE_SCHEMA_DESCRIPTION = """
Test case YAML schema:

test_suite: "Suite name"
phone_number: "+1XXXXXXXXXX"
max_concurrent: 5
max_call_duration_seconds: 300
test_cases:
  - name: "Unique descriptive name"
    type: deterministic | exploratory | voicemail | transfer | after_hours
    tags: [list of strings]
    steps:
      - wait: {until: eou|silence|transfer_or_answer, max_seconds: 15}
      - assert_transcript: {contains: "string"}
        # OR: {contains_any: ["opt1", "opt2"]}
        # OR: {not_contains: "string"}
        # OR: {regex: "pattern"}
      - send_dtmf: {digit: "1", post_delay_ms: 500}
      - assert_transfer: {to_number: "+1..."}
      - assert_voicemail: {expected: true}
      - leave_voicemail: {audio: "filename.wav"}

Rules:
- Always start with wait before first assertion
- Always wait after send_dtmf before next assertion
- assert_transfer is always the last step in a transfer test
- assert_voicemail comes before leave_voicemail
- contains/contains_any use case-insensitive substring matching
"""


def build_generation_prompt(flow_info: str, phone: str) -> str:
    return f"""You are generating an IVR test suite for automated call flow testing.

Phone number under test: {phone}

IVR Flow Information:
{flow_info}

Generate a comprehensive test suite covering:
1. Each main menu option (one test case per option)
2. Language selection paths (if the IVR has language options)
3. Invalid input handling (press an invalid key, expect re-prompt or error message)
4. Transfer paths (any call transfer destinations)
5. After-hours routing (if schedule-based routing is present)
6. Voicemail paths (if voicemail nodes exist)
7. Repeat menu / timeout handling

{TEST_CASE_SCHEMA_DESCRIPTION}

Return ONLY a JSON array of test cases (no YAML, no explanation, just the JSON array).
Each test case must have: name, type, tags, steps.
Use double-quoted strings. Keep assertion text short (3-6 words) to be resilient to minor prompt changes.
For DTMF digits, use only what the flow definition specifies. If unsure, add a comment in the name field.

Example output format:
[
  {{
    "name": "English main menu - press 1 for support",
    "type": "deterministic",
    "tags": ["regression", "business_hours", "english"],
    "steps": [
      {{"wait": {{"until": "eou", "max_seconds": 15}}}},
      {{"assert_transcript": {{"contains": "thank you for calling"}}}},
      {{"send_dtmf": {{"digit": "1", "post_delay_ms": 500}}}},
      {{"wait": {{"until": "eou", "max_seconds": 10}}}},
      {{"assert_transcript": {{"contains_any": ["support", "how can i help"]}}}}
    ]
  }}
]
"""


def get_genesys_flow_info(flow_id: str) -> str:
    """Fetch and parse Genesys IVR flow using existing execution scripts."""
    print(f"Fetching Genesys IVR flow: {flow_id}")

    # Step 1: Fetch raw flow
    fetch_result = subprocess.run(
        [sys.executable, "execution/fetch_ivr_flow.py", "--flow-id", flow_id, "--output", f".tmp/ivr_flow_{flow_id}.yaml"],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    if fetch_result.returncode != 0:
        print(f"Warning: fetch_ivr_flow.py returned non-zero: {fetch_result.stderr}")

    # Step 2: Parse the flow
    parse_result = subprocess.run(
        [sys.executable, "execution/parse_ivr_flow.py", "--input", f".tmp/ivr_flow_{flow_id}.yaml", "--output", f".tmp/ivr_parsed_{flow_id}.json"],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    if parse_result.returncode != 0:
        print(f"Warning: parse_ivr_flow.py returned non-zero: {parse_result.stderr}")

    # Step 3: Read parsed output
    parsed_path = f".tmp/ivr_parsed_{flow_id}.json"
    if os.path.exists(parsed_path):
        with open(parsed_path) as f:
            data = json.load(f)
        # Summarize to keep prompt size manageable
        return json.dumps(data, indent=2)[:8000]
    else:
        # Fallback: use raw flow YAML
        raw_path = f".tmp/ivr_flow_{flow_id}.yaml"
        if os.path.exists(raw_path):
            with open(raw_path) as f:
                return f.read()[:8000]
        return f"[Could not fetch flow {flow_id}. Generating test cases based on flow ID only.]"


def generate_test_cases(flow_info: str, phone: str) -> list:
    """Send flow info to Claude and get test cases back."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = build_generation_prompt(flow_info, phone)

    print(f"Generating test cases via Claude {MODEL}...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=8096,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Extract JSON array from response (Claude may wrap in markdown code block)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        cases = json.loads(raw)
        if not isinstance(cases, list):
            raise ValueError("Expected JSON array")
        return cases
    except json.JSONDecodeError as e:
        print(f"Error: Claude returned invalid JSON: {e}")
        print(f"Raw response:\n{raw[:500]}")
        raise


def build_suite_yaml(test_cases: list, phone: str, suite_name: str) -> dict:
    return {
        "test_suite": suite_name,
        "phone_number": phone,
        "max_concurrent": int(os.environ.get("IVR_MAX_CONCURRENT_CALLS", 5)),
        "max_call_duration_seconds": int(os.environ.get("IVR_MAX_CALL_DURATION", 300)),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "test_cases": test_cases
    }


VALID_STEP_KEYS = {"wait", "assert_transcript", "send_dtmf", "assert_transfer", "assert_voicemail", "leave_voicemail"}
VALID_TYPES = {"deterministic", "exploratory", "voicemail", "transfer", "after_hours"}
VALID_UNTIL = {"eou", "silence", "transfer_or_answer"}


def validate_suite(data: dict) -> list:
    """Returns list of validation errors."""
    errors = []

    if "test_suite" not in data:
        errors.append("Missing 'test_suite' field")
    if "phone_number" not in data:
        errors.append("Missing 'phone_number' field")
    elif not data["phone_number"].startswith("+"):
        errors.append(f"phone_number must be E.164 format (start with +): {data['phone_number']}")
    if "test_cases" not in data or not isinstance(data["test_cases"], list):
        errors.append("Missing or invalid 'test_cases' list")
        return errors

    for i, tc in enumerate(data["test_cases"]):
        prefix = f"test_cases[{i}] ({tc.get('name', 'unnamed')})"
        if "name" not in tc:
            errors.append(f"{prefix}: missing 'name'")
        if "type" not in tc:
            errors.append(f"{prefix}: missing 'type'")
        elif tc["type"] not in VALID_TYPES:
            errors.append(f"{prefix}: invalid type '{tc['type']}'. Valid: {VALID_TYPES}")
        if "steps" not in tc or not isinstance(tc.get("steps"), list):
            errors.append(f"{prefix}: missing or invalid 'steps' list")
            continue
        for j, step in enumerate(tc["steps"]):
            if not isinstance(step, dict) or len(step) != 1:
                errors.append(f"{prefix} step[{j}]: must be dict with exactly one key")
                continue
            key = list(step.keys())[0]
            if key not in VALID_STEP_KEYS:
                errors.append(f"{prefix} step[{j}]: unknown step type '{key}'")
            if key == "wait":
                until = step["wait"].get("until", "")
                if until not in VALID_UNTIL:
                    errors.append(f"{prefix} step[{j}]: invalid 'until' value '{until}'. Valid: {VALID_UNTIL}")
            if key == "send_dtmf":
                digit = step["send_dtmf"].get("digit", "")
                if not digit or digit not in "0123456789*#":
                    errors.append(f"{prefix} step[{j}]: invalid DTMF digit '{digit}'")

    return errors


def print_suite_summary(data: dict):
    cases = data.get("test_cases", [])
    types = {}
    for tc in cases:
        t = tc.get("type", "unknown")
        types[t] = types.get(t, 0) + 1

    concurrency = data.get("max_concurrent", 5)
    avg_duration = 3  # minutes per call estimate
    est_minutes = max(1, len(cases) * avg_duration // concurrency)

    print(f"\nTest suite: {data.get('test_suite', 'unnamed')}")
    print(f"Phone: {data.get('phone_number', 'not set')}")
    print(f"Total test cases: {len(cases)}")
    for t, count in sorted(types.items()):
        print(f"  {t}: {count}")
    print(f"Estimated run time: ~{est_minutes} min at concurrency={concurrency}")


def main():
    parser = argparse.ArgumentParser(description="Generate IVR test suite YAML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--flow-id", help="Genesys IVR flow ID (Mode A)")
    group.add_argument("--describe", help="Natural language IVR description (Mode B)")
    group.add_argument("--validate", metavar="YAML_FILE", help="Validate an existing test suite YAML")
    parser.add_argument("--phone", help="Phone number to test (E.164, required for A and B)")
    parser.add_argument("--output", help="Output YAML path (required for A and B)")
    parser.add_argument("--suite-name", help="Suite name (optional, defaults to filename)")
    args = parser.parse_args()

    # Validation-only mode
    if args.validate:
        with open(args.validate) as f:
            data = yaml.safe_load(f)
        errors = validate_suite(data)
        if errors:
            print(f"Validation FAILED ({len(errors)} errors):")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        else:
            print(f"Validation passed: {args.validate}")
            print_suite_summary(data)
            sys.exit(0)

    # Generation modes require --phone and --output
    if not args.phone:
        parser.error("--phone is required for generation modes")
    if not args.output:
        parser.error("--output is required for generation modes")

    os.makedirs(".tmp", exist_ok=True)

    # Get flow info
    if args.flow_id:
        flow_info = get_genesys_flow_info(args.flow_id)
    else:
        # Mode B: use the description directly
        flow_info = f"IVR Description (no flow definition available):\n{args.describe}"
        print("\nNote: Mode B generates test cases from a description. DTMF digits are inferred.")
        print("      Review the generated YAML carefully before running.\n")

    # Generate test cases
    test_cases = generate_test_cases(flow_info, args.phone)

    # Build suite name
    suite_name = args.suite_name
    if not suite_name:
        # Derive from output filename
        suite_name = os.path.splitext(os.path.basename(args.output))[0].replace("_", " ").title()

    # Build and validate suite
    suite = build_suite_yaml(test_cases, args.phone, suite_name)
    errors = validate_suite(suite)
    if errors:
        print(f"Warning: {len(errors)} schema issue(s) in generated suite:")
        for e in errors:
            print(f"  - {e}")

    # Write YAML
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(suite, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\nTest suite written to: {args.output}")
    print_suite_summary(suite)

    if args.describe:
        print("\n[!] Mode B: Review DTMF digits in the generated YAML before running.")
        print("    Generated cases are inferred from description, not from actual flow definition.")


if __name__ == "__main__":
    main()

# rev 3

# rev 6
