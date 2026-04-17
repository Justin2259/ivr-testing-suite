# IVR Testing Suite

Automated IVR testing via parallel AI call agents. Claude acts as the caller, Deepgram transcribes the IVR audio in real time, and a LiveKit-based assertion engine evaluates the call flow against a YAML-defined test suite.

Runs 20 concurrent simulated calls. Catches regressions in routing, prompts, DTMF handling, and after-hours schedules before they reach customers.

---

## Architecture

```
ivr_test_runner.py          orchestrates the suite
    |
    |-- starts ivr_agent.py worker (LiveKit agent process)
    |
    |-- for each test case (up to N concurrent):
    |       ivr_test_call.py
    |           creates LiveKit room
    |           dials phone via Twilio SIP trunk
    |           dispatches agent to room
    |           polls for result file
    |
    v
ivr_agent.py (per call)
    listens to IVR audio via Deepgram STT
    waits for end-of-utterance events
    evaluates assertions (expected prompt text, fuzzy match)
    sends DTMF digits
    records transcript + step results
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `ivr_agent.py` | LiveKit worker: STT, DTMF, assertion engine |
| `ivr_test_call.py` | Launches a single outbound test call |
| `ivr_test_case_generator.py` | Generates YAML test suites from Genesys flow definitions or natural language |
| `ivr_test_runner.py` | Orchestrates the full suite with concurrency and reporting |

---

## Test Suite Format

Test suites are YAML files. Each test case defines the expected call path:

```yaml
test_suite: Main IVR Regression
phone_number: "+18005551234"
max_concurrent: 5

test_cases:
  - name: English main menu
    type: scripted
    tags: [regression]
    steps:
      - expect: "thank you for calling"
        action: wait
      - expect: "press 1 for"
        action: dtmf
        digit: "1"
      - expect: "connecting you"
        action: assert

  - name: After-hours voicemail
    type: scripted
    tags: [after_hours]
    steps:
      - expect: "our offices are currently closed"
        action: assert
```

---

## Usage

```bash
# Run a full test suite
python ivr_test_runner.py --suite test_suites/main_ivr.yaml

# Run only regression-tagged cases
python ivr_test_runner.py --suite test_suites/main_ivr.yaml --tags regression

# Run a single named case
python ivr_test_runner.py --suite test_suites/main_ivr.yaml --case "English main menu"

# Dry run (validate suite, no calls placed)
python ivr_test_runner.py --suite test_suites/main_ivr.yaml --dry-run

# Override concurrency
python ivr_test_runner.py --suite test_suites/main_ivr.yaml --concurrency 3

# Generate a test suite from an IVR flow definition
python ivr_test_case_generator.py --flow-id <genesys-flow-id> --phone +18005551234 --output test_suites/main_ivr.yaml
```

---

## Environment Variables

```
# LiveKit (SIP-capable project)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret

# Deepgram (STT)
DEEPGRAM_API_KEY=your_key

# Anthropic (for test case generation + exploratory mode)
ANTHROPIC_API_KEY=your_key

# Genesys (for auto-generating test suites from flow definitions)
GENESYS_CLIENT_ID=your_client_id
GENESYS_CLIENT_SECRET=your_client_secret
GENESYS_REGION=mypurecloud.com

# Concurrency (default: 5)
IVR_MAX_CONCURRENT_CALLS=20
```

---

## Output

After each run, results are written to `.tmp/ivr_results_TIMESTAMP.json` and an HTML report is auto-generated. The runner exits with code 1 if any test fails or is blocked, making it CI-friendly.

```
IVR Test Suite Complete: Main IVR Regression
Phone: +18005551234
==============================
Total:   24
Passed:  22 (91%)
Failed:   1 (4%)
Blocked:  1 (4%)

FAILURES:
  [FAIL] Spanish after-hours (step 2: expected 'para espanol', got 'thank you for calling')
```
