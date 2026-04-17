"""
IVR Test Agent
==============
LiveKit Python agent that joins a room, listens to IVR audio via Deepgram STT,
waits for end-of-utterance events, evaluates assertions, and sends DTMF.

Runs as a long-lived worker process. Dispatched by ivr_test_call.py per call.

Usage (start worker):
    py execution/ivr_agent.py

Environment variables required:
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    DEEPGRAM_API_KEY
    ANTHROPIC_API_KEY (for exploratory mode)
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import anthropic
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions, JobContext
from livekit.plugins import deepgram, silero

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ivr_agent] %(levelname)s %(message)s")
log = logging.getLogger("ivr_agent")

MODEL = "claude-opus-4-6"
MAX_STEPS = 20
FUZZY_CONFIDENCE_THRESHOLD = 0.7


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    step_number: int
    action_type: str
    status: str            # passed | failed | skipped | timeout | error | fuzzy | degraded
    expected: Optional[str] = None
    actual: Optional[str] = None
    digit_sent: Optional[str] = None
    confidence: float = 1.0
    duration_ms: int = 0
    eou_triggered: bool = False
    note: Optional[str] = None


@dataclass
class CallResult:
    test_case: str
    room_name: str
    phone_number: str
    started_at: str
    ended_at: str
    duration_seconds: int
    status: str            # passed | failed | timeout | error
    steps: list = field(default_factory=list)
    full_transcript: list = field(default_factory=list)
    recording_path: Optional[str] = None
    transcript_path: Optional[str] = None
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Assertion engine
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_assertion(step: dict, transcript: str) -> tuple[bool, float, str]:
    """
    Evaluate an assert_transcript step against the given transcript text.
    Returns (passed, confidence, actual_text).
    """
    text = transcript.lower()
    a = step.get("assert_transcript", {})

    if "contains" in a:
        keyword = a["contains"].lower()
        passed = keyword in text
        confidence = 0.95 if passed else 0.0
        return passed, confidence, transcript

    if "contains_any" in a:
        keywords = [k.lower() for k in a["contains_any"]]
        passed = any(k in text for k in keywords)
        confidence = 0.90 if passed else 0.0
        return passed, confidence, transcript

    if "not_contains" in a:
        keyword = a["not_contains"].lower()
        passed = keyword not in text
        confidence = 0.95
        return passed, confidence, transcript

    if "regex" in a:
        pattern = a["regex"]
        match = re.search(pattern, transcript, re.IGNORECASE)
        passed = bool(match)
        confidence = 0.90 if passed else 0.0
        return passed, confidence, transcript

    # No assertion condition — always pass
    return True, 1.0, transcript


# ──────────────────────────────────────────────────────────────────────────────
# AI reasoning (exploratory mode)
# ──────────────────────────────────────────────────────────────────────────────

def ask_claude_for_dtmf(transcript: str, history: list[str], tried_digits: list[str]) -> Optional[str]:
    """
    Ask Claude which DTMF digit to press given the current IVR prompt.
    Returns a single digit string or None (hang up).
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    tried_str = ", ".join(tried_digits) if tried_digits else "none yet"
    history_str = "\n".join(f"- {h}" for h in history[-5:]) if history else "none"

    response = client.messages.create(
        model=MODEL,
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f"""You are navigating an IVR phone system for testing purposes.

Current prompt heard: "{transcript}"
Digits already tried at this menu: {tried_str}
Recent transcript history:
{history_str}

What single DTMF digit should be pressed to explore a new path?
If no more paths to explore, respond with "HANGUP".
Respond with ONLY a single digit (0-9, *, #) or the word HANGUP. No explanation."""
        }]
    )

    result = response.content[0].text.strip().upper()
    if result == "HANGUP":
        return None
    # Validate: must be a single valid DTMF character
    if len(result) == 1 and result in "0123456789*#":
        return result
    # If Claude returned more than one char, try to extract first valid digit
    for char in result:
        if char in "0123456789*#":
            return char
    return None


# ──────────────────────────────────────────────────────────────────────────────
# IVR Test Agent
# ──────────────────────────────────────────────────────────────────────────────

class IVRTestAgent(Agent):
    def __init__(self, test_case: dict, room_name: str):
        super().__init__(instructions="IVR test bot. Listen to prompts and follow test steps.")
        self.test_case = test_case
        self.room_name = room_name
        self.steps = test_case.get("steps", [])
        self.test_type = test_case.get("type", "deterministic")

        self.current_step = 0
        self.step_results: list[StepResult] = []
        self.transcript_buffer: list[str] = []
        self.full_transcript: list[str] = []

        # Exploratory mode state
        self.visited_states: set[str] = set()
        self.dtmf_tried_at_state: dict[str, list[str]] = {}
        self.exploratory_history: list[str] = []

        self.started_at = datetime.now(timezone.utc).isoformat()
        self.step_start_time: float = 0.0

        # Set by session after start
        self._session: Optional[AgentSession] = None
        self._room: Optional[rtc.Room] = None
        self._finalized = False

    async def on_enter(self):
        """Called when agent joins the room."""
        log.info(f"[{self.room_name}] Agent entered room for test: {self.test_case['name']}")

    async def on_user_turn_completed(self, turn_ctx, new_message):
        """
        Called by LiveKit when the EOU model detects end of utterance.
        This is the main event loop hook for IVR testing.
        """
        transcript = new_message.text_content if hasattr(new_message, 'text_content') else str(new_message)
        if not transcript:
            return

        self.transcript_buffer.append(transcript)
        self.full_transcript.append(transcript)
        log.info(f"[{self.room_name}] Transcript: {transcript[:80]}")

        if self._finalized:
            return

        if self.test_type == "exploratory":
            await self._handle_exploratory_step(transcript)
        else:
            await self._handle_deterministic_step(transcript)

    async def _handle_deterministic_step(self, transcript: str):
        """Process deterministic test steps in order."""
        import time

        while self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            step_key = list(step.keys())[0]

            # wait steps are resolved by the EOU event itself — just advance
            if step_key == "wait":
                eou_triggered = step["wait"].get("until", "eou") == "eou"
                self.step_results.append(StepResult(
                    step_number=self.current_step + 1,
                    action_type="wait",
                    status="passed",
                    eou_triggered=eou_triggered,
                    duration_ms=int((time.time() - self.step_start_time) * 1000) if self.step_start_time else 0
                ))
                self.current_step += 1
                self.step_start_time = time.time()
                continue

            elif step_key == "assert_transcript":
                passed, confidence, actual = evaluate_assertion(step, " ".join(self.transcript_buffer))
                status = "passed" if passed else "failed"
                if not passed and confidence > 0 and confidence < FUZZY_CONFIDENCE_THRESHOLD:
                    status = "fuzzy"
                expected = (
                    step["assert_transcript"].get("contains") or
                    step["assert_transcript"].get("contains_any") or
                    step["assert_transcript"].get("not_contains") or
                    step["assert_transcript"].get("regex")
                )
                self.step_results.append(StepResult(
                    step_number=self.current_step + 1,
                    action_type="assert_transcript",
                    status=status,
                    expected=str(expected),
                    actual=actual[:200],
                    confidence=confidence
                ))
                if not passed:
                    log.warning(f"[{self.room_name}] Assertion FAILED at step {self.current_step + 1}: expected '{expected}' in '{actual[:80]}'")
                self.current_step += 1
                self.transcript_buffer.clear()

            elif step_key == "send_dtmf":
                digit = step["send_dtmf"]["digit"]
                delay_ms = step["send_dtmf"].get("post_delay_ms", 500)
                await self._publish_dtmf(digit)
                await asyncio.sleep(delay_ms / 1000)
                self.step_results.append(StepResult(
                    step_number=self.current_step + 1,
                    action_type="send_dtmf",
                    status="sent",
                    digit_sent=digit
                ))
                self.current_step += 1
                # After DTMF, break and wait for next EOU
                break

            elif step_key == "assert_voicemail":
                expected_vm = step["assert_voicemail"].get("expected", True)
                vm_keywords = ["leave a message", "after the tone", "after the beep", "voicemail", "not available"]
                transcript_lower = " ".join(self.transcript_buffer).lower()
                vm_detected = any(k in transcript_lower for k in vm_keywords)
                passed = vm_detected == expected_vm
                self.step_results.append(StepResult(
                    step_number=self.current_step + 1,
                    action_type="assert_voicemail",
                    status="passed" if passed else "failed",
                    expected=str(expected_vm),
                    actual=f"voicemail_detected={vm_detected}"
                ))
                self.current_step += 1

            elif step_key == "leave_voicemail":
                audio_file = step["leave_voicemail"].get("audio")
                if audio_file:
                    await self._play_audio_file(audio_file)
                else:
                    await asyncio.sleep(5)  # 5s silence
                self.step_results.append(StepResult(
                    step_number=self.current_step + 1,
                    action_type="leave_voicemail",
                    status="sent",
                    note=audio_file or "5s_silence"
                ))
                self.current_step += 1
                await self._finalize()
                return

            elif step_key == "assert_transfer":
                # Transfer detection is handled by room event monitoring (participant disconnect)
                # For now, record as pending and resolve in room event handler
                self.step_results.append(StepResult(
                    step_number=self.current_step + 1,
                    action_type="assert_transfer",
                    status="monitoring",
                    expected=step["assert_transfer"].get("to_number", "any")
                ))
                self.current_step += 1
                break

            else:
                self.current_step += 1

        # All steps completed
        if self.current_step >= len(self.steps) and not self._finalized:
            await self._finalize()

    async def _handle_exploratory_step(self, transcript: str):
        """For exploratory tests: ask Claude what to press."""
        # Loop detection
        state_key = hashlib.md5(f"{transcript[:120]}".encode()).hexdigest()
        tried = self.dtmf_tried_at_state.get(state_key, [])

        if len(self.step_results) >= MAX_STEPS:
            log.info(f"[{self.room_name}] Max steps reached ({MAX_STEPS}), finalizing")
            await self._finalize()
            return

        # Ask Claude for next digit
        digit = ask_claude_for_dtmf(transcript, self.exploratory_history, tried)
        self.exploratory_history.append(transcript[:80])

        if digit is None:
            log.info(f"[{self.room_name}] Claude says HANGUP — no more paths to explore")
            self.step_results.append(StepResult(
                step_number=len(self.step_results) + 1,
                action_type="exploratory_decision",
                status="hangup",
                actual=transcript[:200],
                note="Claude decided to hang up"
            ))
            await self._finalize()
            return

        # Check if we've already tried this digit at this menu
        new_state_key = hashlib.md5(f"{transcript[:120]}{digit}".encode()).hexdigest()
        if new_state_key in self.visited_states:
            log.warning(f"[{self.room_name}] Loop detected at step {len(self.step_results) + 1}")
            self.step_results.append(StepResult(
                step_number=len(self.step_results) + 1,
                action_type="exploratory_decision",
                status="loop_detected",
                note=f"State revisited at digit={digit}"
            ))
            await self._finalize()
            return

        self.visited_states.add(new_state_key)
        self.dtmf_tried_at_state.setdefault(state_key, []).append(digit)

        # Record and send
        self.step_results.append(StepResult(
            step_number=len(self.step_results) + 1,
            action_type="exploratory_dtmf",
            status="sent",
            digit_sent=digit,
            actual=transcript[:200],
            note="Claude-decided"
        ))
        await self._publish_dtmf(digit)

    async def _publish_dtmf(self, digit: str):
        """Send a DTMF digit via LiveKit."""
        try:
            if self._session and hasattr(self._session, 'room'):
                room = self._session.room
                await room.local_participant.publish_dtmf(
                    rtc.SipDTMF(code=ord(digit), digit=digit)
                )
                log.info(f"[{self.room_name}] DTMF sent: {digit}")
            else:
                log.warning(f"[{self.room_name}] Cannot send DTMF: no room reference")
        except Exception as e:
            log.error(f"[{self.room_name}] DTMF send error: {e}")
            # Retry once with delay
            await asyncio.sleep(0.2)
            try:
                if self._session and hasattr(self._session, 'room'):
                    await self._session.room.local_participant.publish_dtmf(
                        rtc.SipDTMF(code=ord(digit), digit=digit)
                    )
                    log.info(f"[{self.room_name}] DTMF retry succeeded: {digit}")
            except Exception as e2:
                log.error(f"[{self.room_name}] DTMF retry also failed: {e2}")

    async def _play_audio_file(self, audio_file: str):
        """Play a WAV file into the call (for voicemail). Currently plays silence as fallback."""
        # TODO: implement actual audio file playback via LiveKit audio track injection
        # For MVP: just wait the duration of a typical voicemail message
        log.info(f"[{self.room_name}] Playing voicemail audio: {audio_file} (or 5s silence)")
        await asyncio.sleep(5)

    async def _finalize(self):
        """Write result JSON and disconnect."""
        if self._finalized:
            return
        self._finalized = True

        ended_at = datetime.now(timezone.utc).isoformat()
        started_dt = datetime.fromisoformat(self.started_at)
        ended_dt = datetime.fromisoformat(ended_at)
        duration = int((ended_dt - started_dt).total_seconds())

        # Determine overall status
        step_statuses = [r.status for r in self.step_results]
        if "failed" in step_statuses:
            overall = "failed"
        elif "timeout" in step_statuses or "error" in step_statuses:
            overall = "error"
        else:
            overall = "passed"

        result = CallResult(
            test_case=self.test_case["name"],
            room_name=self.room_name,
            phone_number=self.test_case.get("phone_number", ""),
            started_at=self.started_at,
            ended_at=ended_at,
            duration_seconds=duration,
            status=overall,
            steps=[asdict(s) for s in self.step_results],
            full_transcript=self.full_transcript,
            recording_path=f".tmp/recordings/{self.room_name}.mp3",
            transcript_path=f".tmp/transcripts/{self.room_name}.json"
        )

        # Write transcript
        os.makedirs(".tmp/transcripts", exist_ok=True)
        with open(f".tmp/transcripts/{self.room_name}.json", "w") as f:
            json.dump(self.full_transcript, f, indent=2)

        # Write result
        os.makedirs(".tmp/ivr_results", exist_ok=True)
        result_path = f".tmp/ivr_results/{self.room_name}.json"
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        log.info(f"[{self.room_name}] Finalized: {overall} ({len(self.step_results)} steps, {duration}s)")
        log.info(f"[{self.room_name}] Result written: {result_path}")

        # Disconnect
        if self._session:
            try:
                await self._session.aclose()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Agent entrypoint
# ──────────────────────────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext):
    """Called by LiveKit worker framework for each dispatched job."""
    await ctx.connect()

    room = ctx.room
    room_name = room.name
    log.info(f"[{room_name}] Job started")

    # Extract test case from dispatch metadata
    metadata = ctx.job.metadata if ctx.job.metadata else "{}"
    try:
        job_data = json.loads(metadata)
        test_case = job_data.get("test_case", {})
    except json.JSONDecodeError:
        log.error(f"[{room_name}] Invalid metadata JSON: {metadata}")
        return

    if not test_case:
        log.error(f"[{room_name}] No test_case in metadata")
        return

    log.info(f"[{room_name}] Running: {test_case.get('name', 'unnamed')}")

    agent = IVRTestAgent(test_case=test_case, room_name=room_name)

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="en",
            smart_format=True,
            api_key=os.environ["DEEPGRAM_API_KEY"]
        ),
        vad=silero.VAD.load(),
        # No TTS — we use DTMF, not speech output
    )

    agent._session = session

    # Set up room event handlers for transfer detection
    @room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        log.info(f"[{room_name}] Participant disconnected: {participant.identity}")
        # Check if this is the SIP/IVR participant disconnecting (transfer complete)
        if participant.identity == "ivr-target":
            # Find any pending transfer assertion and resolve it
            for result in agent.step_results:
                if result.action_type == "assert_transfer" and result.status == "monitoring":
                    result.status = "passed"
                    result.note = "SIP participant disconnected (transfer or hangup)"
                    log.info(f"[{room_name}] Transfer assertion resolved: {result.expected}")

    # Start agent session
    await session.start(
        room=room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=None  # raw audio — don't noise cancel (may affect IVR tones)
        )
    )

    # Timeout watchdog
    max_duration = int(os.environ.get("IVR_MAX_CALL_DURATION", 300))
    try:
        await asyncio.wait_for(session.wait_for_disconnect(), timeout=max_duration)
    except asyncio.TimeoutError:
        log.warning(f"[{room_name}] Call exceeded max duration ({max_duration}s), forcing finalize")
        # Mark any in-progress step as timeout
        for result in agent.step_results:
            if result.status in ("monitoring", "in_progress"):
                result.status = "timeout"
        if not agent._finalized:
            await agent._finalize()


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="ivr-test-agent"
        )
    )

# revised

# rev 1

# rev 5
