"""
IVR Test Call Launcher
======================
Launches a single outbound test call:
  1. Creates a LiveKit room
  2. Dispatches ivr_agent to the room with test case metadata
  3. Dials the phone number via LiveKit SIP (Twilio Elastic SIP Trunk)
  4. Polls for result file
  5. Returns result path

Called by ivr_test_runner.py. Not meant to be run directly.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("ivr_test_call")


async def run_test_call(test_case: dict, phone_number: str) -> dict:
    """
    Execute one test case. Returns the result dict.
    Raises TimeoutError if call does not complete within IVR_MAX_CALL_DURATION.
    """
    from livekit import api

    room_name = f"ivr-test-{uuid4()}"
    max_duration = int(os.environ.get("IVR_MAX_CALL_DURATION", 300))
    livekit_url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    sip_trunk_sid = os.environ.get("TWILIO_SIP_TRUNK_SID", "")
    livekit_sip_trunk_id = os.environ.get("LIVEKIT_SIP_TRUNK_ID", "")

    # Attach phone_number to test_case so agent can record it
    test_case_with_phone = {**test_case, "phone_number": phone_number}

    lk_api = api.LiveKitAPI(url=livekit_url, api_key=api_key, api_secret=api_secret)

    try:
        # 1. Create room
        log.info(f"[{room_name}] Creating room")
        await lk_api.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=30,
                max_participants=10,
            )
        )

        # 2. Dispatch ivr-test-agent
        log.info(f"[{room_name}] Dispatching ivr-test-agent")
        await lk_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="ivr-test-agent",
                room=room_name,
                metadata=json.dumps({"test_case": test_case_with_phone})
            )
        )

        # Small delay to let agent connect before the SIP call arrives
        await asyncio.sleep(1.5)

        # 3. Dial the phone number via LiveKit SIP
        log.info(f"[{room_name}] Dialing {phone_number}")

        # Build SIP URL — Twilio Elastic SIP Trunking endpoint
        sip_call_to = f"sip:{phone_number.lstrip('+')}@sip.twilio.com"

        # Use LIVEKIT_SIP_TRUNK_ID (the LiveKit-side trunk ID) if available,
        # fall back to TWILIO_SIP_TRUNK_SID if that's what was configured
        trunk_id = livekit_sip_trunk_id or sip_trunk_sid

        sip_request = api.CreateSIPParticipantRequest(
            sip_trunk_id=trunk_id,
            sip_call_to=sip_call_to,
            room_name=room_name,
            participant_identity="ivr-target",
            participant_name="IVR Under Test",
        )
        await lk_api.sip.create_sip_participant(sip_request)

        # 4. Poll for result file
        result_path = f".tmp/ivr_results/{room_name}.json"
        deadline = time.time() + max_duration
        poll_interval = 2.0

        log.info(f"[{room_name}] Waiting for agent to complete (max {max_duration}s)")
        while time.time() < deadline:
            if os.path.exists(result_path):
                with open(result_path) as f:
                    result = json.load(f)
                log.info(f"[{room_name}] Call complete: {result.get('status', '?')} ({result.get('duration_seconds', 0)}s)")
                return result
            await asyncio.sleep(poll_interval)

        # Timeout — build timeout result
        log.warning(f"[{room_name}] Timed out after {max_duration}s")
        timeout_result = {
            "test_case": test_case.get("name", "unknown"),
            "room_name": room_name,
            "phone_number": phone_number,
            "status": "timeout",
            "steps": [],
            "full_transcript": [],
            "error": f"Call timed out after {max_duration}s",
            "duration_seconds": max_duration,
        }

        # Write timeout result so aggregator picks it up
        os.makedirs(".tmp/ivr_results", exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(timeout_result, f, indent=2)

        return timeout_result

    except Exception as e:
        log.error(f"[{room_name}] Call setup error: {e}")
        error_result = {
            "test_case": test_case.get("name", "unknown"),
            "room_name": room_name,
            "phone_number": phone_number,
            "status": "error",
            "steps": [],
            "full_transcript": [],
            "error": str(e),
            "duration_seconds": 0,
        }
        os.makedirs(".tmp/ivr_results", exist_ok=True)
        result_path = f".tmp/ivr_results/{room_name}.json"
        with open(result_path, "w") as f:
            json.dump(error_result, f, indent=2)
        return error_result

    finally:
        # Always clean up the room
        try:
            await lk_api.room.delete_room(api.DeleteRoomRequest(name=room_name))
            log.debug(f"[{room_name}] Room deleted")
        except Exception:
            pass
        try:
            await lk_api.aclose()
        except Exception:
            pass

# rev 2
