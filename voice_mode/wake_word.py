"""
Background wake word listener for interrupting Claude's thinking phase.

When enabled, this module runs an asyncio Task after converse() returns the
user's transcript to Claude. While Claude is processing (thinking), the listener
records audio in short chunks, checks for the configured wake word via Whisper STT,
and if detected plays an acknowledgment then records the user's follow-up message.

The captured follow-up transcript is stored as a pending interruption. On the next
converse() call, the interruption is returned immediately instead of recording fresh.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("voicemode")

# ── Module-level state ────────────────────────────────────────────────────────

@dataclass
class Interruption:
    """A wake-word-triggered interruption from the user."""
    transcript: str
    detected_at: float


_pending_interruption: Optional[Interruption] = None
_listener_task: Optional[asyncio.Task] = None

# Energy threshold below which audio chunks are treated as silence
_ENERGY_THRESHOLD = 150
# Duration of each recorded chunk fed to the wake word detector
_CHUNK_DURATION_S = 2.0


# ── Public API ────────────────────────────────────────────────────────────────

def get_pending_interruption() -> Optional[Interruption]:
    """Return any interruption captured since the last clear, or None."""
    return _pending_interruption


def clear_pending_interruption() -> None:
    """Discard any stored interruption (call after it has been handled)."""
    global _pending_interruption
    _pending_interruption = None


def start_wake_word_listener(
    wake_word: str,
    acknowledgment: str,
    voice_params: dict,
) -> None:
    """
    Spawn a background asyncio Task to listen for the wake word.

    Call this just before converse() returns, while still inside the running
    event loop.  The task will be cancelled automatically the next time
    converse() is entered.

    Args:
        wake_word: Comma-separated words/phrases to trigger on (case-insensitive).
        acknowledgment: Text to speak when the wake word is heard.
        voice_params: Dict with TTS settings to pass through (voice, model,
                      audio_format, tts_provider, speed, instructions).
    """
    global _listener_task

    # Cancel any stale listener before starting a new one
    _cancel_listener_task()

    _listener_task = asyncio.create_task(
        _listen_loop(wake_word, acknowledgment, voice_params),
        name="wake_word_listener",
    )
    logger.debug("Wake word listener task started")


async def stop_wake_word_listener() -> None:
    """Cancel the background listener and wait for it to finish."""
    _cancel_listener_task()
    # Give the event loop a tick to process the cancellation
    await asyncio.sleep(0)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cancel_listener_task() -> None:
    global _listener_task
    if _listener_task and not _listener_task.done():
        _listener_task.cancel()
    _listener_task = None


async def _listen_loop(wake_word: str, acknowledgment: str, voice_params: dict) -> None:
    """
    Continuously record short audio chunks and test each for the wake word.

    Runs until cancelled (when converse() is called again) or until a wake
    word is detected and the follow-up has been captured.
    """
    from voice_mode.config import SAMPLE_RATE

    wake_words = [w.strip().lower() for w in wake_word.split(",") if w.strip()]
    chunk_samples = int(SAMPLE_RATE * _CHUNK_DURATION_S)

    logger.info(f"🔍 Wake word listener active — listening for: {wake_words}")

    try:
        while True:
            # Record one chunk without blocking the event loop
            loop = asyncio.get_event_loop()
            chunk = await loop.run_in_executor(
                None, _record_chunk_sync, chunk_samples, SAMPLE_RATE
            )

            if chunk is None:
                await asyncio.sleep(0.1)
                continue

            # Skip silent chunks to avoid wasting STT calls
            if np.abs(chunk).mean() < _ENERGY_THRESHOLD:
                continue

            # Transcribe with the existing STT pipeline
            transcript = await _transcribe_chunk(chunk)
            if not transcript:
                continue

            logger.debug(f"Wake word check: '{transcript}'")

            if _matches_wake_word(transcript, wake_words):
                logger.info(f"🔔 Wake word detected: '{transcript}'")
                await _handle_detection(acknowledgment, voice_params)
                return

    except asyncio.CancelledError:
        logger.debug("Wake word listener cancelled")
        raise


def _record_chunk_sync(num_samples: int, sample_rate: int) -> Optional[np.ndarray]:
    """Record a fixed-length audio chunk synchronously (runs in executor)."""
    import sounddevice as sd

    try:
        chunk = sd.rec(
            num_samples,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocking=True,
        )
        return chunk.flatten()
    except Exception as exc:
        logger.warning(f"Wake word chunk recording failed: {exc}")
        return None


async def _transcribe_chunk(audio_data: np.ndarray) -> Optional[str]:
    """Send a chunk to STT and return the transcript text, or None on failure."""
    # Import here to avoid circular imports at module level
    from voice_mode.tools.converse import speech_to_text

    try:
        result = await speech_to_text(
            audio_data,
            save_audio=False,
            audio_dir=None,
            transport="local",
        )
        if isinstance(result, dict):
            return result.get("text")
    except Exception as exc:
        logger.debug(f"Wake word STT error: {exc}")
    return None


def _matches_wake_word(transcript: str, wake_words: list[str]) -> bool:
    text = transcript.lower().strip()
    return any(w in text for w in wake_words)


async def _handle_detection(acknowledgment: str, voice_params: dict) -> None:
    """Play acknowledgment, record the user's follow-up, and store it."""
    global _pending_interruption

    from voice_mode.tools.converse import (
        text_to_speech_with_failover,
        speech_to_text,
        record_audio_with_silence_detection,
    )
    from voice_mode.config import (
        audio_operation_lock,
        SAVE_AUDIO,
        AUDIO_DIR,
        DEFAULT_LISTEN_DURATION,
        MIN_RECORDING_DURATION,
    )

    async with audio_operation_lock:
        # Speak the acknowledgment
        logger.info(f"🔔 Speaking acknowledgment: '{acknowledgment}'")
        await text_to_speech_with_failover(
            message=acknowledgment,
            voice=voice_params.get("voice"),
            model=voice_params.get("model"),
            instructions=voice_params.get("instructions"),
            audio_format=voice_params.get("audio_format"),
            initial_provider=voice_params.get("tts_provider"),
            speed=voice_params.get("speed"),
        )

        # Record the user's follow-up
        logger.info("🎤 Recording wake word follow-up...")
        loop = asyncio.get_event_loop()
        audio_data, speech_detected = await loop.run_in_executor(
            None,
            lambda: record_audio_with_silence_detection(
                max_duration=DEFAULT_LISTEN_DURATION,
                min_duration=MIN_RECORDING_DURATION,
            ),
        )

        if not speech_detected or len(audio_data) == 0:
            logger.info("Wake word: no follow-up speech detected")
            return

        # Transcribe the follow-up
        stt_result = await speech_to_text(
            audio_data,
            save_audio=SAVE_AUDIO,
            audio_dir=AUDIO_DIR if SAVE_AUDIO else None,
            transport="local",
        )

        if isinstance(stt_result, dict) and stt_result.get("text"):
            transcript = stt_result["text"]
            logger.info(f"🔔 Wake word follow-up captured: '{transcript}'")
            _pending_interruption = Interruption(
                transcript=transcript,
                detected_at=time.time(),
            )
        else:
            logger.info("Wake word: could not transcribe follow-up")
