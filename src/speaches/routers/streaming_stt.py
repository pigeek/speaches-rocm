"""WebSocket endpoint for live streaming transcription with LocalAgreement2.

Faithful port of the faster-whisper-ws handler_v2.py architecture:
- Receive loop starts immediately, buffering audio during model load
- Background streaming loop waits on events (new_audio / silence / stop)
- asyncio.wait(FIRST_COMPLETED) for zero-latency wakeup
- Stop signals the streaming loop to exit after current transcription finishes
- No task cancellation — in-flight transcription results are preserved

Protocol:
  Client sends:
    - JSON message: {"type": "config", "model": "...", "language": "en"}  (optional, first message)
    - Binary messages: raw 16-bit signed LE PCM at 16 kHz mono
    - JSON message: {"type": "stop"}  (end of audio)

  Server sends:
    - JSON: {"type": "transcript", "text": "...", "is_final": false}  (incremental)
    - JSON: {"type": "transcript", "text": "...", "is_final": true}   (final)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# CRITICAL: These MUST NOT be in TYPE_CHECKING — FastAPI needs them at runtime
# for dependency injection. ruff TC001 will try to move them; add noqa if needed.
from speaches.dependencies import ConfigDependency, ExecutorRegistryDependency  # noqa: TC001
from speaches.streaming import WhisperStreamer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streaming-stt"])

DEFAULT_MODEL = "Systran/faster-whisper-large-v3"


@router.websocket("/v1/audio/transcriptions/stream")
async def streaming_transcription(
    ws: WebSocket,
    executor_registry: ExecutorRegistryDependency,
    config: ConfigDependency,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
) -> None:
    await ws.accept()
    logger.info("Streaming STT WebSocket connected (model=%s, language=%s)", model, language)

    loop = asyncio.get_event_loop()

    # Shared state
    stop_event = asyncio.Event()
    disconnected = asyncio.Event()
    new_audio_event = asyncio.Event()
    silence_detected_event = asyncio.Event()
    streamer_ref: list[WhisperStreamer | None] = [None]
    audio_buffer: list[bytes] = []  # Buffer audio while model loads

    async def _receive_loop() -> None:
        """Receive messages immediately. Buffers audio until streamer is ready."""
        try:
            while not stop_event.is_set():
                message = await ws.receive()

                if message.get("type") == "websocket.disconnect":
                    disconnected.set()
                    stop_event.set()
                    return

                if message.get("bytes"):
                    s = streamer_ref[0]
                    if s is not None:
                        s.push_audio(message["bytes"])
                        if s.silence_triggered:
                            silence_detected_event.set()
                        new_audio_event.set()
                    else:
                        # Model still loading — buffer raw audio
                        audio_buffer.append(message["bytes"])

                elif message.get("text"):
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue

                    if data.get("type") == "stop":
                        logger.info("Received stop message")
                        stop_event.set()
                        return

                    if data.get("type") == "config":
                        logger.info("Received config: %s", data)

        except WebSocketDisconnect:
            disconnected.set()
            stop_event.set()
        except Exception:  # noqa: BLE001
            logger.warning("Receiver error", exc_info=True)
            stop_event.set()

    # Start receiving IMMEDIATELY — audio arrives during model load
    recv_task = asyncio.create_task(_receive_loop())

    # Load model concurrently with audio reception
    whisper_executor = executor_registry._whisper_executor  # noqa: SLF001
    model_ctx = await loop.run_in_executor(None, whisper_executor.model_manager.load_model, model)
    whisper_model = await loop.run_in_executor(None, model_ctx.__enter__)

    try:
        # Create streamer
        streamer = await loop.run_in_executor(
            None,
            lambda: WhisperStreamer(
                whisper_model,
                language=language,
                min_chunk_sec=config.streaming_min_chunk_sec,
                buffer_trimming_sec=config.streaming_buffer_trimming_sec,
                vad_enabled=config.streaming_vad_enabled,
                vad_threshold=config.streaming_vad_threshold,
                vad_min_silence_ms=config.streaming_vad_min_silence_ms,
            ),
        )

        # Feed buffered audio that arrived during model load
        if audio_buffer:
            logger.info("Feeding %d buffered audio chunks to streamer", len(audio_buffer))
            for chunk in audio_buffer:
                streamer.push_audio(chunk)
            audio_buffer.clear()
            if streamer.should_transcribe():
                new_audio_event.set()

        streamer_ref[0] = streamer
        logger.info("Model and streamer ready")

        # If stop already arrived during model load, skip streaming and finalize
        if stop_event.is_set():
            logger.info("Stop received during model load, finalizing directly")
            if not disconnected.is_set():
                final = await loop.run_in_executor(None, streamer.finalize)
                full_text = streamer.committed_text
                logger.info("Finalize: full_text='%s'", full_text[:80] if full_text else "")
                await ws.send_json(
                    {
                        "type": "transcript",
                        "text": final if final else "",
                        "is_final": True,
                        "full_text": full_text,
                    }
                )
            return

        # Background streaming loop — waits on events including stop_event
        # When stop fires, the loop exits AFTER the current transcription finishes
        # (no cancellation, so in-flight results are preserved)
        async def _streaming_loop() -> None:
            try:
                while not stop_event.is_set():
                    _done, pending = await asyncio.wait(
                        [
                            asyncio.create_task(new_audio_event.wait()),
                            asyncio.create_task(silence_detected_event.wait()),
                            asyncio.create_task(stop_event.wait()),
                        ],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()

                    new_audio_event.clear()
                    silence_detected_event.clear()

                    if stop_event.is_set():
                        break

                    if not streamer.should_transcribe():
                        continue

                    # This runs in executor — if stop arrives during transcription,
                    # the transcription completes naturally, result is preserved,
                    # then loop checks stop_event on next iteration and exits
                    delta = await loop.run_in_executor(None, streamer.transcribe_and_commit)
                    if delta:
                        await ws.send_json(
                            {
                                "type": "transcript",
                                "text": delta,
                                "is_final": False,
                            }
                        )

            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Error in streaming loop")

        streaming_task = asyncio.create_task(_streaming_loop())

        # Wait for streaming loop to finish (exits when stop_event is set)
        await streaming_task

        # Finalize: get remaining uncommitted words
        if not disconnected.is_set():
            final = await loop.run_in_executor(None, streamer.finalize)
            full_text = streamer.committed_text
            logger.info("Finalize: full_text='%s'", full_text[:80] if full_text else "")
            await ws.send_json(
                {
                    "type": "transcript",
                    "text": final if final else "",
                    "is_final": True,
                    "full_text": full_text,
                }
            )

    except WebSocketDisconnect:
        logger.info("Streaming STT WebSocket disconnected")
    except Exception:
        logger.exception("Streaming STT error")
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": "Internal server error"})
    finally:
        stop_event.set()
        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await recv_task
        model_ctx.__exit__(None, None, None)
        logger.info("Streaming STT WebSocket closed")
