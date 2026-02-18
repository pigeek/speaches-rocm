"""WebSocket endpoint for live streaming transcription with LocalAgreement2.

Provides a WebSocket-based streaming STT endpoint that accepts live PCM audio
and returns incremental committed transcription text using the LocalAgreement2
algorithm for word-level stability.

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
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from speaches.dependencies import ConfigDependency, ExecutorRegistryDependency
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
    """WebSocket endpoint for live streaming transcription with LocalAgreement2.

    Query parameters:
        model: Whisper model ID (default: Systran/faster-whisper-large-v3)
        language: Language code (e.g. "en") or None for auto-detect

    After connection, send raw PCM16 audio as binary frames.
    Send {"type": "stop"} JSON to finalize and get remaining text.
    """
    await ws.accept()
    logger.info("Streaming STT WebSocket connected (model=%s, language=%s)", model, language)

    # Get the whisper executor and load the model
    whisper_executor = executor_registry._whisper_executor
    model_ctx = whisper_executor.model_manager.load_model(model)
    whisper_model = model_ctx.__enter__()

    try:
        streamer = WhisperStreamer(
            whisper_model,
            language=language,
            min_chunk_sec=config.streaming_min_chunk_sec,
            buffer_trimming_sec=config.streaming_buffer_trimming_sec,
            vad_enabled=config.streaming_vad_enabled,
            vad_threshold=config.streaming_vad_threshold,
            vad_min_silence_ms=config.streaming_vad_min_silence_ms,
        )

        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                # Binary frame: raw PCM16 audio
                pcm_data = message["bytes"]
                streamer.push_audio(pcm_data)

                if streamer.should_transcribe():
                    # Run transcription in executor to avoid blocking event loop
                    delta = await asyncio.get_event_loop().run_in_executor(
                        None, streamer.transcribe_and_commit
                    )
                    if delta:
                        await ws.send_json({
                            "type": "transcript",
                            "text": delta,
                            "is_final": False,
                        })

            elif "text" in message and message["text"]:
                # JSON frame: control message
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")

                if msg_type == "config":
                    # Allow reconfiguring mid-stream (mainly for first message)
                    logger.info("Received config: %s", data)

                elif msg_type == "stop":
                    # Finalize transcription
                    final = await asyncio.get_event_loop().run_in_executor(
                        None, streamer.finalize
                    )
                    full_text = streamer.committed_text
                    await ws.send_json({
                        "type": "transcript",
                        "text": final if final else "",
                        "is_final": True,
                        "full_text": full_text,
                    })
                    break

    except WebSocketDisconnect:
        logger.info("Streaming STT WebSocket disconnected")
    except Exception:
        logger.error("Streaming STT error", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": "Internal server error"})
        except Exception:
            pass
    finally:
        model_ctx.__exit__(None, None, None)
        logger.info("Streaming STT WebSocket closed")
