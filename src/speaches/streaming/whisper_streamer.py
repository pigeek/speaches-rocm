"""WhisperStreamer: protocol-agnostic streaming transcription orchestrator.

Manages audio buffering, incremental transcription via faster-whisper, and
the LocalAgreement2 algorithm for stable word commitment. Designed to be
called from any transport (HTTP SSE, WebSocket, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from speaches.streaming.hypothesis_buffer import HypothesisBuffer, Word
from speaches.streaming.vad import SileroVADDetector

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

_logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


class WhisperStreamer:
    """Audio buffer + LocalAgreement2 streaming orchestrator.

    Usage::

        streamer = WhisperStreamer(model, language="en")
        for chunk in audio_chunks:
            streamer.push_audio(chunk)
            if streamer.should_transcribe():
                delta = streamer.transcribe_and_commit()
                if delta:
                    yield delta  # send to client
        final = streamer.finalize()
        if final:
            yield final

    Args:
        model: A loaded faster_whisper.WhisperModel instance.
        language: Language code (e.g. "en") or None for auto-detect.
        min_chunk_sec: Minimum audio duration before first transcription.
        buffer_trimming_sec: Trim audio buffer when it exceeds this duration.
        vad_enabled: Whether to use VAD for early transcription trigger.
        vad_threshold: Speech probability threshold for VAD.
        vad_min_silence_ms: Silence duration (ms) to trigger transcription.
    """

    def __init__(
        self,
        model: WhisperModel,
        *,
        language: str | None = None,
        min_chunk_sec: float = 1.0,
        buffer_trimming_sec: float = 15.0,
        vad_enabled: bool = True,
        vad_threshold: float = 0.5,
        vad_min_silence_ms: int = 500,
    ) -> None:
        self._model = model
        self._language = language

        self._min_chunk_sec = min_chunk_sec
        self._buffer_trimming_sec = buffer_trimming_sec
        self._vad_enabled = vad_enabled

        # Audio buffer
        self._audio_buffer = np.array([], dtype=np.float32)
        self._buffer_time_offset: float = 0.0
        self._samples_since_last_transcribe: int = 0

        # LocalAgreement2 state
        self._hypothesis_buffer = HypothesisBuffer()
        self._all_committed: list[Word] = []
        self._committed_text: str = ""

        # VAD
        self._vad: SileroVADDetector | None = None
        self._silence_triggered: bool = False
        if vad_enabled:
            try:
                self._vad = SileroVADDetector(
                    threshold=vad_threshold,
                    min_silence_ms=vad_min_silence_ms,
                )
            except Exception:
                _logger.warning("Failed to initialize VAD, disabling", exc_info=True)
                self._vad_enabled = False

    @property
    def committed_text(self) -> str:
        """Full committed text so far."""
        return self._committed_text

    @property
    def silence_triggered(self) -> bool:
        """Whether VAD has detected silence since last transcription."""
        return self._silence_triggered

    def push_audio(self, pcm_bytes: bytes) -> None:
        """Accumulate PCM16 audio data.

        Args:
            pcm_bytes: Raw 16-bit signed little-endian PCM at 16 kHz mono.
        """
        # Ensure even number of bytes for int16 parsing
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]
        if not pcm_bytes:
            return
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_buffer = np.append(self._audio_buffer, samples)
        self._samples_since_last_transcribe += len(samples)

        # VAD processing
        if self._vad and len(samples) >= 512:
            try:
                _is_speech, silence = self._vad.process_chunk(samples)
                if silence:
                    self._silence_triggered = True
            except Exception:
                _logger.warning("VAD error", exc_info=True)

    def push_audio_float32(self, audio: np.ndarray) -> None:
        """Accumulate float32 audio data (already normalized to [-1, 1]).

        Args:
            audio: Float32 numpy array of audio samples at 16 kHz mono.
        """
        self._audio_buffer = np.append(self._audio_buffer, audio)
        self._samples_since_last_transcribe += len(audio)

        if self._vad and len(audio) >= 512:
            try:
                _is_speech, silence = self._vad.process_chunk(audio)
                if silence:
                    self._silence_triggered = True
            except Exception:
                _logger.warning("VAD error", exc_info=True)

    def should_transcribe(self) -> bool:
        """Check if enough audio has accumulated or silence was detected."""
        min_samples = int(self._min_chunk_sec * SAMPLING_RATE)
        return self._samples_since_last_transcribe >= min_samples or self._silence_triggered

    def transcribe_and_commit(self) -> str:
        """Run transcription on current buffer, return newly committed text.

        Returns:
            Newly committed text delta (may be empty string if nothing committed yet).
        """
        self._samples_since_last_transcribe = 0
        self._silence_triggered = False

        if len(self._audio_buffer) < 100:
            return ""

        words = self._transcribe_buffer()
        if not words:
            return ""

        # Insert into hypothesis buffer (words have timestamps relative to buffer start,
        # offset makes them absolute)
        self._hypothesis_buffer.insert(words, self._buffer_time_offset)

        # Flush committed words via LCP
        committed = self._hypothesis_buffer.flush()
        if not committed:
            return ""

        self._all_committed.extend(committed)
        delta = "".join(w.text for w in committed)
        self._committed_text += delta

        _logger.info(
            "Committed: '%s' | buffer: %.1fs",
            delta.strip()[:40],
            len(self._audio_buffer) / SAMPLING_RATE,
        )

        # Trim buffer if needed
        buffer_duration = len(self._audio_buffer) / SAMPLING_RATE
        if buffer_duration > self._buffer_trimming_sec:
            self._trim_buffer(committed)

        return delta

    def finalize(self) -> str:
        """Get remaining uncommitted words at end of stream.

        Returns:
            Any remaining text that wasn't committed during streaming.
        """
        remaining = self._hypothesis_buffer.complete()
        _logger.info(
            "Finalize: hypothesis_buffer.complete() returned %d words, committed_text='%s', buffer=%.1fs",
            len(remaining),
            self._committed_text[:40] if self._committed_text else "",
            len(self._audio_buffer) / SAMPLING_RATE,
        )

        # If nothing was committed AND hypothesis buffer is empty, do a single transcription
        # (e.g. very short audio where LA2 never ran, or only ran once so complete() is empty)
        if not self._committed_text and not remaining and len(self._audio_buffer) > 0:
            _logger.info("Finalize: no committed text and no remaining words, running fresh transcription")
            words = self._transcribe_buffer()
            if words:
                remaining = words

        text = ""
        for w in remaining:
            t = w.text.strip()
            if t:
                text += w.text

        if text:
            self._committed_text += text
        _logger.info("Finalize result: '%s'", text[:80] if text else "")
        return text

    def _prompt(self) -> str:
        """Get prompt text (committed words outside the audio buffer) for context."""
        k = len(self._all_committed)
        while k > 0 and self._all_committed[k - 1].end > self._buffer_time_offset:
            k -= 1

        outside_buffer = self._all_committed[:k]

        parts: list[str] = []
        char_count = 0
        for word in reversed(outside_buffer):
            if char_count >= 200:
                break
            parts.append(word.text)
            char_count += len(word.text) + 1

        return "".join(reversed(parts))

    def _transcribe_buffer(self) -> list[Word]:
        """Transcribe current audio buffer and return word list."""
        prompt = self._prompt()
        audio_duration = len(self._audio_buffer) / SAMPLING_RATE

        try:
            segments_generator, info = self._model.transcribe(
                self._audio_buffer,
                language=self._language,
                initial_prompt=prompt if prompt else None,
                word_timestamps=True,
                condition_on_previous_text=True,
            )

            words: list[Word] = []
            segment_count = 0
            for segment in segments_generator:
                segment_count += 1
                _logger.debug(
                    "Segment %d: no_speech_prob=%.3f, text='%s', words=%d",
                    segment_count,
                    segment.no_speech_prob,
                    segment.text[:50] if segment.text else "",
                    len(segment.words) if hasattr(segment, "words") and segment.words else 0,
                )
                if segment.no_speech_prob > 0.9:
                    continue
                if hasattr(segment, "words") and segment.words:
                    for w in segment.words:
                        words.append(Word(w.start, w.end, w.word))

            _logger.info(
                "Transcribed %.1fs audio: %d segments, %d words, lang=%s",
                audio_duration, segment_count, len(words), info.language,
            )
            return words
        except Exception:
            _logger.error("Transcription error", exc_info=True)
            return []

    def _trim_buffer(self, committed: list[Word]) -> None:
        """Trim audio buffer after committed words."""
        if not committed:
            return

        trim_time = committed[-1].end
        trim_duration = trim_time - self._buffer_time_offset
        if trim_duration <= 0:
            return

        trim_samples = int(trim_duration * SAMPLING_RATE)
        if trim_samples >= len(self._audio_buffer):
            return

        old_dur = len(self._audio_buffer) / SAMPLING_RATE
        self._audio_buffer = self._audio_buffer[trim_samples:]
        self._buffer_time_offset = trim_time
        self._hypothesis_buffer.pop_commited(trim_time)
        new_dur = len(self._audio_buffer) / SAMPLING_RATE

        _logger.info("Buffer trimmed: %.1fs -> %.1fs (offset: %.1fs)", old_dur, new_dur, self._buffer_time_offset)
