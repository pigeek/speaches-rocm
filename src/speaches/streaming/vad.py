"""Silero VAD wrapper for streaming silence detection.

Wraps the Silero VAD model to detect speech/silence boundaries in audio streams,
used to trigger early transcription when the speaker pauses.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

_logger = logging.getLogger(__name__)


class SileroVADDetector:
    """Silero VAD wrapper for silence detection in streaming audio.

    Tracks speech state and silence duration to trigger end-of-speech events.
    """

    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 500) -> None:
        """Initialize Silero VAD.

        Args:
            threshold: Speech probability threshold (0-1).
            min_silence_ms: Minimum silence duration (ms) to trigger end of speech.
        """
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()
        self._reset_states()

    def _reset_states(self) -> None:
        self.model.reset_states()
        self._silence_samples = 0
        self._is_speech = False

    def reset(self) -> None:
        """Reset VAD for a new session."""
        self._reset_states()

    def process_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[bool, bool]:
        """Process audio chunk and detect speech/silence.

        Args:
            audio: Audio samples (float32, -1 to 1).
            sample_rate: Must be 16000.

        Returns:
            (is_speech, silence_triggered):
            - is_speech: True if current chunk contains speech.
            - silence_triggered: True if silence threshold exceeded after speech.
        """
        if sample_rate != 16000:
            raise ValueError("Silero VAD requires 16kHz audio")

        audio_tensor = torch.from_numpy(audio).float()

        with torch.no_grad():
            speech_prob = self.model(audio_tensor, sample_rate).item()

        is_speech = speech_prob >= self.threshold

        if is_speech:
            self._silence_samples = 0
            self._is_speech = True
        else:
            self._silence_samples += len(audio)

        silence_ms = (self._silence_samples / sample_rate) * 1000
        silence_triggered = self._is_speech and silence_ms >= self.min_silence_ms

        return is_speech, silence_triggered
