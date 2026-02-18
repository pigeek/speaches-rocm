"""LocalAgreement2 streaming transcription module.

Ported from ufal/whisper_streaming via faster-whisper-ws.
Provides incremental speech-to-text with committed word stability.
"""

from speaches.streaming.hypothesis_buffer import HypothesisBuffer, Word
from speaches.streaming.vad import SileroVADDetector
from speaches.streaming.whisper_streamer import WhisperStreamer

__all__ = ["HypothesisBuffer", "SileroVADDetector", "WhisperStreamer", "Word"]
