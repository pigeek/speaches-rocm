"""LocalAgreement2 hypothesis buffer for streaming transcription.

Ported from ufal/whisper_streaming. Maintains two consecutive transcription
hypotheses and commits words that appear in the longest common prefix (LCP)
of both, ensuring stability of emitted text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


@dataclass
class Word:
    """A word with absolute timestamps (relative to original audio start)."""

    start: float
    end: float
    text: str


class HypothesisBuffer:
    """Buffer for managing streaming transcription hypotheses.

    Maintains three lists:
    - commited_in_buffer: words committed and still in audio buffer
    - buffer: the previous transcription (for LCP comparison)
    - new: the new transcription

    All timestamps are ABSOLUTE (relative to original audio start).
    """

    def __init__(self) -> None:
        self.commited_in_buffer: list[Word] = []
        self.buffer: list[Word] = []
        self.new: list[Word] = []

        self.last_commited_time: float = 0.0
        self.last_commited_word: str = ""

    def insert(self, new_words: list[Word], offset: float = 0.0) -> None:
        """Insert new transcription, adjusting timestamps by offset and filtering.

        Args:
            new_words: Words from current transcription (timestamps relative to buffer start).
            offset: buffer_time_offset - add to timestamps to make them absolute.
        """
        # Adjust timestamps by offset to make them absolute
        new_words = [Word(w.start + offset, w.end + offset, w.text) for w in new_words]

        # Filter to only words that start after last committed time (with small tolerance)
        self.new = [w for w in new_words if w.start > self.last_commited_time - 0.1]

        # Remove n-gram overlaps with committed text
        if len(self.new) >= 1 and abs(self.new[0].start - self.last_commited_time) < 1:
            if self.commited_in_buffer:
                cn = len(self.commited_in_buffer)
                nn = len(self.new)
                for i in range(1, min(min(cn, nn), 5) + 1):
                    c = " ".join([self.commited_in_buffer[-j].text for j in range(1, i + 1)][::-1])
                    tail = " ".join([self.new[j - 1].text for j in range(1, i + 1)])
                    if c == tail:
                        words_removed = []
                        for _ in range(i):
                            words_removed.append(self.new.pop(0).text)
                        _logger.debug("Removed %d overlapping words: %s", i, words_removed)
                        break

    def flush(self) -> list[Word]:
        """Find and return committed words (LCP between buffer and new).

        Returns the longest common prefix of the last two inserts.
        """
        commit: list[Word] = []
        while self.new:
            if len(self.buffer) == 0:
                break

            if self.new[0].text == self.buffer[0].text:
                w = self.new[0]
                commit.append(w)
                self.last_commited_word = w.text
                self.last_commited_time = w.end
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break

        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time: float) -> None:
        """Remove committed words that end before the given time."""
        while self.commited_in_buffer and self.commited_in_buffer[0].end <= time:
            self.commited_in_buffer.pop(0)

    def complete(self) -> list[Word]:
        """Return all remaining uncommitted words."""
        return self.buffer + self.new

    def clear(self) -> None:
        """Reset the buffer."""
        self.commited_in_buffer.clear()
        self.buffer.clear()
        self.new.clear()
        self.last_commited_time = 0.0
        self.last_commited_word = ""
