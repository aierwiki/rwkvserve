"""
Incremental stop-string checker for RWKVServe.

Byte-safe incremental detokenization + OpenAI stop-string semantics:
- feeds tokens one at a time, keeps a pending byte buffer so multi-byte UTF-8
  characters that span tokens never pollute the decoded text;
- detects stop strings over the *completed* text only;
- on hit, reports how many generated tokens to keep (OpenAI spec: the output
  must not include the stop string) and the truncated text;
- reports the "safe" text length that may already be streamed to a client
  (completed length minus the longest stop string, so a partial stop is never
  streamed out).

Used by both execution paths (SchedulerCore sync loop and RWKVEngine async loop)
so stop behaves identically everywhere.
"""

from bisect import bisect_right
from typing import Dict, List, Optional, Tuple


class IncrementalStopChecker:
    """Feed generated token ids; query hit / safe-length / truncated text."""

    def __init__(self, idx2token: Dict[int, bytes], stops: List[str]):
        self.idx2token = idx2token
        self.stops = [s for s in (stops or []) if s]
        self.longest = max((len(s) for s in self.stops), default=0)
        self.tokens: List[int] = []
        self._pending: bytearray = bytearray()      # bytes not yet decodable
        self._pending_tokens: List[int] = []        # tokens owning _pending bytes
        self.text: str = ""                          # completed, clean text
        self.token_end_chars: List[int] = []         # completed-char count after each token
        self.hit: bool = False
        self.keep_tokens: Optional[int] = None       # generated_tokens[:keep] on hit
        self.keep_text: Optional[str] = None         # truncated text on hit

    # -- incremental detokenization ---------------------------------------

    def _flush_pending(self) -> None:
        """Move decodable leading bytes from pending into completed text.

        Handles multi-byte characters that span several byte-fallback tokens:
        tries the whole pending buffer first, then the longest decodable prefix.
        Bytes left in pending wait for more tokens.
        """
        while self._pending:
            data = bytes(self._pending)
            best, piece_str = 0, ""
            for L in range(len(data), 0, -1):
                try:
                    piece_str = data[:L].decode("utf-8")
                    best = L
                    break
                except UnicodeDecodeError:
                    continue
            if best == 0:
                break  # need more bytes
            self.text += piece_str
            end = len(self.text)
            consumed_bytes, consumed_tokens = 0, 0
            while consumed_bytes < best and self._pending_tokens:
                t = self._pending_tokens.pop(0)
                consumed_bytes += len(self.idx2token.get(t, b""))
                consumed_tokens += 1
                self.token_end_chars.append(end)
            del self._pending[:consumed_bytes]

    def feed(self, token_id: int) -> Tuple[bool, Optional[int]]:
        """Append a token. Returns (hit, keep_tokens). keep_tokens None unless hit."""
        self.tokens.append(token_id)
        piece = self.idx2token.get(token_id)
        if piece is None:  # unknown id: treat as empty completion
            self.token_end_chars.append(len(self.text))
            return self._check()
        self._pending += piece
        self._pending_tokens.append(token_id)
        self._flush_pending()
        return self._check()

    def _check(self) -> Tuple[bool, Optional[int]]:
        if not self.stops or self.hit:
            return self.hit, self.keep_tokens
        found = None
        for s in self.stops:
            idx = self.text.find(s)
            if idx != -1 and (found is None or idx + len(s) < found[0] + len(found[1])):
                found = (idx, s)
        if found is None:
            return False, None
        idx, s = found
        cut_char = idx  # text must end before the stop string
        # last token whose completed-char end <= cut_char
        keep = bisect_right(self.token_end_chars, cut_char)
        self.hit = True
        self.keep_tokens = keep
        self.keep_text = self.text[:cut_char]
        return True, keep

    # -- queries -----------------------------------------------------------

    @property
    def safe_text_len(self) -> int:
        """How much of `text` may already be streamed without leaking a partial stop."""
        if not self.stops:
            return len(self.text)
        return max(0, len(self.text) - self.longest + 1)

    def truncated_text(self, full_text: str) -> str:
        """Apply stop truncation to an arbitrary full text (e.g. client-assembled)."""
        if not self.stops or not self.hit:
            return full_text
        for s in self.stops:
            idx = full_text.find(s)
            if idx != -1:
                return full_text[:idx]
        return self.full_text_prefix(full_text)

    def full_text_prefix(self, full_text: str) -> str:
        return full_text[: min(len(full_text), len(self.text))]
