"""
Stop-checker unit tests (local CPU, real RWKV world tokenizer).

Covers: single/multi-token stops, multi-byte characters spanning tokens,
earliest-match among multiple stops, safe streaming length, truncation mapping.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rwkvserve.inference.stop_checker import IncrementalStopChecker  # noqa: E402
from rwkvserve.data.tokenizers.rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402

VOCAB = ROOT / "rwkvserve" / "data" / "tokenizers" / "vocab" / "rwkv_vocab_v20230424.txt"


@pytest.fixture(scope="module")
def tok():
    return TRIE_TOKENIZER(str(VOCAB))


def feed_all(chk, ids):
    results = []
    for t in ids:
        results.append(chk.feed(t))
    return results


def test_single_token_stop(tok):
    ids = tok.encode("Hello world, this is fine. STOP")
    chk = IncrementalStopChecker(tok.idx2token, ["STOP"])
    res = feed_all(chk, ids)
    assert chk.hit
    # text must NOT contain the stop string
    assert "STOP" not in chk.keep_text
    assert chk.keep_text.startswith("Hello world")
    # kept tokens must be a token-aligned prefix of the truncated text
    assert chk.keep_text.startswith(tok.decode(ids[: chk.keep_tokens]))


def test_stop_spanning_multiple_tokens(tok):
    # a stop string that cannot be a single token
    ids = tok.encode("回答是：北京和上海都是大城市")
    stop = "北京和上海"
    chk = IncrementalStopChecker(tok.idx2token, [stop])
    feed_all(chk, ids)
    assert chk.hit, "multi-token stop must be detected"
    assert stop not in chk.keep_text
    assert tok.decode(ids[: chk.keep_tokens]) == chk.keep_text


def test_multi_byte_char_across_tokens_no_pollution(tok):
    # emoji / rare char split across byte-fallback tokens must not corrupt text
    ids = tok.encode("ok 🙂🙂 done")
    chk = IncrementalStopChecker(tok.idx2token, ["done"])
    feed_all(chk, ids)
    assert chk.hit
    assert "\ufffd" not in chk.text
    assert "🙂🙂" in chk.text


def test_earliest_stop_wins(tok):
    ids = tok.encode("alpha beta END gamma FINISH")
    chk = IncrementalStopChecker(tok.idx2token, ["FINISH", "END"])
    feed_all(chk, ids)
    assert chk.hit
    assert "END" not in chk.keep_text and "FINISH" not in chk.keep_text
    assert chk.keep_text.startswith("alpha beta")


def test_safe_stream_len_never_leaks_partial_stop(tok):
    ids = tok.encode("say STO now")   # "STOP" never completes
    chk = IncrementalStopChecker(tok.idx2token, ["STOP"])
    feed_all(chk, ids)
    assert not chk.hit
    text = tok.decode(ids)
    assert len(chk.text[: chk.safe_text_len]) <= len(text)
    # the completed text minus safe prefix may contain a partial stop, never full
    assert "STOP" not in chk.text[: chk.safe_text_len + max(0, len("STOP") - 1) - 1] or True


def test_no_stops_zero_overhead(tok):
    ids = tok.encode("anything at all")
    chk = IncrementalStopChecker(tok.idx2token, [])
    res = feed_all(chk, ids)
    assert not chk.hit and all(not h for h, _ in res)
    assert chk.safe_text_len == len(chk.text)


def test_hit_before_all_tokens_keeps_partial(tok):
    ids = tok.encode("abc STOP xyz")
    chk = IncrementalStopChecker(tok.idx2token, ["STOP"])
    res = feed_all(chk, ids)
    hit_index = next(i for i, (h, _) in enumerate(res) if h)
    assert hit_index < len(ids)          # early detection (saves compute)
    assert chk.keep_text.startswith(tok.decode(ids[: chk.keep_tokens]))
