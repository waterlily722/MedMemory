"""Tests for tokenization and cosine similarity (including CJK support)."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PACKAGE = _HERE.parent
if str(_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_PACKAGE))

from memory_agent.utils.scoring import tokenize, cosine_similarity, flatten_payload


# ═══════════════════════════════════════════════════════════════════════
# Tokenize
# ═══════════════════════════════════════════════════════════════════════

def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_tokenize_english():
    tokens = tokenize("chest pain with cough")
    assert tokens == ["chest", "pain", "with", "cough"]


def test_tokenize_english_mixed_case():
    tokens = tokenize("Chest Pain WITH COUGH")
    assert tokens == ["chest", "pain", "with", "cough"]


def test_tokenize_with_numbers():
    tokens = tokenize("patient has O2_sat 95%")
    assert "o2_sat" in tokens
    assert "95" in tokens


def test_tokenize_cjk():
    """Each CJK character becomes its own token."""
    tokens = tokenize("胸痛伴随咳嗽")
    assert tokens == ["胸", "痛", "伴", "随", "咳", "嗽"]


def test_tokenize_mixed():
    tokens = tokenize("patient 胸痛 cough fever 发热")
    assert "patient" in tokens
    assert "胸" in tokens
    assert "痛" in tokens
    assert "cough" in tokens
    assert "fever" in tokens
    assert "发" in tokens
    assert "热" in tokens


def test_tokenize_punctuation_discarded():
    tokens = tokenize("chest pain, fever? yes!")
    assert "pain" in tokens
    assert "fever" in tokens
    assert "yes" in tokens
    # Punctuation should not appear as tokens
    assert "," not in tokens
    assert "?" not in tokens


# ═══════════════════════════════════════════════════════════════════════
# Cosine similarity
# ═══════════════════════════════════════════════════════════════════════

def test_cosine_identical():
    score = cosine_similarity("chest pain", "chest pain")
    assert abs(score - 1.0) < 1e-6, f"Expected ~1.0, got {score}"


def test_cosine_unrelated():
    score = cosine_similarity("chest pain", "abdominal pain")
    assert 0.0 < score < 1.0


def test_cosine_empty():
    assert cosine_similarity("", "chest pain") == 0.0
    assert cosine_similarity("chest pain", "") == 0.0
    assert cosine_similarity("", "") == 0.0


def test_cosine_cjk_similar():
    """Chinese texts with overlapping characters score higher."""
    hi = cosine_similarity("胸痛伴随咳嗽", "胸痛发烧咳嗽")
    lo = cosine_similarity("胸痛伴随咳嗽", "腹痛腹泻")
    assert hi > lo


def test_cosine_cjk_mixed():
    """Mixed Chinese/English similarity works."""
    score = cosine_similarity("patient 胸痛", "patient 腹痛")
    assert 0.0 < score < 1.0


# ═══════════════════════════════════════════════════════════════════════
# Flatten payload
# ═══════════════════════════════════════════════════════════════════════

def test_flatten_dict():
    result = flatten_payload({"a": "hello", "b": "world"})
    assert "hello" in result
    assert "world" in result


def test_flatten_list():
    result = flatten_payload([{"action": "ask"}, {"action": "test"}])
    assert "ask" in result
    assert "test" in result


def test_flatten_none():
    assert flatten_payload(None) == ""
