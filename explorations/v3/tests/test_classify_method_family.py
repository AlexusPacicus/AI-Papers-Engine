from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import classify_method_family as module  # noqa: E402


def test_gan_not_matched_inside_other_words():
    family, signals = module.classify_text("a study of organization in teams")
    assert family == "unknown"
    assert not any(signal.startswith("gan:") for signal in signals)


def test_gan_still_matched_as_whole_word():
    family, _ = module.classify_text("we train a gan on faces")
    assert family == "gan"


def test_diffusion_transformer_is_not_a_tie_with_diffusion():
    family, _ = module.classify_text("a diffusion transformer for images")
    assert family == "diffusion_transformer_or_hybrid"
