"""Rank coverage and policy-weighted hardness must use every tokenizer."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pool_policy_ranking import average_ranks, compare_policy


def test_a_concentrated_policy_can_select_the_true_hardest_or_the_easiest():
    gaps = np.linspace(-.1, .2, 1000)
    for selected, rank in [(999, 1), (0, 1000)]:
        p = np.zeros(1000);p[selected] = 1
        result = compare_policy(gaps, p)
        assert result['dominant_rank'] == rank
        assert result['policy_expected_gap'] == gaps[selected]
        assert result['oracle_minus_policy'] == pytest.approx(.2 - gaps[selected])
        assert result['probability_on_hardest']['10'] == (1 if selected == 999 else 0)


def test_uniform_policy_and_ties_are_handled_without_invented_correlation():
    result = compare_policy(np.ones(1000) * .1, np.ones(1000) / 1000)
    assert result['policy_minus_uniform'] == pytest.approx(0)
    assert result['probability_gap_spearman'] is None
    assert average_ranks([3, 1, 1]).tolist() == [3, 1.5, 1.5]


def test_missing_tokenizers_or_invalid_probabilities_cannot_be_silently_excluded():
    with pytest.raises(ValueError):
        compare_policy(np.ones(999), np.ones(999) / 999)
    for invalid in [np.full(1000, np.nan), np.zeros(1000), np.full(1000, -.001)]:
        with pytest.raises(ValueError):
            compare_policy(np.ones(1000), invalid)
