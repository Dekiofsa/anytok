"""Compare a saved policy with a complete measured 1,000-tokenizer ranking."""
import numpy as np


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def compare_policy(gaps, probabilities):
    gaps = np.asarray(gaps, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (gaps.shape != (1000,) or probabilities.shape != (1000,)
            or not np.isfinite(gaps).all() or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or abs(float(probabilities.sum()) - 1) > 1e-10):
        raise ValueError('Complete finite 1,000-tokenizer scores and saved probabilities required')
    hardest = np.argsort(-gaps, kind='stable')
    ranks = np.empty(1000, dtype=int); ranks[hardest] = np.arange(1, 1001)
    dominant = int(probabilities.argmax())
    expected = float(probabilities @ gaps)
    x, y = average_ranks(probabilities), average_ranks(gaps)
    correlation = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    return dict(dominant_index=dominant, dominant_probability=float(probabilities[dominant]),
        dominant_rank=int(ranks[dominant]), dominant_gap=float(gaps[dominant]),
        hardest_index=int(hardest[0]), hardest_gap=float(gaps[hardest[0]]),
        uniform_mean_gap=float(gaps.mean()), uniform_median_gap=float(np.median(gaps)),
        policy_expected_gap=expected, policy_minus_uniform=expected - float(gaps.mean()),
        oracle_minus_policy=float(gaps.max()) - expected,
        probability_gap_spearman=correlation,
        probability_on_hardest={str(k): float(probabilities[hardest[:k]].sum()) for k in [1, 10, 50, 100]},
        ranks=ranks.tolist())
