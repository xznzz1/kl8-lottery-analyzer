"""动态 Beta-Bernoulli 模型测试。"""

from __future__ import annotations

import numpy as np

from src.research_v2.bayesian import (
    FAIR_PROBABILITY,
    BayesianParameters,
    candidate_sets,
    discounted_beta_bernoulli,
    rank_probabilities,
)


def _persistent_high_frequency_history(periods: int = 400) -> np.ndarray:
    history = np.zeros((periods, 80), dtype=np.float64)
    history[:, 0] = 1.0
    for index in range(periods):
        rotating = 1 + (np.arange(19) + index) % 78
        history[index, rotating] = 1.0
    return history


def test_prior_only_fair_history_returns_exact_quarter_probability() -> None:
    prediction = discounted_beta_bernoulli(
        np.empty((0, 80), dtype=np.float64),
        BayesianParameters(decay=0.99, prior_strength=20.0),
    )

    assert np.allclose(prediction.posterior_mean, FAIR_PROBABILITY)
    assert np.isclose(prediction.posterior_mean.sum(), 20.0)


def test_probabilities_intervals_and_variances_are_legal() -> None:
    prediction = discounted_beta_bernoulli(
        _persistent_high_frequency_history(),
        BayesianParameters(decay=0.99, prior_strength=20.0),
    )

    assert np.all((prediction.posterior_mean > 0.0) & (prediction.posterior_mean < 1.0))
    assert np.isclose(prediction.posterior_mean.sum(), 20.0, atol=1e-9)
    assert np.all(prediction.posterior_variance > 0.0)
    assert np.all(prediction.credible_interval_lower < prediction.posterior_mean)
    assert np.all(prediction.posterior_mean < prediction.credible_interval_upper)


def test_long_term_high_frequency_number_has_higher_posterior_mean() -> None:
    prediction = discounted_beta_bernoulli(
        _persistent_high_frequency_history(),
        BayesianParameters(decay=0.995, prior_strength=20.0),
    )

    assert prediction.posterior_mean[0] > prediction.posterior_mean[79]


def test_stronger_prior_pulls_posterior_toward_fair_probability() -> None:
    history = _persistent_high_frequency_history()
    weak = discounted_beta_bernoulli(
        history, BayesianParameters(decay=0.99, prior_strength=5.0)
    )
    strong = discounted_beta_bernoulli(
        history, BayesianParameters(decay=0.99, prior_strength=80.0)
    )

    weak_distance = abs(weak.posterior_mean[0] - FAIR_PROBABILITY)
    strong_distance = abs(strong.posterior_mean[0] - FAIR_PROBABILITY)
    assert strong_distance < weak_distance


def test_smaller_decay_gives_more_weight_to_recent_history() -> None:
    history = np.zeros((400, 80), dtype=np.float64)
    history[:, 2:21] = 1.0
    history[:200, 0] = 1.0
    history[200:, 1] = 1.0

    faster = discounted_beta_bernoulli(
        history, BayesianParameters(decay=0.97, prior_strength=20.0)
    )
    slower = discounted_beta_bernoulli(
        history, BayesianParameters(decay=0.995, prior_strength=20.0)
    )

    faster_recent_gap = faster.posterior_mean[1] - faster.posterior_mean[0]
    slower_recent_gap = slower.posterior_mean[1] - slower.posterior_mean[0]
    assert faster_recent_gap > slower_recent_gap > 0.0


def test_extreme_history_never_produces_nan_or_infinity() -> None:
    history = np.zeros((5000, 80), dtype=np.float64)
    history[:, :20] = 1.0
    prediction = discounted_beta_bernoulli(
        history, BayesianParameters(decay=0.995, prior_strength=5.0)
    )

    for values in (
        prediction.alpha,
        prediction.beta,
        prediction.posterior_mean,
        prediction.posterior_variance,
        prediction.credible_interval_lower,
        prediction.credible_interval_upper,
    ):
        assert np.isfinite(values).all()


def test_candidate_ties_are_resolved_by_number_ascending() -> None:
    candidates = candidate_sets(np.full(80, FAIR_PROBABILITY, dtype=np.float64))

    assert candidates[1] == (1,)
    assert candidates[10] == tuple(range(1, 11))


def _manual_exponential_frequency_ranking(
    history: np.ndarray, decay: float
) -> np.ndarray:
    ages = np.arange(len(history) - 1, -1, -1, dtype=np.float64)
    weights = np.power(decay, ages)
    scores = weights @ history / weights.sum()
    numbers = np.arange(1, 81, dtype=np.int64)
    return numbers[np.lexsort((numbers, -scores))]


def test_dynamic_ranking_equals_manual_exponential_frequency_at_same_decay() -> None:
    history = _persistent_high_frequency_history()
    parameters = BayesianParameters(decay=0.99, prior_strength=20.0)
    prediction = discounted_beta_bernoulli(history, parameters)

    assert np.array_equal(
        rank_probabilities(prediction.posterior_mean),
        _manual_exponential_frequency_ranking(history, parameters.decay),
    )


def test_prior_strength_does_not_change_ranking_at_same_decay() -> None:
    history = _persistent_high_frequency_history()
    rankings = []
    for prior_strength in (5.0, 20.0, 80.0):
        prediction = discounted_beta_bernoulli(
            history,
            BayesianParameters(decay=0.995, prior_strength=prior_strength),
        )
        rankings.append(rank_probabilities(prediction.posterior_mean))

    assert np.array_equal(rankings[0], rankings[1])
    assert np.array_equal(rankings[1], rankings[2])


def test_prior_strength_changes_probability_distance_from_quarter() -> None:
    history = _persistent_high_frequency_history()
    distances = []
    for prior_strength in (5.0, 20.0, 80.0):
        prediction = discounted_beta_bernoulli(
            history,
            BayesianParameters(decay=0.99, prior_strength=prior_strength),
        )
        distances.append(
            float(np.linalg.norm(prediction.posterior_mean - FAIR_PROBABILITY))
        )

    assert distances[0] > distances[1] > distances[2] > 0.0
