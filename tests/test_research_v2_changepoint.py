"""第二阶段变点模型的公式、状态与排序测试。"""

from __future__ import annotations

import numpy as np

from src.research_v2.bayesian import rank_probabilities
from src.research_v2.changepoint import (
    HIGH_CHANGE_DECAY,
    NORMAL_DECAY,
    RECENT_WINDOW_GRID,
    ChangepointParameters,
    changepoint_beta_bernoulli,
    frequency_change_score,
    precompute_change_scores,
    preregistered_changepoint_parameters,
)


def _rotating_indicators(periods: int) -> np.ndarray:
    indicators = np.zeros((periods, 80), dtype=np.float64)
    for period in range(periods):
        numbers = (np.arange(20, dtype=np.int64) * 7 + period * 3) % 80
        indicators[period, numbers] = 1.0
    return indicators


def _random_indicators(periods: int, seed: int = 8128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indicators = np.zeros((periods, 80), dtype=np.float64)
    for period in range(periods):
        indicators[period, rng.choice(80, size=20, replace=False)] = 1.0
    return indicators


def test_preregistered_configuration_is_exact_and_cannot_expand() -> None:
    parameters = preregistered_changepoint_parameters()
    assert tuple(item.recent_window for item in parameters) == RECENT_WINDOW_GRID
    assert all(item.reference_window == 240 for item in parameters)
    assert all(item.minimum_effective_history == 120 for item in parameters)
    assert all(item.change_quantile == 0.90 for item in parameters)
    assert all(item.high_change_decay == 0.95 for item in parameters)
    assert all(item.normal_decay == 0.995 for item in parameters)
    assert all(item.prior_strength == 80.0 for item in parameters)
    with np.testing.assert_raises(ValueError):
        ChangepointParameters(recent_window=45)
    with np.testing.assert_raises(ValueError):
        ChangepointParameters(recent_window=30, change_quantile=0.95)


def test_target_and_future_never_enter_change_score() -> None:
    indicators = _random_indicators(500)
    parameters = ChangepointParameters(60)
    target_index = 400
    expected = frequency_change_score(indicators[:target_index], parameters)
    changed = indicators.copy()
    changed[target_index:] = np.roll(changed[target_index:], shift=19, axis=1)
    actual = frequency_change_score(changed[:target_index], parameters)
    precomputed = precompute_change_scores(changed, parameters)
    assert actual == expected
    assert precomputed[target_index] == expected


def test_clear_change_triggers_high_decay() -> None:
    history = np.zeros((400, 80), dtype=np.float64)
    history[:340, :20] = 1.0
    history[340:, 60:] = 1.0
    parameters = ChangepointParameters(30)
    scores = precompute_change_scores(history, parameters)
    threshold = float(np.quantile(scores[270:340], 0.90))
    prediction = changepoint_beta_bernoulli(
        history,
        change_threshold=threshold,
        parameters=parameters,
    )
    assert prediction.high_change
    assert prediction.decay == HIGH_CHANGE_DECAY
    assert prediction.effective_history_length == 120


def test_stable_period_uses_normal_decay() -> None:
    history = np.tile(np.concatenate((np.ones(20), np.zeros(60))), (400, 1)).astype(
        np.float64
    )
    parameters = ChangepointParameters(30)
    prediction = changepoint_beta_bernoulli(
        history,
        change_threshold=0.0,
        parameters=parameters,
    )
    assert prediction.change_score == 0.0
    assert not prediction.high_change
    assert prediction.decay == NORMAL_DECAY
    assert prediction.effective_history_length == len(history)


def test_high_change_reduces_the_influence_of_old_data() -> None:
    first = _random_indicators(420)
    second = first.copy()
    second[:200] = np.roll(second[:200], shift=11, axis=1)
    parameters = ChangepointParameters(30)
    first_high = changepoint_beta_bernoulli(
        first, change_threshold=0.0, parameters=parameters
    )
    second_high = changepoint_beta_bernoulli(
        second, change_threshold=0.0, parameters=parameters
    )
    first_normal = changepoint_beta_bernoulli(
        first, change_threshold=1_000.0, parameters=parameters
    )
    second_normal = changepoint_beta_bernoulli(
        second, change_threshold=1_000.0, parameters=parameters
    )
    np.testing.assert_array_equal(first_high.posterior_mean, second_high.posterior_mean)
    assert not np.array_equal(first_normal.posterior_mean, second_normal.posterior_mean)


def test_probabilities_intervals_and_sum_are_legal_under_extreme_history() -> None:
    history = np.tile(np.concatenate((np.ones(20), np.zeros(60))), (500, 1)).astype(
        np.float64
    )
    prediction = changepoint_beta_bernoulli(
        history,
        change_threshold=1.0,
        parameters=ChangepointParameters(120),
    )
    assert np.isfinite(prediction.posterior_mean).all()
    assert np.isfinite(prediction.posterior_variance).all()
    assert np.all((prediction.posterior_mean > 0.0) & (prediction.posterior_mean < 1.0))
    np.testing.assert_allclose(prediction.posterior_mean.sum(), 20.0, atol=1e-9)
    assert np.all(prediction.credible_interval_lower < prediction.posterior_mean)
    assert np.all(prediction.posterior_mean < prediction.credible_interval_upper)


def test_ranking_is_reproducible_and_can_differ_from_fixed_exponential() -> None:
    history = _random_indicators(500)
    parameters = ChangepointParameters(30)
    first = changepoint_beta_bernoulli(
        history, change_threshold=0.0, parameters=parameters
    )
    second = changepoint_beta_bernoulli(
        history, change_threshold=0.0, parameters=parameters
    )
    np.testing.assert_array_equal(first.ranking, second.ranking)

    ages = np.arange(len(history) - 1, -1, -1, dtype=np.float64)
    fixed_scores = np.power(0.99, ages) @ history
    fixed_ranking = rank_probabilities(fixed_scores)
    assert not np.array_equal(first.ranking, fixed_ranking)
