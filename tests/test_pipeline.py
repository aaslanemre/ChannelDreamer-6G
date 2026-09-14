"""End-to-end Phase-1 tests on synthetic data (no real DeepSense data required)."""

from __future__ import annotations

import numpy as np
import pytest

from channeldreamer.data import generate_synthetic, make_windows, optimal_beam, split_segments
from channeldreamer.data.deepsense import fill_nan_power, resolve_columns
from channeldreamer.envs import OfflineBeamEnv
from channeldreamer.eval import (
    STABLE,
    TRANSITION,
    compute_metrics,
    label_regimes,
    power_loss_db,
    regime_for_windows,
    top_k_accuracy,
)
from channeldreamer.models import MarkovBaseline, OracleBaseline, ReactiveBaseline


@pytest.fixture(scope="module")
def synth():
    return generate_synthetic(n_segments=8, segment_length=300, transition_prob=0.02, seed=123)


# ------------------------------------------------------------------ windowing
def test_windows_never_cross_segments():
    power = np.random.default_rng(0).random((20, 64))
    seg = np.array([0] * 7 + [1] * 5 + [2] * 8)  # segment 1 has only 5 steps
    w = make_windows(power, seg, history=4, horizon=2)
    # per segment: valid t in [H-1, len-k-1] -> len-H-k+1 windows (0 if negative)
    expected = sum(max(0, n - 4 - 2 + 1) for n in (7, 5, 8))
    assert len(w) == expected == 2 + 0 + 3
    for hist, tgt_i, last_i, s in zip(w.histories, w.target_index, w.last_index, w.segment_ids):
        idx = np.arange(last_i - 3, tgt_i + 1)
        assert np.all(seg[idx] == s)
        np.testing.assert_array_equal(hist, power[last_i - 3 : last_i + 1])
        assert tgt_i == last_i + 2
    np.testing.assert_array_equal(w.target_power, power[w.target_index])
    np.testing.assert_array_equal(w.target_beam, optimal_beam(power[w.target_index]))


def test_windows_shape_and_stride(synth):
    w = make_windows(synth.power, synth.segment_ids, history=8, horizon=1, stride=3)
    assert w.histories.shape[1:] == (8, 64)
    assert w.target_power.shape == (len(w), 64)
    assert np.all(np.diff(w.last_index)[np.diff(w.segment_ids) == 0] == 3)


def test_split_segments_disjoint(synth):
    tr, te = split_segments(synth.segment_ids, 0.25, seed=1)
    assert set(tr).isdisjoint(te)
    assert len(tr) + len(te) == len(np.unique(synth.segment_ids))
    assert len(te) >= 1


# ------------------------------------------------------------ regime labelling
def test_regime_labels_detect_synthetic_events(synth):
    labels = label_regimes(synth.power, synth.segment_ids, beam_jump_threshold=3,
                           power_drop_db_threshold=3.0, smoothing_window=3, dilation=1)
    assert labels.shape == (synth.n_samples,)
    assert set(np.unique(labels)) <= {STABLE, TRANSITION}
    ev = np.flatnonzero(synth.event_mask)
    assert len(ev) > 0
    # every injected event is labelled TRANSITION within +-2 steps
    hits = [labels[max(0, i - 2) : i + 3].max() == TRANSITION for i in ev]
    assert np.mean(hits) > 0.95
    # but most steps are stable (events are rare)
    assert np.mean(labels == STABLE) > 0.6


def test_regime_criteria_independently():
    n, b = 40, 64
    power = np.full((n, b), 1e-3)
    beams = np.full(n, 10)
    beams[20:] = 30  # a beam jump at t=20, no power change
    power[np.arange(n), beams] = 1.0
    lab = label_regimes(power, None, beam_jump_threshold=3, power_drop_db_threshold=100,
                        smoothing_window=1, dilation=0)
    assert lab[20] == TRANSITION and lab[:19].sum() == 0 and lab[22:].sum() == 0

    power2 = np.full((n, b), 1e-3)
    power2[:, 5] = 1.0
    power2[25:, 5] = 0.1  # 10 dB drop at t=25, same beam
    lab2 = label_regimes(power2, None, beam_jump_threshold=100, power_drop_db_threshold=3.0,
                         smoothing_window=1, drop_lookback=2, dilation=0)
    assert lab2[25] == TRANSITION and lab2[:25].sum() == 0 and lab2[28:].sum() == 0
    # a 1 dB drop must not trigger
    power3 = power2.copy()
    power3[25:, 5] = 10 ** (-0.1)
    assert label_regimes(power3, None, beam_jump_threshold=100, power_drop_db_threshold=3.0,
                         smoothing_window=1, dilation=0).sum() == 0


def test_regime_labels_respect_segment_boundaries():
    power = np.full((20, 64), 1e-3)
    power[:10, 3] = 1.0
    power[10:, 40] = 1.0  # jump exactly at a segment boundary -> not a transition
    seg = np.array([0] * 10 + [1] * 10)
    assert label_regimes(power, seg, smoothing_window=1, dilation=0).sum() == 0
    assert label_regimes(power, None, smoothing_window=1, dilation=0).sum() == 1


# ------------------------------------------------------- offline MDP / rewards
def test_reward_table_fully_observed(synth):
    w = make_windows(synth.power, synth.segment_ids, history=4, horizon=1)
    env = OfflineBeamEnv(w, reward_unit="linear", switching_penalty=0.0)
    table = env.reward_table()
    assert table.shape == (len(w), 64)
    np.testing.assert_allclose(table, w.target_power)  # every action's reward is measured
    db = env.reward_table("db")
    np.testing.assert_allclose(db, 10 * np.log10(w.target_power))
    assert np.array_equal(env.optimal_actions(), w.target_beam)


def test_switching_penalty_and_policy_reward(synth):
    w = make_windows(synth.power, synth.segment_ids, history=4, horizon=1)
    env = OfflineBeamEnv(w, reward_unit="db", switching_penalty=2.0)
    np.testing.assert_array_equal(env.switching_penalty_cost(np.array([1, 2, 3]), np.array([1, 5, 3])),
                                  [0.0, 2.0, 0.0])
    # the oracle picks the optimal beam every step: zero regret
    pr_opt = env.evaluate_policy_reward(w.target_beam)
    assert pr_opt.regret == pytest.approx(0.0)
    assert pr_opt.n_switches == int((w.last_beam() != w.target_beam).sum())
    np.testing.assert_allclose(pr_opt.per_step, env.reward_table().max(axis=1) - pr_opt.switch_cost)
    # holding the current beam pays no penalty but has non-negative regret
    pr_hold = env.evaluate_policy_reward(w.last_beam())
    assert pr_hold.n_switches == 0 and pr_hold.switch_cost.sum() == 0.0
    assert pr_hold.regret >= 0.0
    with pytest.raises(ValueError):
        env.evaluate_policy_reward(np.full(len(w), 64))


# ------------------------------------------------------------------- metrics
def test_metric_primitives():
    scores = np.array([[0.1, 0.9, 0.0], [0.5, 0.2, 0.3]])
    tgt = np.array([1, 2])
    assert top_k_accuracy(scores, tgt, 1) == 0.5
    assert top_k_accuracy(scores, tgt, 2) == 1.0
    tp = np.array([[1.0, 2.0, 0.5], [1.0, 0.1, 2.0]])
    np.testing.assert_allclose(power_loss_db(scores, tp), [0.0, 10 * np.log10(2.0)])


# ----------------------------------------------------------------- baselines
def _pipeline(synth, history=8, horizon=1):
    w = make_windows(synth.power, synth.segment_ids, history=history, horizon=horizon)
    labels = label_regimes(synth.power, synth.segment_ids)
    tr, te = split_segments(synth.segment_ids, 0.25, seed=0)
    train = w.subset(np.isin(w.segment_ids, tr))
    test = w.subset(np.isin(w.segment_ids, te))
    return train, test, regime_for_windows(labels, test.target_index)


def test_reactive_baseline(synth):
    train, test, reg = _pipeline(synth)
    scores = ReactiveBaseline().fit(train).predict_scores(test)
    assert scores.shape == (len(test), 64)
    np.testing.assert_array_equal(np.argmax(scores, axis=1), test.last_beam())
    onehot = ReactiveBaseline(rank_by_power=False).predict_scores(test)
    np.testing.assert_array_equal(np.argmax(onehot, axis=1), test.last_beam())
    m = compute_metrics(scores, test, reg)
    assert set(m) == {"overall", "stable", "transition"}
    assert m["overall"].n == len(test) and m["stable"].n + m["transition"].n == len(test)
    assert m["overall"].top1 > 0.5  # slow drift: holding the beam is usually right


def test_markov_baseline(synth):
    train, test, reg = _pipeline(synth)
    mk = MarkovBaseline(n_beams=64, smoothing=0.1).fit(train)
    assert mk.transition.shape == (64, 64)
    np.testing.assert_allclose(mk.transition.sum(axis=1), 1.0)
    scores = mk.predict_scores(test)
    assert scores.shape == (len(test), 64)
    m = compute_metrics(scores, test, reg)
    assert m["overall"].top1 > 0.5
    assert m["overall"].top3 >= m["overall"].top1
    # oracle is a strict upper bound
    o = compute_metrics(OracleBaseline().predict_scores(test), test, reg)
    assert o["overall"].top1 == 1.0 and o["overall"].power_loss_db_mean == pytest.approx(0.0)
    with pytest.raises(RuntimeError):
        MarkovBaseline().predict_scores(test)


def test_reactive_collapses_at_transitions(synth):
    """Key thesis check: reactive top-1 at TRANSITION <= at STABLE (and power loss larger)."""
    train, test, reg = _pipeline(synth)
    assert (reg == TRANSITION).sum() > 0 and (reg == STABLE).sum() > 0
    m = compute_metrics(ReactiveBaseline().fit(train).predict_scores(test), test, reg)
    assert m["transition"].top1 <= m["stable"].top1
    assert m["transition"].power_loss_db_mean >= m["stable"].power_loss_db_mean


# --------------------------------------------------------- column resolution
def test_resolve_columns_scenario33_header():
    header = ["index", "unit1_rgb", "unit1_pwr_60ghz", "unit1_lidar", "unit1_radar", "unit1_loc",
              "unit2_loc", "unit1_beam", "unit1_max_pwr", "time_stamp", "seq_index",
              "unit2_spd_over_grnd_kmph", "unit2_num_sats", "unit2_altitude"]
    r = resolve_columns(header)
    assert r["power"] == "unit1_pwr_60ghz"
    assert r["beam_label"] == "unit1_beam"
    assert r["max_power"] == "unit1_max_pwr"
    assert r["image"] == "unit1_rgb"
    assert r["lidar"] == "unit1_lidar"
    assert r["radar"] == "unit1_radar"
    assert set(r["gps"]) >= {"unit1_loc", "unit2_loc"}
    assert r["time"] == "time_stamp"
    assert r["sequence"] == "seq_index"


def test_nan_power_handling():
    """DeepSense has a few rows with dropped (NaN) beams; argmax must ignore them and the
    default policy interpolates along the beam axis."""
    power = np.random.default_rng(1).random((5, 64)) * 0.5
    power[1, 10] = 0.9
    power[1, 11] = np.nan  # dropped beam next to the peak
    power[3, 20:23] = np.nan
    assert optimal_beam(power)[1] == 10
    filled, n_rows = fill_nan_power(power, "interpolate")
    assert n_rows == 2 and np.isfinite(filled).all()
    assert filled[1, 11] == pytest.approx((power[1, 10] + power[1, 12]) / 2)
    np.testing.assert_array_equal(filled[0], power[0])
    assert np.isnan(fill_nan_power(power, "keep")[0]).any()
    with pytest.raises(ValueError):
        fill_nan_power(power, "raise")


# ------------------------------------------------------- predict-then-act
def _small_pta(**kw):
    from channeldreamer.models import PredictThenActBaseline

    base = {"epochs": 8, "batch_size": 128, "d_model": 32, "n_layers": 1, "n_heads": 2, "dim_feedforward": 64,
                "patience": 100, "device": "cpu", "mixed_precision": False}
    base.update(kw)
    return PredictThenActBaseline(**base)


def test_predict_then_act_interface(synth):
    train, test, reg = _pipeline(synth)
    pta = _small_pta().fit(train)
    scores = pta.predict_scores(test)
    assert scores.shape == (len(test), 64)
    assert np.isfinite(scores).all()
    assert pta.n_parameters() > 0
    assert len(pta.history_log) == 8
    m = compute_metrics(scores, test, reg)
    assert set(m) == {"overall", "stable", "transition"}
    assert m["overall"].n == len(test)
    with pytest.raises(RuntimeError):
        _small_pta().predict_scores(test)
    with pytest.raises(ValueError):  # trained history must match
        pta.predict_scores(make_windows(synth.power, synth.segment_ids, history=4, horizon=1))


def test_predict_then_act_learns_and_beats_reactive(synth):
    """Better prediction: at a 3-step horizon the forecaster must beat hold-last-beam overall
    and in the stable regime (where drift is extrapolable); at horizon 1 it must at least tie."""
    train, test, reg = _pipeline(synth, horizon=3)
    pta = _small_pta(epochs=25, loss="power").fit(train)
    assert pta.history_log[-1]["train_loss"] < pta.history_log[0]["train_loss"]
    m_pta = compute_metrics(pta.predict_scores(test), test, reg)
    m_re = compute_metrics(ReactiveBaseline().fit(train).predict_scores(test), test, reg)
    assert m_pta["overall"].top1 > m_re["overall"].top1 + 0.05
    assert m_pta["stable"].power_loss_db_mean < m_re["stable"].power_loss_db_mean

    train1, test1, reg1 = _pipeline(synth, horizon=1)
    m1 = compute_metrics(_small_pta(epochs=15).fit(train1).predict_scores(test1), test1, reg1)
    r1 = compute_metrics(ReactiveBaseline().predict_scores(test1), test1, reg1)
    assert m1["overall"].top1 >= r1["overall"].top1 - 0.05


def test_predict_then_act_tcn_variant(synth):
    train, test, _reg = _pipeline(synth, horizon=3)
    pta = _small_pta(arch="tcn", epochs=10).fit(train)
    assert pta.predict_scores(test).shape == (len(test), 64)
    with pytest.raises(ValueError):
        _small_pta(arch="rnn").fit(train)


def test_predict_then_act_ce_loss(synth):
    train, test, _reg = _pipeline(synth)
    pta = _small_pta(loss="ce", epochs=5).fit(train)
    assert np.isfinite(pta.predict_scores(test)).all()
