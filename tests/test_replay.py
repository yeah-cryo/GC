import random

import pytest

from aigi_detection.training import ReservoirReplay, build_replay


def test_replay_is_disabled_by_default():
    assert build_replay({}) is None
    assert build_replay({'replay': {'enabled': False}}) is None


def test_reservoir_is_bounded_and_does_not_change_global_rng():
    random.seed(9)
    expected = random.random()
    random.seed(9)
    replay = ReservoirReplay(capacity=4, batch_size=3, max_weight=0.5, seed=42)
    replay.add(range(20))
    assert random.random() == expected
    assert replay.seen == 20
    assert len(replay.indices) == 4
    assert len(replay.sample()) == 3
    assert replay.weight == 0.5


def test_replay_resume_reproduces_sampling_and_replacement():
    first = ReservoirReplay(capacity=8, batch_size=4, max_weight=0.5, seed=7)
    first.add(range(13))
    state = first.state_dict()

    resumed = ReservoirReplay(capacity=8, batch_size=4, max_weight=0.5, seed=7)
    resumed.load_state_dict(state)
    assert resumed.sample() == first.sample()
    resumed.add(range(13, 30))
    first.add(range(13, 30))
    assert resumed.state_dict() == first.state_dict()


def test_replay_weight_ramps_with_fill():
    replay = ReservoirReplay(capacity=10, batch_size=4, max_weight=0.5, seed=1)
    assert replay.weight == 0.0
    replay.add(range(4))
    assert replay.weight == pytest.approx(0.2)
