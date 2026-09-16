import random


class ReservoirReplay:
    """Fixed-size, seeded reservoir of indices from a training stream."""

    def __init__(self, capacity, batch_size, max_weight, seed):
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.max_weight = float(max_weight)
        self.seed = int(seed)
        if self.capacity < 1:
            raise ValueError('Replay capacity must be positive.')
        if self.batch_size < 1:
            raise ValueError('Replay batch size must be positive.')
        if not 0.0 <= self.max_weight < 1.0:
            raise ValueError('Replay max_weight must be in [0, 1).')
        self.indices = []
        self.seen = 0
        self._rng = random.Random(self.seed)

    def sample(self):
        count = min(self.batch_size, len(self.indices))
        return self._rng.sample(self.indices, count) if count else []

    def add(self, indices):
        for index in indices:
            index = int(index)
            self.seen += 1
            if len(self.indices) < self.capacity:
                self.indices.append(index)
                continue
            slot = self._rng.randrange(self.seen)
            if slot < self.capacity:
                self.indices[slot] = index

    @property
    def weight(self):
        """Ramp replay contribution linearly while the reservoir fills."""
        return self.max_weight * min(1.0, len(self.indices) / self.capacity)

    def state_dict(self):
        return {
            'capacity': self.capacity,
            'batch_size': self.batch_size,
            'max_weight': self.max_weight,
            'seed': self.seed,
            'indices': list(self.indices),
            'seen': self.seen,
            'rng_state': self._rng.getstate(),
        }

    def load_state_dict(self, state):
        expected = {
            'capacity': self.capacity,
            'batch_size': self.batch_size,
            'max_weight': self.max_weight,
            'seed': self.seed,
        }
        actual = {key: state[key] for key in expected}
        if actual != expected:
            raise ValueError('Replay checkpoint configuration differs.')
        indices = [int(index) for index in state['indices']]
        seen = int(state['seen'])
        if len(indices) > self.capacity or seen < len(indices):
            raise ValueError('Invalid replay checkpoint state.')
        self.indices = indices
        self.seen = seen
        self._rng.setstate(state['rng_state'])


def build_replay(config):
    replay = config.get('replay')
    if not replay or not replay.get('enabled', False):
        return None
    required = ('capacity', 'batch_size', 'max_weight', 'seed')
    missing = [key for key in required if key not in replay]
    if missing:
        raise ValueError(f'Missing replay settings: {", ".join(missing)}')
    return ReservoirReplay(*(replay[key] for key in required))
