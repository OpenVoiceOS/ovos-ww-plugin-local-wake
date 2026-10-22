import itertools

import numpy as np
import pytest

from ovos_ww_plugin_local_wake.inference import dtw_distance


def cosine_cost(x, y):
    return np.array([[1 - a @ b / np.linalg.norm(a) / np.linalg.norm(b) for b in y.T] for a in x.T])


def brute_force(x, y):
    """Cheapest monotone path from (0, 0) to (n-1, m-1), found by trying every path."""
    cost = cosine_cost(x, y)
    n, m = cost.shape
    best = np.inf
    steps = [(1, 1), (1, 0), (0, 1)]
    max_len = n + m - 2
    for length in range(max_len + 1):
        for path in itertools.product(steps, repeat=length):
            i = j = 0
            total = cost[0, 0]
            for di, dj in path:
                i, j = i + di, j + dj
                if i >= n or j >= m:
                    break
                total += cost[i, j]
            else:
                if (i, j) == (n - 1, m - 1):
                    best = min(best, total)
    return best / (n + m)


def test_hand_worked_example():
    # columns: x = [e1, e2], y = [e1, e1, e2]; the path (0,0) (0,1) (1,2) costs 0 + 0 + 0
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert dtw_distance(x, y) == 0.0
    # orthogonal frames cost 1 each; one frame against one frame: 1 / (1 + 1)
    assert dtw_distance(np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])) == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((5, rng.integers(1, 5)))
    y = rng.standard_normal((5, rng.integers(1, 5)))
    assert dtw_distance(x, y) == pytest.approx(brute_force(x, y), abs=1e-12)


def test_symmetric_and_zero_on_identity():
    rng = np.random.default_rng(7)
    x, y = rng.standard_normal((96, 6)), rng.standard_normal((96, 9))
    assert dtw_distance(x, y) == pytest.approx(dtw_distance(y, x))
    assert dtw_distance(x, x) == pytest.approx(0.0, abs=1e-12)
