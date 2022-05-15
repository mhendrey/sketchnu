"""
Sketchnu has Numba implementations of sketch algorithms and other useful functions
that utilize hash functions.

Copyright (C) 2022 Matthew Hendrey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
import numpy as np
from pathlib import Path
import pytest

from sketchnu.hyperloglog import HyperLogLog

n_keys = 50000
n_trials = 20
keys = list(
    set([bytes(k) for k in np.random.randint(0, 256, (n_keys, 8)).astype(np.uint8)])
)
n_keys = len(keys)


def test_empty(p: int = 10):
    """
    Test that empty HyperLogLog returns 0.0

    Parameters
    ----------
    p : int, optional
        Precision of the HyperLogLog. Default is 10
    """
    hll = HyperLogLog(p)
    assert hll.query() == 0.0


def test_repeat_keys(p: int = 10):
    """
    Test that estimated count does change when you put the same set of keys in
    """
    hll = HyperLogLog(p)
    hll.update(keys)
    n_uniq = hll.query()
    hll.update(keys)
    assert n_uniq == hll.query()


def test_update_low(p: int = 10, n_trials: int = 20, t_stat: float = 3.291):
    """
    Uses a t-test to test the null hypothesis that the mean of the
    difference between the true count and estimated count is 0. Uses a
    confidence level of 99.9% to reject the null hypotheses. The test asserts
    that we should fail to reject the null hypothesis.

    We insert a random number of keys between [5, threshold - 5]. We repeat this
    test n_trials times using different seeds. The query() should equal the number
    of unique keys that were added to the HyperLogLog.

    Parameters
    ----------
    p : int, optional
        Precision of the HyperLogLog. Default is a relative low value of 10 in order
        to lower the threshold value for faster testing. Default is 10
    n_trials : int, optional
        Number of different experiments to run. Each experiment uses a different
        seed to initialize a HyperLogLog. Default is 20
    t_stat : float, optional
        t-test statistic which sets the confidence interval. Default corresponds
        to 99.9%. Default is 3.291
    """
    hll = HyperLogLog(p)
    error = np.zeros(n_trials)
    n_uniq = np.random.randint(5, hll.threshold - 5)
    n_uniq = min(n_uniq, n_keys)
    for seed in range(n_trials):
        hll = HyperLogLog(p, seed)
        hll.update(keys[:n_uniq])
        error[seed] = n_uniq - hll.query()

    if error.std() == 0.0:
        assert error.mean() == 0.0, f"std = 0, but mean != 0, {error.mean():.3f}"
    else:
        t_value = np.abs(error.mean() / (error.std() / np.sqrt(n_trials)))
        # 99.9% confidence interval
        assert t_value < t_stat, f"t-value {t_value:.4} is above {t_stat}"


def test_update_med(p: int = 10, n_trials: int = 20, t_stat: float = 3.291):
    """
    Uses a t-test to test the null hypothesis that the mean of the
    difference between the true count and estimated count is 0. Uses a
    confidence level of 99.9% to reject the null hypotheses. The test asserts
    that we should fail to reject the null hypothesis.

    We insert a random number of keys between [threshold+20, 5 * 2^p]. We repeat
    this test n_trials times using different seeds. The query() should equal the
    number of unique keys that were added to the HyperLogLog.

    Parameters
    ----------
    p : int, optional
        Precision of the HyperLogLog. Default is a relative low value of 10 in order
        to lower the threshold value for faster testing. Default is 10
    n_trials : int, optional
        Number of different experiments to run. Each experiment uses a different
        seed to initialize a HyperLogLog. Default is 20
    t_stat : float, optional
        t-test statistic which sets the confidence interval. Default corresponds
        to 99.9%. Default is 3.291
    """
    hll = HyperLogLog(p)
    error = np.zeros(n_trials)
    n_uniq = np.random.randint(hll.threshold + 20, 5 * (2 ** p))
    n_uniq = min(n_uniq, n_keys)
    for seed in range(n_trials):
        hll = HyperLogLog(p, seed)
        hll.update(keys[:n_uniq])
        error[seed] = n_uniq - hll.query()

    if error.std() == 0.0:
        assert error.mean() == 0.0, f"std = 0, but mean != 0, {error.mean():.3f}"
    else:
        t_value = np.abs(error.mean() / (error.std() / np.sqrt(n_trials)))
        # 99.9% confidence interval
        assert t_value < t_stat, f"t-value {t_value:.4} is above {t_stat}"


def test_update_hi(p: int = 10, n_trials: int = 20, t_stat: float = 3.291):
    """
    Uses a t-test to test the null hypothesis that the mean of the
    difference between the true count and estimated count is 0. Uses a
    confidence level of 99.9% to reject the null hypotheses. The test asserts
    that we should fail to reject the null hypothesis.

    We insert all the keys which tests the algorithm above the 5 * 2^p limit.
    We repeat this test n_trials times using different seeds. The query() should
    equal the number of unique keys that were added to the HyperLogLog.

    Parameters
    ----------
    p : int, optional
        Precision of the HyperLogLog. Default is a relative low value of 10 in order
        to lower the threshold value for faster testing. Default is 10
    n_trials : int, optional
        Number of different experiments to run. Each experiment uses a different
        seed to initialize a HyperLogLog. Default is 20
    t_stat : float, optional
        t-test statistic which sets the confidence interval. Default corresponds
        to 99.9%. Default is 3.291
    """
    error = np.zeros(n_trials)
    assert n_keys > (5 * (2 ** p))
    n_uniq = n_keys
    for seed in range(n_trials):
        hll = HyperLogLog(p, seed)
        hll.update(keys[:n_uniq])
        error[seed] = n_uniq - hll.query()

    if error.std() == 0.0:
        assert error.mean() == 0.0, f"std = 0, but mean != 0, {error.mean():.3f}"
    else:
        t_value = np.abs(error.mean() / (error.std() / np.sqrt(n_trials)))
        # 99.9% confidence interval
        assert t_value < t_stat, f"t-value {t_value:.4} is above {t_stat}"


def test_update_ngram(p: int = 16):
    """
    Test that ngraming is done right

    Parameters
    ----------
    p : int, optional
        Precision of the HyperLogLog. Default is 16
    """
    data = b"0123456789abcdef"
    for ngram in range(1, len(data) + 1):
        hll = HyperLogLog(p)
        hll.update_ngram([data], ngram)
        true_count = len(data) - ngram + 1
        assert hll.query() == pytest.approx(true_count, rel=1e-3)


def test_merge(p: int = 10, n_trials: int = 20, t_stat: float = 3.291):
    """
    Uses a t-test to test the null hypothesis that the mean of the
    difference between the true count and estimated count is 0. Uses a
    confidence level of 99.9% to reject the null hypotheses. The test asserts
    that we should fail to reject the null hypothesis.

    We insert `n_keys` random keys into the first HyperLogLog and `n_keys`
    different random keys into a second HyperLogLog. We merge the second into the
    first and then check that the estimated count is equal to the size of the union
    of the two sets of random keys.

    We repeat this test n_trials times using different seeds. The query() should
    equal the number of unique keys that were added to the HyperLogLog.

    Parameters
    ----------
    p : int, optional
        Precision of the HyperLogLog. Default is a relative low value of 10 in order
        to lower the threshold value for faster testing. Default is 10
    n_trials : int, optional
        Number of different experiments to run. Each experiment uses a different
        seed to initialize a HyperLogLog. Default is 20
    t_stat : float, optional
        t-test statistic which sets the confidence interval. Default corresponds
        to 99.9%. Default is 3.291
    """
    keys2 = list(
        set([bytes(k) for k in np.random.randint(0, 256, (n_keys, 8)).astype(np.uint8)])
    )
    n_uniq = len(set(keys2).union(set(keys)))

    error = np.zeros(n_trials)
    for seed in range(n_trials):
        hll1 = HyperLogLog(p, seed)
        hll2 = HyperLogLog(p, seed)
        hll1.update(keys)
        hll2.update(keys2)
        hll1.merge(hll2)
        error[seed] = n_uniq - hll1.query()

    if error.std() == 0.0:
        assert error.mean() == 0.0, f"std = 0, but mean !=0, {error.mean():.3f}"
    else:
        t_value = np.abs(error.mean() / (error.std() / np.sqrt(n_trials)))
        # 99.9% confidence level
        assert t_value < t_stat, f"t-value {t_value:.4} is above {t_stat}"


def test_saving(tmp_path: Path, p: int = 10):
    """
    Test saving and loading a HyperLogLog

    Parameters
    ----------
    tmp_path : Path
        This is a pytest fixture which will provide a temporary directory unique
        to the test invocation, created in the base temporary directory
    p : int, optional
        Precision of the HyperLogLog. Default is 10
    """
    hll = HyperLogLog(p)
    hll.update(keys)
    filename = tmp_path / "test_hll.npz"

    hll.save(filename)
    hll_load = HyperLogLog.load(filename)

    assert hll.query() == hll_load.query()


if __name__ == "__main__":
    test_empty()
    test_repeat_keys()
    test_update_low()
    test_update_med()
    test_update_hi()
    test_merge()
