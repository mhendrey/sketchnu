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

from collections import Counter
import numpy as np
from pathlib import Path
import pytest

from sketchnu.countmin import CountMin, load
from .zipf import zipf


vocab_size = 25
vocab = list(
    set([bytes(k) for k in np.random.randint(0, 256, (vocab_size, 8)).astype(np.uint8)])
)
zipf_p = zipf(vocab_size)
n_data = 50000
data_stream = np.random.choice(vocab, n_data, p=zipf_p).tolist()


def test_empty_linear(width: int = 25, depth: int = 8):
    """
    Test that a linear cms starts empty

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 25
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("linear", width, depth)
    for key in vocab:
        assert cms[key] == 0.0


def test_empty_log16(width: int = 25, depth: int = 8):
    """
    Test that a log16 cms starts empty

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 1024
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("log16", width, depth)
    for data in data_stream:
        assert cms[data] == 0.0


def test_empty_log8(width: int = 25, depth: int = 8):
    """
    Test that a log8 cms starts empty

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 1024
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("log8", width, depth)
    for data in data_stream:
        assert cms[data] == 0.0


def test_n_added_linear(width: int = 25, depth: int = 8):
    """
    Test that number of elements added to the sketch is properly tracked

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 25 (same as vocab_size)
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("linear", width, depth)
    cms.update(data_stream)
    assert cms.n_added() == len(data_stream)


def test_n_added_log16(width: int = 25, depth: int = 8):
    """
    Test that number of elements added to the sketch is properly tracked

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 25 (same as vocab_size)
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("log16", width, depth)
    cms.update(data_stream)
    assert cms.n_added() == len(data_stream)


def test_n_added_log8(width: int = 25, depth: int = 8):
    """
    Test that number of elements added to the sketch is properly tracked

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 25 (same as vocab_size)
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("log8", width, depth)
    cms.update(data_stream)
    assert cms.n_added() == len(data_stream)


def test_query_linear(width: int = 25, depth: int = 8):
    """
    Test that cms.query() is the same as __getitem__()

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 25 (same as vocab_size)
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("linear", width, depth)
    cms.update(data_stream[:100])
    assert cms.query(data_stream[0]) == cms[data_stream[0]]


def test_query_log16(width: int = 25, depth: int = 8):
    """
    Test that cms.query() is the same as __getitem__()

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 25 (same as vocab_size)
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("log16", width, depth)
    cms.update(data_stream[:5000])
    assert cms.query(data_stream[0]) == cms[data_stream[0]]


def test_query_log8(width: int = 25, depth: int = 8):
    """
    Test that cms.query() is the same as __getitem__()

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 25 (same as vocab_size)
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("log8", width, depth)
    cms.update(data_stream[:5000])
    assert cms.query(data_stream[0]) == cms[data_stream[0]]


def test_update_linear(width: int = 25, depth: int = 8):
    """
    Test that the count-min sketch error quarantees hold. We first assert that
    all estimates are greater than or equal to the true count.  Then we assert
    that with a probability of at least `1 - exp(-depth)` that:

        estimate <= true + N * exp(1) / width

    where `N` is the total number of elements added to the sketch.

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 25 (same as vocab_size)
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("linear", width, depth)
    cms.update(data_stream)
    true_count = Counter(data_stream)

    error = np.zeros(vocab_size)
    for i, data in enumerate(vocab):
        error[i] = cms.query(data) - true_count[data]

    max_error = cms.n_added() * np.exp(1) / width
    n_over_max = error[error > max_error].shape[0]

    assert error.min() >= 0.0, f"Minimum error, {error.min():.3f} is negative"
    assert (n_over_max / n_data) < np.exp(
        -depth
    ), f"Exceeded the limit too often, {n_over_max}"


def test_update_log16(
    width: int = 200,
    depth: int = 8,
    max_count: int = 4294967295,
    num_reserved: int = 1023,
    n_trials: int = 20,
    t_stat: float = 3.291,
):
    """
    Uses a t-test to test the null hypothesis that the mean of the
    difference between the true count and estimated count is 0. Uses a
    confidence level of 99.9% to reject the null hypotheses. The test asserts
    that we should fail to reject the null hypothesis.

    A log counter has an unbiased estimator of `n` with a variance of
    `(x-1)n(n + 1)/2`, where `x` is the log base.

    This test is for the updating log counters and not for the count-min sketch
    itself. In order to limit the errors introduced by collisions in the
    count-min sketch, we set the width >> vocab_size.

    Parameters
    ----------
    width : int, optional
        Width of the count-min sketch. Default is much larger than vocab_size to avoid
        collisions in vocab elements. Default is 200
    depth : int, optional
        Depth of the count-min sketch. Default is 8
    max_count : int, optional
        Maximum value that each log counter can store. Default is 2^32 - 1
    num_reserved : int, optional
        Perform linear counting for values [0, num_reserved]. After that use log
        counters. This gives more precise estimates for the number of times a key
        is seen for counts <= num_reserved. Default is 1023. This must be less than
        65,535 (2^16 - 1).
    n_trials : int, optional
        Number of different experiments to run. Default is 20
    t_stat : float, optional
        t-test statistic which sets the confidence interval. Default corresponds
        to 99.9%. Default is 3.291
    """
    # Randomly select n_trials keys. Test above & below the num_reserved
    keys = np.random.choice(vocab, n_trials, replace=False).tolist()
    for c in [500, 25000]:
        cms = CountMin("log16", width, depth, max_count, num_reserved)
        true_count = Counter()
        for _ in range(c):
            cms.update(keys)
            true_count.update(keys)

        error = np.zeros(n_trials)
        for i, key in enumerate(keys):
            error[i] = cms.query(key) - true_count[key]

        if error.std() == 0.0:
            assert error.mean() == 0.0, f"std = 0, but mean !=0, {error.mean():.3f}"
        else:
            t_value = np.abs(error.mean() / (error.std() / np.sqrt(n_trials)))
            # 99.9% confidence level
            assert t_value < t_stat, f"t-value {t_value:.4} is above {t_stat}"


def test_update_log8(
    width: int = 200,
    depth: int = 8,
    max_count: int = 4294967295,
    num_reserved: int = 15,
    n_trials: int = 20,
    t_stat: float = 3.291,
):
    """
    Uses a t-test to test the null hypothesis that the mean of the
    difference between the true count and estimated count is 0. Uses a
    confidence level of 99.9% to reject the null hypotheses. The test asserts
    that we should fail to reject the null hypothesis.

    A log counter has an unbiased estimator of `n` with a variance of
    `(x-1)n(n + 1)/2`, where `x` is the log base.

    This test is for the updating log counters and not for the count-min sketch
    itself. In order to limit the errors introduced by collisions in the
    count-min sketch, we set the width >> vocab_size.

    Parameters
    ----------
    width : int, optional
        Width of the count-min sketch. Default is much larger than vocab_size to avoid
        collisions in vocab elements. Default is 200
    depth : int, optional
        Depth of the count-min sketch. Default is 8
    max_count : int, optional
        Maximum value that each log counter can store. Default is 2^32 - 1
    num_reserved : int, optional
        Perform linear counting for values [0, num_reserved]. After that use log
        counters. This gives more precise estimates for the number of times a key
        is seen for counts <= num_reserved. Default is 15. This must be less than
        255 (2^8 - 1).
    n_trials : int, optional
        Number of different experiments to run. Default is 20
    t_stat : float, optional
        t-test statistic which sets the confidence interval. Default corresponds
        to 99.9%. Default is 3.291
    """
    # Randomly select n_trials keys. Test above & below the num_reserved
    keys = np.random.choice(vocab, n_trials, replace=False).tolist()
    for c in [12, 25000]:
        cms = CountMin("log8", width, depth, max_count, num_reserved)
        true_count = Counter()
        for _ in range(c):
            cms.update(keys)
            true_count.update(keys)

        error = np.zeros(n_trials)
        for i, key in enumerate(keys):
            error[i] = cms.query(key) - true_count[key]

        if error.std() == 0.0:
            assert error.mean() == 0.0, f"std = 0, but mean !=0, {error.mean():.3f}"
        else:
            t_value = np.abs(error.mean() / (error.std() / np.sqrt(n_trials)))
            # 99.9% confidence level
            assert t_value < t_stat, f"t-value {t_value:.4} is above {t_stat}"


def test_update_ngram_linear(width: int = 200, depth: int = 8):
    """
    Test that ngraming is done right. Default width is set wide enough to ensure
    that we don't have collisions in the count-min sketch.

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 200
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    data = b"0123456789abcdef"
    for ngram in range(1, len(data) + 2):
        cms = CountMin("linear", width, depth)
        cms.update_ngram([data], ngram)
        counter = Counter()
        for i in range(len(data) - ngram + 1):
            counter.update([data[i : i + ngram]])
        for k, v in counter.items():
            assert cms[k] == v


def test_update_ngram_log16(width: int = 200, depth: int = 8):
    """
    Test that ngraming is done right. Default width is set wide enough to ensure
    that we don't have collisions in the count-min sketch.

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 200
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    data = b"0123456789abcdef"
    for ngram in range(1, len(data) + 2):
        cms = CountMin("log16", width, depth)
        cms.update_ngram([data], ngram)
        counter = Counter()
        for i in range(len(data) - ngram + 1):
            counter.update([data[i : i + ngram]])
        for k, v in counter.items():
            assert cms[k] == v


def test_update_ngram_log8(width: int = 200, depth: int = 8):
    """
    Test that ngraming is done right. Default width is set wide enough to ensure
    that we don't have collisions in the count-min sketch.

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 200
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    data = b"0123456789abcdef"
    for ngram in range(1, len(data) + 2):
        cms = CountMin("log8", width, depth)
        cms.update_ngram([data], ngram)
        counter = Counter()
        for i in range(len(data) - ngram + 1):
            counter.update([data[i : i + ngram]])
        for k, v in counter.items():
            assert cms[k] == v


def test_merge_linear(width: int = 200, depth: int = 8):
    """
    Add same data stream into two sketches. Merge them. The results should match the
    true count since width is set large enough that there are no collisions between
    vocab elements (width >> vocab_size)

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 200
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms1 = CountMin("linear", width, depth)
    cms2 = CountMin("linear", width, depth)

    cms1.update(data_stream)
    cms2.update(data_stream)
    true_count = Counter(data_stream)
    true_count.update(data_stream)

    cms1.merge(cms2)
    for key in vocab:
        assert cms1.query(key) == true_count[key]


def test_merge_log16(width: int = 200, depth: int = 8, t_stat: float = 3.291):
    """
    Add same data stream into two sketches. Merge them. Results should be close
    to the true_count. width >> vocab_size to avoid collisions in vocab elements

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 200
    depth : int, optional
        Depth of the sketch. Default is 8
    t_stat : float, optional
        t-test statistic which sets the confidence interval. Default corresponds
        to 99.9%. Default is 3.291
    """
    cms1 = CountMin("log16", width, depth)
    cms2 = CountMin("log16", width, depth)

    cms1.update(data_stream)
    cms2.update(data_stream)
    true_count = Counter(data_stream)
    true_count.update(data_stream)

    cms1.merge(cms2)
    error = np.zeros(vocab_size)
    for i, key in enumerate(vocab):
        error[i] = cms1[key] - true_count[key]

    if error.std() == 0.0:
        assert error.mean() == 0.0, f"std = 0, but mean !=0, {error.mean():.3f}"
    else:
        t_value = np.abs(error.mean() / (error.std() / np.sqrt(vocab_size)))
        # 99.9% confidence level
        assert t_value < t_stat, f"t-value {t_value:.4} is above {t_stat}"


def test_merge_log8(width: int = 200, depth: int = 8, t_stat: float = 3.291):
    """
    Add same data stream into two sketches. Merge them. Results should be close
    to the true_count. width >> vocab_size to avoid collisions in vocab elements

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 200
    depth : int, optional
        Depth of the sketch. Default is 8
    t_stat : float, optional
        t-test statistic which sets the confidence interval. Default corresponds
        to 99.9%. Default is 3.291
    """
    cms1 = CountMin("log8", width, depth)
    cms2 = CountMin("log8", width, depth)

    cms1.update(data_stream)
    cms2.update(data_stream)
    true_count = Counter(data_stream)
    true_count.update(data_stream)

    cms1.merge(cms2)
    error = np.zeros(vocab_size)
    for i, key in enumerate(vocab):
        error[i] = cms1[key] - true_count[key]

    if error.std() == 0.0:
        assert error.mean() == 0.0, f"std = 0, but mean !=0, {error.mean():.3f}"
    else:
        t_value = np.abs(error.mean() / (error.std() / np.sqrt(vocab_size)))
        # 99.9% confidence level
        assert t_value < t_stat, f"t-value {t_value:.4} is above {t_stat}"


def test_saving_linear(tmp_path: Path, width: int = 25, depth: int = 8):
    """
    Test saving and loading a linear count-min sketch

    Parameters
    ----------
    tmp_path : Path
        This is a pytest fixture which will provide a temporary directory unique
        to the test invocation, created in the base temporary directory
    width : int, optional
        Width of the sketch. Default is 25
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("linear", width, depth)
    cms.update(data_stream)
    filename = tmp_path / "test_linear.npz"

    cms.save(filename)
    cms_load = load(filename)

    for data in vocab:
        assert cms[data] == cms_load[data]


def test_saving_log16(tmp_path: Path, width: int = 25, depth: int = 8):
    """
    Test saving and loading a log16 count-min sketch

    Parameters
    ----------
    tmp_path : Path
        This is a pytest fixture which will provide a temporary directory unique
        to the test invocation, created in the base temporary directory
    width : int, optional
        Width of the sketch. Default is 25
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("log16", width, depth)
    cms.update(data_stream)
    filename = tmp_path / "test_log16.npz"

    cms.save(filename)
    cms_load = load(filename)

    for data in vocab:
        assert cms[data] == cms_load[data]


def test_saving_log8(tmp_path: Path, width: int = 25, depth: int = 8):
    """
    Test saving and loading a log8 count-min sketch

    Parameters
    ----------
    tmp_path : Path
        This is a pytest fixture which will provide a temporary directory unique
        to the test invocation, created in the base temporary directory
    width : int, optional
        Width of the sketch. Default is 25
    depth : int, optional
        Depth of the sketch. Default is 8
    """
    cms = CountMin("log8", width, depth)
    cms.update(data_stream)
    filename = tmp_path / "test_log8.npz"

    cms.save(filename)
    cms_load = load(filename)

    for data in vocab:
        assert cms[data] == cms_load[data]


def test_max_count_log16(width: int = 25, depth: int = 8, max_count: int = 100000):
    """
    Test that the counters max out at max_count

    Parameters
    ----------
    width : int, optional
        Width of the count-min sketch. Default is 25
    depth : int, optional
        Depth of the count-min sketch. Default is 8
    max_count : int, optional
        Maximum value that each log counter can store. If set too low, then
        ValueError is thrown because the base of log counter is ~ 1.0.
        Default is 100,000
    """
    cms = CountMin("log16", width, depth, max_count)
    for _ in range(int(max_count * 1.25)):
        cms.update(vocab)

    for key in vocab:
        assert cms[key] == pytest.approx(max_count)


def test_max_count_log8(width: int = 25, depth: int = 8, max_count: int = 1000):
    """
    Test that the counters max out at max_count

    Parameters
    ----------
    width : int, optional
        Width of the count-min sketch. Default is 25
    depth : int, optional
        Depth of the count-min sketch. Default is 8
    max_count : int, optional
        Maximum value that each log counter can store. If set too low, then
        ValueError is thrown because the base of log counter is ~ 1.0.
        Default is 1000
    """
    cms = CountMin("log8", width, depth, max_count)
    for _ in range(int(max_count * 2.0)):
        cms.update(vocab)

    for key in vocab:
        assert cms[key] == pytest.approx(max_count)


def test_num_reserved_log16(
    width: int = 200,
    depth: int = 8,
    max_count: int = 4294967295,
    num_reserved: int = 50,
):
    """
    Test that linear counting is done for counts between [0, num_reserved + 1]

    Parameters
    ----------
    width : int, optional
        Width of the count-min sketch. Default is 25
    depth : int, optional
        Depth of the count-min sketch. Default is 8
    max_count : int, optional
        Maximum value that each log counter can store. Default is 2^32 - 1
    num_reserved : int, optional
        Perform linear counting from [0, num_reserved+1]. Default is 50
    """
    cms = CountMin("log16", width, depth, max_count, num_reserved)
    for _ in range(num_reserved + 1):
        cms.update(vocab)
    for key in vocab:
        assert cms[key] == pytest.approx(num_reserved + 1)

    cms.update(vocab)
    # Assert that we are now approximating
    for key in vocab:
        assert cms[key] != pytest.approx(num_reserved + 2)


def test_num_reserved_log8(
    width: int = 200,
    depth: int = 8,
    max_count: int = 4294967295,
    num_reserved: int = 50,
):
    """
    Test that linear counting is done for counts between [0, num_reserved + 1]

    Parameters
    ----------
    width : int, optional
        Width of the count-min sketch. Default is 25
    depth : int, optional
        Depth of the count-min sketch. Default is 8
    max_count : int, optional
        Maximum value that each log counter can store. Default is 2^32 - 1
    num_reserved : int, optional
        Perform linear counting from [0, num_reserved+1]. Default is 50
    """
    cms = CountMin("log8", width, depth, max_count, num_reserved)
    for _ in range(num_reserved + 1):
        cms.update(vocab)
    for key in vocab:
        assert cms[key] == pytest.approx(num_reserved + 1)

    cms.update(vocab)
    # Assert that we are now approximating
    for key in vocab:
        assert cms[key] != pytest.approx(num_reserved + 2)


if __name__ == "__main__":
    test_empty_linear()
    test_empty_log16()
    test_empty_log8()
    test_n_added_linear()
    test_n_added_log16()
    test_n_added_log8()
    test_query_linear()
    test_query_log16()
    test_query_log8()
    test_update_linear()
    test_update_log16()
    test_update_log8()
    test_update_ngram_linear()
    test_update_ngram_log16()
    test_update_ngram_log8()
    test_merge_linear()
    test_merge_log16()
    test_merge_log8()
    test_max_count_log16()
    test_max_count_log8()
    test_num_reserved_log16()
    test_num_reserved_log8()
    # test_saving_linear() # Need to give directory & delete afterwards
    # test_saving_log16()
    # test_saving_log8()
