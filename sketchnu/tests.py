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

from sketchnu.hashes import fasthash32, murmur3
from sketchnu.countmin import CountMin
from sketchnu.hyperloglog import HyperLogLog


def test_fasthash():
    """
    Compare the fasthash32 against the smhasher C++ version. Since the
    fasthash32() calls fasthash64() and then does some bit mixing, by testing
    the fasthash32() we are also testing the fasthash64().

    Values we assert against are from running the C++ version with the given
    keys and seeds. C++ code was taken from
    https://github.com/rurban/smhasher/blob/master/fasthash.cpp
    https://github.com/rurban/smhasher/blob/master/fasthash.h

    """
    key = b"0123456789abcdef"

    assert fasthash32(key, 0) == 128551002
    assert fasthash32(key, 5) == 571860520

    # Now check the different lengths
    assert fasthash32(key[:15], 3) == 4264631007
    assert fasthash32(key[:14], 4) == 3611610185
    assert fasthash32(key[:13], 5) == 2978977373
    assert fasthash32(key[:12], 6) == 2071843509
    assert fasthash32(key[:11], 7) == 3386775091
    assert fasthash32(key[:10], 8) == 2472970926
    assert fasthash32(key[:9], 21) == 1787443542
    assert fasthash32(key[:8], 22) == 2970440548
    assert fasthash32(key[:7], 23) == 3793135117
    assert fasthash32(key[:6], 24) == 3662885582
    assert fasthash32(key[:5], 25) == 2453668041
    assert fasthash32(key[:4], 26) == 635486060
    assert fasthash32(key[:3], 27) == 58999216
    assert fasthash32(key[:2], 28) == 3486011618
    assert fasthash32(key[:1], 29) == 3407281718

    hv1 = fasthash32(b"test", 0)
    assert hv1 == 2542785854

    hv2 = fasthash32(b"abc", 1)
    assert hv2 == 558486214

    hv3 = fasthash32(b"123", 2)
    assert hv3 == 3103508967


def test_murmur3():
    """
    Compare the murmur3 against the smhasher C++ version found in the
    MurmurHash3_x86_32() function.

    Values we assert against are from running the C++ version with the given
    keys and seeds. C++ code was taken from
    https://github.com/rurban/smhasher/blob/master/MurmurHash3.cpp
    https://github.com/rurban/smhasher/blob/master/MurmurHash3.h

    """
    hv1 = murmur3(b"test", 0)
    assert hv1 == 3127628307

    hv2 = murmur3(b"abc", 1)
    assert hv2 == 2859854335

    hv3 = murmur3(b"123", 2)
    assert hv3 == 1498078391


def test_cms_linear(
    n_keys: int = 1024,
    c_min: int = 1,
    c_max: int = 1024,
    width: int = 1024,
    depth: int = 8,
):
    """
    Test that the count-min sketch error quarantees hold. We first assert that
    all estimates are greater than or equal to the true count.  Then we assert
    that with a probability of at least `1 - exp(-depth)` that:

        estimate <= true + N * exp(1) / width

    where `N` is the total number of elements added to the sketch.

    We generate `n_keys` random keys. Each key is added a different number
    of times based upon a linear spacing between [`c_min`, `c_max`].

    Parameters
    ----------
    n_keys : int, optional
        Number of keys to test over. Default is 1024
    c_min : int, optional
        Minimum number of times a key is added into the sketch. Default is 1
    c_max : int, optional
        Maximum number of times a key is added into the sketch. Default is 1024
    width : int, optional
        Width of the count-min sketch. Default is 1024
    depth : int, optional
        Depth of the count-min sketch. Default is 8

    """
    # Generate n_keys random 16-byte keys
    keys = [bytes(r) for r in np.random.randint(0, 256, (n_keys, 16), np.uint8)]
    # Generate the number of times we will insert each of the keys
    # Equally spaced between c_min and c_max
    n_times = np.linspace(c_min, c_max, n_keys, dtype=np.int64)

    cms = CountMin("linear", width, depth)
    for n, key in zip(n_times, keys):
        for i in range(n):
            cms.add(key)

    error = np.zeros(n_keys)
    for i, (n, key) in enumerate(zip(n_times, keys)):
        error[i] = cms.query(key) - n

    max_error = cms.n_added() * np.exp(1) / width
    n_over_max = error[error > max_error].shape[0]

    assert error.min() >= 0.0, f"Minimum error, {error.min():.3f} is negative"
    assert (n_over_max / n_keys) < np.exp(
        -depth
    ), f"Exceeded the limit too often, {n_over_max}"


def test_cms_log8_update(
    n_keys: int = 500,
    c_min: int = 15,
    c_mid: int = 5000,
    c_max: int = 100000,
    max_count: int = 2 ** 32 - 1,
    num_reserved: int = 15,
):
    """
    Uses a t-test to test the null hypothesis that the mean of the
    difference between the true count and estimated count is 0. Uses a
    confidence level of 99% to reject the null hypotheses. The test asserts
    that we should fail to reject the null hypothesis.

    A log counter has an unbiased estimator of `n` (cardinality) with a variance
    of `(x-1)n(n + 1)/2`, where `x` is the log base.

    We generate `n_keys` random keys and then add them a total of `c_max` times to
    the count-min sketch. We check the error against the true value at three
    different values: `c_min`, `c_mid`, & `c_max`.

    The default `c_min` value is set to test the linear regime to ensure that is
    done well. The other two values, `c_mid` & `c_max`, test the log regime.

    This test is for the updating log counters and not for the count-min sketch
    itself. In order to limit the errors introduced by collisions in the
    count-min sketch, we set the width equal to the total number of elements
    inserted into the sketch times 3, i.e., `n_keys*c_max*3`.  This should give
    a maximum error of 

        count_estimate <= count_true + e/3

    that is not exceeded with probability of `1-exp(-depth)`.

    Parameters
    ----------
    n_keys : int, optional
        Number of keys to test over. Default is 500
    c_min : int, optional
        Lowest value to check the error between the estimated count and the
        true count for all n_keys. Default is 15
    c_mid : int, optional
        Medium value to check the error between the estimated count and the
        true count for all n_keys. Default is 5000
    c_max : int, optional
        Highest value to check the error between the estimated count and the
        true count for all n_keys. Default is 100000.
    max_count : int, optional
        Passed to CountMinLog() to specify max value available in the count-min
        sketch. Default is 2**32-1
    num_reserved : int, optional
        Passed to CountMinLog() to specify the num_reserved. Default is 15

    """
    # Generate n_keys random 16-byte keys
    keys = [bytes(r) for r in np.random.randint(0, 256, (n_keys, 16), np.uint8)]
    # Extremely small chance that duplicate keys are created
    keys = list(set(keys))
    n_keys = len(keys)

    w = n_keys * c_max * 3
    cms = CountMin("log8", w, max_count=max_count, num_reserved=num_reserved)
    error = np.zeros(n_keys)

    for n in range(c_max):
        cms.update(keys)
        if (n + 1) in [c_min, c_mid, c_max]:
            for i, key in enumerate(keys):
                error[i] = cms.query(key) - (n + 1)
            if error.std() == 0.0:
                assert (
                    error.mean() == 0.0
                ), f"After {n+1} inserts std=0, but mean = {error.mean():.3f}"
            else:
                t_value = np.abs(error.mean()) / (error.std() / np.sqrt(n_keys))
                # 99% Confidence Level
                assert (
                    t_value < 2.576
                ), f"After {n+1} inserts: t-value {t_value:.4} is above 2.576"


def test_cms_log8_merge(
    n_keys: int = 100,
    cms_updates=5000,
    other_updates=300,
    max_count: int = 2 ** 32 - 1,
    num_reserved: int = None,
):
    """
    Uses a t-test to test the null hypothesis that the mean of the
    difference between the true count and estimated count is 0. Uses a
    confidence level of 99% to reject the null hypotheses. The test asserts
    that we should fail to reject the null hypothesis.

    A log counter has an unbiased estimator of `n` (cardinality) with a variance
    of `(x-1)n(n + 1)/2`, where `x`` is the log base.

    We insert n_keys into two sketches, 'cms' & 'other'. We insert each key
    `cms_updates` times into 'cms' and `other_updates` times into 'other'. We then
    merge 'other' into 'cms'. The estimated count for each key should equal to
    `cms_updates + other_updates`.

    This test is for the merging log counters and not for the count-min sketch
    itself. In order to limit the errors introduced by collisions in the
    count-min sketch, we set the width equal to the total number of elements
    inserted into the sketch times 3.  This should give a maximum error of 

        count_estimate <= count_true + e/3

    that is not exceeded with probability of `1-exp(-depth)`.

    Parameters
    ----------
    n_keys : int, optional
        Number of keys to test over. Default is 100
    cms_updates : int, optional
        Number of times to insert each key into the first sketch.
        Default is 5000
    other_updates : int, optional
        Number of times to insert each key into the second sketch.
        Default is 300.
    max_count : int, optional
        Passed to CountMin() to specify max value available in the count-min
        sketch. Default is 2**32-1
    num_reserved : int, optional
        Passed to CountMin() to specify the num_reserved. Default is None.

    """
    # Random 16-byte keys
    keys = [bytes(r) for r in np.random.randint(0, 256, (n_keys, 16), np.uint8)]

    l_cms = keys * cms_updates
    l_other = keys * other_updates
    N = len(l_cms) + len(l_other)

    cms = CountMin("log8", 3 * N, max_count=max_count, num_reserved=num_reserved)
    other = CountMin("log8", 3 * N, max_count=max_count, num_reserved=num_reserved)

    cms.update(l_cms)
    other.update(l_other)
    cms.merge(other)

    error = np.zeros(n_keys)
    for i in range(n_keys):
        error[i] = cms.query(keys[i]) - (cms_updates + other_updates)

    # If we happen to have stayed in the linear regime
    # then if we have zero std, we should also have zero mean.
    if error.std() == 0.0:
        assert error.mean() == 0.0, f"std = 0, but mean !=0, {error.mean():.3f}"
    else:
        t_value = np.abs(error.mean()) / (error.std() / np.sqrt(n_keys))
        # 99% Confidence Level
        assert t_value < 2.576, f"t-value {t_value:.4} is above 2.576"


def test_cms_log16_update(
    n_keys: int = 500,
    c_min: int = 1023,
    c_mid: int = 5000,
    c_max: int = 100000,
    max_count: int = 2 ** 32 - 1,
    num_reserved: int = 1023,
):
    """
    Uses a t-test to test the null hypothesis that the mean of the
    difference between the true count and estimated count is 0. Uses a
    confidence level of 99% to reject the null hypotheses. The test asserts
    that we should fail to reject the null hypothesis.

    We generate `n_keys` random keys and then add them a total of `c_max` times to
    the count-min sketch. We check the error against the true value at three
    different values: `c_min`, `c_mid`, & `c_max`.

    The default `c_min` value is set to test the linear regime to ensure that is
    done well. The other two values, `c_mid` & `c_max`, test the log regime.

    This test is for the updating log counters and not for the count-min sketch
    itself. In order to limit the errors introduced by collisions in the
    count-min sketch, we set the width equal to the total number of elements
    inserted into the sketch times 3, i.e., `n_keys*c_max*3`.  This should give
    a maximum error of

        count_estimate <= count_true + e/3

    that is not exceeded with probability of `1-exp(-depth)`.

    Parameters
    ----------
    n_keys : int, optional
        Number of keys to test over. Default is 500
    c_min : int, optional
        Lowest value to check the error between the estimated count and the
        true count for all n_keys. Default is 1023
    c_mid : int, optional
        Medium value to check the error between the estimated count and the
        true count for all n_keys. Default is 5000
    c_max : int, optional
        Highest value to check the error between the estimated count and the
        true count for all n_keys. Default is 100000.
    max_count : int, optional
        Passed to CountMinLog() to specify max value available in the count-min
        sketch. Default is 2**32-1
    num_reserved : int, optional
        Passed to CountMinLog() to specify the num_reserved. Default is 1023.
    """
    # Generate n_keys random 16-byte keys
    keys = [bytes(r) for r in np.random.randint(0, 256, (n_keys, 16), np.uint8)]
    # Extremely small chance that duplicate keys are created
    keys = list(set(keys))
    n_keys = len(keys)

    w = n_keys * c_max * 3
    cms = CountMin("log16", w, max_count=max_count, num_reserved=num_reserved)
    error = np.zeros(n_keys)

    for n in range(c_max):
        cms.update(keys)
        if (n + 1) in [c_min, c_mid, c_max]:
            for i, key in enumerate(keys):
                error[i] = cms.query(key) - (n + 1)
            if error.std() == 0.0:
                assert (
                    error.mean() == 0.0
                ), f"After {n+1} inserts std=0, but mean = {error.mean():.3f}"
            else:
                t_value = np.abs(error.mean()) / (error.std() / np.sqrt(n_keys))
                # 99% Confidence Level
                assert (
                    t_value < 2.576
                ), f"After {n+1} inserts: t-value {t_value:.4} is above 2.576"


def test_cms_log16_merge(
    n_keys: int = 100,
    cms_updates=5000,
    other_updates=300,
    max_count: int = 2 ** 32 - 1,
    num_reserved: int = 50,
):
    """
    Uses a t-test to test the null hypothesis that the mean of the
    difference between the true count and estimated count is 0. Uses a
    confidence level of 99% to reject the null hypotheses. The test asserts
    that we should fail to reject the null hypothesis.

    We insert `n_keys` into two sketches, 'cms' & 'other'. We insert each key
    `cms_updates` times into 'cms' and `other_updates` times into 'other'. We then
    merge 'other' into 'cms'. The estimated count for each key should equal to
    `cms_updates + other_updates`.

    This test is for the merging log counters and not for the count-min sketch
    itself. In order to limit the errors introduced by collisions in the
    count-min sketch, we set the width equal to the total number of elements
    inserted into the sketch times 3.  This should give a maximum error of

        count_estimate <= count_true + e/3

    that is not exceeded with probability of `1-exp(-depth)`.

    Parameters
    ----------
    n_keys : int, optional
        Number of keys to test over. Default is 100
    cms_updates : int, optional
        Number of times to insert each key into the first sketch.
        Default is 5000
    other_updates : int, optional
        Number of times to insert each key into the second sketch.
        Default is 300.
    max_count : int, optional
        Passed to CountMinLog() to specify max value available in the count-min
        sketch. Default is 2**32-1
    num_reserved : int, optional
        Passed to CountMinLog() to specify the num_reserved. Default is 50.

    """
    # Random 16-byte keys
    keys = [bytes(r) for r in np.random.randint(0, 256, (n_keys, 16), np.uint8)]

    l_cms = keys * cms_updates
    l_other = keys * other_updates
    N = len(l_cms) + len(l_other)

    cms = CountMin("log16", 3 * N, max_count=max_count, num_reserved=num_reserved)
    other = CountMin("log16", 3 * N, max_count=max_count, num_reserved=num_reserved)

    cms.update(l_cms)
    other.update(l_other)
    cms.merge(other)

    error = np.zeros(n_keys)
    for i in range(n_keys):
        error[i] = cms.query(keys[i]) - (cms_updates + other_updates)

    # If we happen to have stayed in the linear regime
    # then if we have zero std, we should also have zero mean.
    if error.std() == 0.0:
        assert error.mean() == 0.0, f"std = 0, but mean !=0, {error.mean():.3f}"
    else:
        t_value = np.abs(error.mean()) / (error.std() / np.sqrt(n_keys))
        # 99% Confidence Level
        assert t_value < 2.576, f"t-value {t_value:.4} is above 2.576"


def test_hll_update(p: int = 10, n_keys: int = 100000, n_trials: int = 100):
    """
    Uses a t-test to test the null hypothesis that the mean of the
    difference between the true count and estimated count is 0. Uses a
    confidence level of 99% to reject the null hypotheses. The test asserts
    that we should fail to reject the null hypothesis.

    The HyperLogLog's estimated cardinality "is asymptotically almost unbiased".
    Despite this, even low values typically pass the test.

    We insert the same `n_keys` random keys into a HyperLogLog with precision
    `p` with different seeds a total of `n_trials` times. The query() should equal
    the number of unique keys that were added to the HyperLogLog.

    Parameters
    ----------
    p: int, optional
        Precision of the HyperLogLog. Default is 10 which is relatively
        low precision, but this is so we can easily exceed the sub-algorithm
        threshold of 5 * 2**p when we use the estimated bias correction.
    n_keys: int, optional
        Number of random keys to generate. Default is 100,000. We create
        keys that are random 16-byte values so this should be the number
        of unique keys, but we do check against only the number of unique
        keys which may be slightly less than n_keys.
    n_trials: int, optional
        The number of independent times to measure the error. Each run
        uses a different seed to the HyperLogLog. Default is 100

    """
    # Create n_keys random 16-byte keys.
    keys_r = np.random.randint(0, 256, (n_keys, 16)).astype(np.uint8)
    # Remove any possible duplicates
    keys = list(set([bytes(k) for k in keys_r]))
    n_keys = len(keys)

    error = np.zeros(n_trials)
    for n in range(n_trials):
        hll = HyperLogLog(p, n)
        hll.update(keys)
        error[n] = n_keys - hll.query()

    if error.std() == 0.0:
        assert error.mean() == 0.0, f"std = 0, but mean !=0, {error.mean():.3f}"
    else:
        t_value = np.abs(error.mean() / (error.std() / np.sqrt(n_trials)))
        # 99% confidence level
        assert t_value < 2.576, f"t-value {t_value:.4} is above 2.576"


def test_hll_merge(
    p: int = 10, n_keys1: int = 100000, n_keys2: int = 25000, n_trials: int = 100
):
    """
    Uses a t-test to test the null hypothesis that the mean of the
    difference between the true count and estimated count is 0. Uses a
    confidence level of 99% to reject the null hypotheses. The test asserts
    that we should fail to reject the null hypothesis.

    The HyperLogLog's estimated cardinality "is asymptotically almost unbiased".
    Despite this, even low values typically pass the test.

    We insert `n_keys1` random keys into the first HyperLogLog and `n_keys2`
    random keys into a second HyperLogLog. We merge the second into the first
    and then check that the estimated count is equal to the size of the union
    of the two sets of random keys.

    Parameters
    ----------
    p: int, optional
        Precision of the HyperLogLog. Default is 10 which is relatively low
        precision, but this is so we can easily exceed the sub-algorithm threshold
        of `5 * 2\*\*p` when we use the estimated bias correction.
    n_keys1: int, optional
        Number of keys to add to the first HyperLogLog. Default is 100k.
    n_keys2: int, optional
        Number of keys to add to the second HyperLogLog. Default is 25k.
    n_trials: int, optional
        The number of independent times to measure the error. Each run uses a
        different seed to the HyperLogLogs. Default is 100

    """
    # Create random 16-byte keys
    keys1 = [bytes(r) for r in np.random.randint(0, 256, (n_keys1, 16), dtype=np.uint8)]
    keys1 = list(set(keys1))
    n_keys1 = len(keys1)

    keys2 = [bytes(r) for r in np.random.randint(0, 256, (n_keys2, 15), dtype=np.uint8)]
    keys2 = keys2 + keys1
    keys2 = list(set(keys2))
    n_keys2 = len(keys2)

    n_true = len(set(keys1).union(set(keys2)))

    error = np.zeros(n_trials)
    for n in range(n_trials):
        hll1 = HyperLogLog(p, n)
        hll2 = HyperLogLog(p, n)
        hll1.update(keys1)
        hll2.update(keys2)
        hll1.merge(hll2)
        error[n] = hll1.query() - n_true

    if error.std() == 0.0:
        assert error.mean() == 0.0, f"std = 0, but mean !=0, {error.mean():.3f}"
    else:
        t_value = np.abs(error.mean() / (error.std() / np.sqrt(n_trials)))
        # 99% confidence level
        assert t_value < 2.576, f"t-value {t_value:.4} is above 2.576"
