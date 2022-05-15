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

from sketchnu.heavyhitters import HeavyHitters
from sketchnu.countmin import CountMin
from .zipf import zipf


vocab_size = 50000
vocab = list(
    set([bytes(k) for k in np.random.randint(0, 256, (vocab_size, 8)).astype(np.uint8)])
)
zipf_p = zipf(vocab_size)
n_data = 50000
data_stream = np.random.choice(vocab, n_data, p=zipf_p).tolist()


def test_n_added(width: int = 100, depth: int = 4, max_key_len: int = 8):
    """
    Test that number of elements added to the sketch is properly tracked

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 100
    depth : int, optional
        Depth of the sketch. Default is 4
    max_key_len : int, optional
        Maximum length of any key inserted into the sketch. Default is 8
    """
    hh = HeavyHitters(width, depth, max_key_len)
    hh.update(data_stream)
    assert hh.n_added() == n_data


def test_update_theory(width: int = 500, depth: int = 4, max_key_len: int = 8):
    """
    Tests both lemmas in the original paper (3.4.3 & 3.4.4). This uses the theoretical
    work which includes using a count-min sketch to estimate frequency of occurrence.
    In practice, following advice from the paper, you can drop the count-min sketch and
    use the lhh_count as an estimate. The lhh_count <= true_count <= cms estimate

    Lemma 3.4.3
    -----------
    Topkapi with depth = log(2/delta) & width = 1/eps where eps < phi misses to report
    key with frequency >= phi * N with probability at most delta / 2

    Lemma 3.4.4
    -----------
    Topkapi with depth = log(2/delta) & width = 1/eps where eps < phi reports key with
    frequency <= (phi - eps) * N with probability at most delta / 2

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 500
    depth : int, optional
        Depth of the sketch. Default is 4
    max_key_len : int, optional
        Maximum length of any key inserted into the sketch. Default is 8
    """
    eps = 1 / width
    phi = 1.1 * eps
    threshold = int(n_data * phi)
    failure_rate = np.exp(-depth)
    failure_miss = []
    failure_reports = []

    hh = HeavyHitters(width, depth, max_key_len, phi)
    cms = CountMin("linear", width, depth)

    hh.update(data_stream)
    cms.update(data_stream)
    true_count = Counter(data_stream)

    # This is the full Topkapi algorithm. See Sec. 3.2 of the paper
    # Consider the union of all keys in LHH as candidate set. Estimate their counts
    # using the CMS & report all elements with count higher than phi * N
    hh.generate_candidate_set(0)  # Puts union of all keys into candidate_set
    full_approx = Counter()
    for k, v in hh.candidate_set.items():
        if cms[k] >= threshold:
            full_approx[k] = cms[k]

    for v in true_count:
        # Misses to report when it should
        if true_count[v] >= (phi * n_data):
            if v in full_approx:
                failure_miss.append(0)
            else:
                failure_miss.append(1)
        # Reports when it shouldn't
        if true_count[v] <= (phi - eps) * n_data:
            if v in full_approx:
                failure_reports.append(1)
            else:
                failure_reports.append(0)

    if len(failure_miss) > 0:
        assert np.mean(failure_miss) <= failure_rate
    if len(failure_reports) > 0:
        assert np.mean(failure_reports) <= failure_rate


def test_update_ideal(width: int = 2500, depth: int = 4, max_key_len: int = 8):
    """
    Setting width >> vocab_size to make sure the machinery is working as expected.

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 2500
    depth : int, optional
        Depth of the sketch. Default is 4
    max_key_len : int, optional
        Maximum length of any key inserted into the sketch. Default is 8
    """
    vocab_size = 25
    vocab = list(
        set(
            [
                bytes(k)
                for k in np.random.randint(0, 256, (vocab_size, 8)).astype(np.uint8)
            ]
        )
    )
    zipf_p = zipf(vocab_size)
    n_data = 50000
    data_stream = np.random.choice(vocab, n_data, p=zipf_p).tolist()

    hh = HeavyHitters(width, depth, max_key_len)
    hh.update(data_stream)
    true_count = Counter(data_stream)

    for key, value in true_count.items():
        assert value == hh[key]


def test_update(width: int = 500, depth: int = 4, max_key_len: int = 8):
    """
    Test that keys returned by HeavyHitters.query() all appear within the set of
    keys where the true count of each is >= n_data / width.

    Because HeavyHitters uses the lhh_count instead of count-min sketch, following the
    practical advice in the paper, when HeavyHitters keeps only keys with lhh_count >=
    threshold the true count of those keys is greater than or equal to the sketch's
    estimate. So this means that we aren't getting all the keys we should and thus all
    of the sketch's keys should be found in the set of keys whose true counts are
    greater than or equal to the threshold.

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 500
    depth : int, optional
        Depth of the sketch. Default is 4
    max_key_len : int, optional
        Maximum length of any key inserted into the sketch. Default is 8
    """
    eps = 1 / width
    threshold = int(n_data * eps)

    hh = HeavyHitters(width, depth, max_key_len)

    hh.update(data_stream)
    true_count = Counter(data_stream)

    hh_set = set([k for k, _ in hh.query(vocab_size)])
    true_set = set([k for k, v in true_count.items() if v >= threshold])

    assert hh_set.intersection(true_set) == hh_set


def test_merge_ideal(width: int = 2500, depth: int = 4, max_key_len: int = 8):
    """
    Add same data stream into two sketches. Merge them.

    Setting width >> vocab_size to make sure the machinery is working as expected.

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 2500
    depth : int, optional
        Depth of the sketch. Default is 4
    max_key_len : int, optional
        Maximum length of any key inserted into the sketch. Default is 8
    """
    vocab_size = 25
    vocab = list(
        set(
            [
                bytes(k)
                for k in np.random.randint(0, 256, (vocab_size, 8)).astype(np.uint8)
            ]
        )
    )
    zipf_p = zipf(vocab_size)
    n_data = 50000
    data_stream = np.random.choice(vocab, n_data, p=zipf_p).tolist()

    hh1 = HeavyHitters(width, depth, max_key_len)
    hh2 = HeavyHitters(width, depth, max_key_len)

    hh1.update(data_stream)
    hh2.update(data_stream)
    true_count = Counter(data_stream)
    true_count.update(data_stream)

    hh1.merge(hh2)
    for key, value in true_count.items():
        assert value == hh1[key]


def test_merge_mixed_key_len(width: int = 2500, depth: int = 4, max_key_len: int = 8):
    """
    Test that we can handle the merging of different length keys.

    Setting width >> vocab_size to make sure the machinery is working as expected.

    Parameters
    ----------
    width : int, optional
        Width of the sketch. Default is 2500
    depth : int, optional
        Depth of the sketch. Default is 4
    max_key_len : int, optional
        Maximum length of any key inserted into the sketch. Default is 8
    """
    assert max_key_len - 3 > 0, "max_key_len is too small must be > 3"

    vocab_size = 25
    vocab = list(
        set(
            [
                bytes(k)
                for k in np.random.randint(0, 256, (vocab_size, 8)).astype(np.uint8)
            ]
        )
    )
    zipf_p = zipf(vocab_size)
    n_data = 50000
    data_stream = np.random.choice(vocab, n_data, p=zipf_p).tolist()

    vocab2 = list(
        set(
            [
                bytes(k)
                for k in np.random.randint(
                    0, 256, (vocab_size, max_key_len - 3)
                ).astype(np.uint8)
            ]
        )
    )
    zipf_p2 = zipf(vocab_size)
    data_stream2 = np.random.choice(vocab2, n_data, p=zipf_p2).tolist()

    hh1 = HeavyHitters(width, depth, max_key_len)
    hh2 = HeavyHitters(width, depth, max_key_len)

    hh1.update(data_stream)
    hh2.update(data_stream2)
    true_count = Counter(data_stream)
    true_count.update(data_stream2)

    hh1.merge(hh2)
    for key, value in true_count.items():
        assert value == hh1[key]

    # Reverse the ordering
    hh1 = HeavyHitters(width, depth, max_key_len)
    hh2 = HeavyHitters(width, depth, max_key_len)

    hh1.update(data_stream2)
    hh2.update(data_stream)

    hh1.merge(hh2)
    for key, value in true_count.items():
        assert value == hh1[key]


def test_saving(
    tmp_path: Path,
    width: int = 500,
    depth: int = 4,
    max_key_len: int = 8,
    topk: int = 10,
):
    """
    Test saving and loading the sketch

    Parameters
    ----------
    tmp_path : Path
        This is a pytest fixture which will provide a temporary directory unique
        to the test invocation, created in the base temporary directory
    width : int, optional
        Width of the sketch. Default is 500
    depth : int, optional
        Depth of the sketch. Default is 4
    max_key_len : int, optional
        Maximum length of any key inserted into the sketch. Default is 8
    topk : int, optional
        Query for the topk most frequent. Default is 10
    """
    hh = HeavyHitters(width, depth, max_key_len)
    hh.update(data_stream)
    filename = tmp_path / "test_hh.npz"

    hh.save(filename)
    hh_load = HeavyHitters.load(filename)

    for orig, saved in zip(hh.query(topk), hh_load.query(topk)):
        assert orig == saved
