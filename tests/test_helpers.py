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
from typing import Iterable
import numpy as np
from pathlib import Path

from sketchnu.countmin import CountMin
from sketchnu.heavyhitters import HeavyHitters
from sketchnu.hyperloglog import HyperLogLog
from sketchnu.helpers import parallel_add
from .zipf import zipf


vocab_size = 1000
vocab = list(
    set([bytes(k) for k in np.random.randint(0, 256, (vocab_size, 8)).astype(np.uint8)])
)
zipf_p = zipf(vocab_size)
n_data = 25000
data_stream1 = np.random.choice(vocab, n_data, p=zipf_p).tolist()
data_stream2 = np.random.choice(vocab, n_data, p=zipf_p).tolist()


def process_q_item(q_item: Iterable, batch_size: int):
    for i in range(0, len(q_item), batch_size):
        yield q_item[i : i + batch_size]


def test_parallel_add(tmp_path: Path, batch_size: int = 100):
    """
    Test that parallel processing provides the same answers as single threaded. For cms
    we make it wide enough so that the answers are the same for all keys. For hh, the
    order in which things are added can change the values in lhh_count, so we just test
    that we get the same set of keys from parallel & single threaded. We also in sure
    that the n_records() in cms & hh are properly recorded.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory to write the log file
    batch_size : int, optional
        A record is a batch from the data stream of this size. Default is 100
    """
    cms_args = {"cms_type": "linear", "width": 10000, "depth": 8}
    hh_args = {"width": 70, "depth": 4, "max_key_len": 8}
    hll_args = {"p": 16}

    true_count = Counter(data_stream1)
    true_count.update(data_stream2)
    n_records = int(n_data * 2 / batch_size)

    cms, hh, hll = parallel_add(
        [data_stream1, data_stream2],
        process_q_item,
        n_workers=2,
        cms_args=cms_args,
        hh_args=hh_args,
        hll_args=hll_args,
        log_file=tmp_path / "sketchnu.log",
        batch_size=batch_size,
    )

    cms_single = CountMin(**cms_args)
    cms_single.update(data_stream1)
    cms_single.update(data_stream2)
    assert cms.n_records() == n_records
    for i, v in enumerate(vocab):
        assert cms[v] == cms_single[v], f"Failed on {i+1} out of {len(vocab)}, {cms[v]}"

    hh_single = HeavyHitters(**hh_args)
    hh_single.update(data_stream1)
    hh_single.update(data_stream2)
    hh_topk = set([k for k, _ in hh.query(500)])
    hh_single_topk = set([k for k, _ in hh_single.query(500)])
    assert hh.n_records() == n_records
    assert hh_topk == hh_single_topk

    hll_single = HyperLogLog(**hll_args)
    hll_single.update(data_stream1)
    hll_single.update(data_stream2)
    assert hll.query() == hll_single.query()
