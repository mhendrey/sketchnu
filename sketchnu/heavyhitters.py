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


Numba implementation of the topkapi algorithm from

A. Mandal, H. Jiang, A. Shrivastava, and V. Sarkar, "Topkapi: Parallel and Fast
Sketches for Finding Top-K Frequent Elements", Advances in Neural Information
Processing Systems **31**, (2018).

"""
from collections import Counter
import gc
from multiprocessing.shared_memory import SharedMemory
from time import sleep
from numba import njit, uint8, uint32, uint64, types
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

from sketchnu.hashes import fasthash64


@njit(
    types.void(
        uint8[:, :, :],
        uint32[:, :],
        uint8[:, :],
        uint64[:],
        uint64,
        uint64,
        uint64,
        uint32,
        types.Bytes(uint8, 1, "C"),
    )
)
def _add(
    lhh,
    lhh_count,
    key_lens,
    n_added_records,
    width,
    depth,
    max_key_len,
    uint_maxval,
    key,
):
    """
    Numba function that adds `key` into the sketch
    """
    key_len = np.uint64(len(key))

    # Handle different key sizes
    if key_len == max_key_len:
        key_array = np.frombuffer(key, uint8)
    elif key_len < max_key_len:
        key_array = np.zeros(max_key_len, uint8)
        key_array[:key_len] = np.frombuffer(key, uint8)
    # Only use the first max_key_len bytes if key is too long
    else:
        key = key[:max_key_len]
        key_len = max_key_len
        key_array = np.frombuffer(key, uint8)

    n_added_records[0] += uint64(1)
    for row in range(depth):
        col = fasthash64(key, row) % width
        if np.all(key_array == lhh[row, col]):
            if lhh_count[row, col] < uint_maxval:
                lhh_count[row, col] += uint32(1)
        elif lhh_count[row, col] == uint32(0):
            lhh[row, col, :] = key_array
            lhh_count[row, col] += uint32(1)
            key_lens[row, col] = uint8(key_len)
        else:
            lhh_count[row, col] -= uint32(1)


@njit(
    types.void(
        uint8[:, :, :],
        uint32[:, :],
        uint8[:, :],
        uint64[:],
        uint64,
        uint64,
        uint64,
        uint32,
        types.Bytes(uint8, 1, "C"),
        uint64,
    )
)
def _add_ngram(
    lhh,
    lhh_count,
    key_lens,
    n_added_records,
    width,
    depth,
    max_key_len,
    uint_maxval,
    key,
    ngram,
):
    """
    Numba function that shingles `key` into ngrams of size `ngram` and then adds each
    ngram into the sketch
    """
    key_len = np.uint64(len(key))
    if key_len <= ngram:
        _add(
            lhh,
            lhh_count,
            key_lens,
            n_added_records,
            width,
            depth,
            max_key_len,
            uint_maxval,
            key,
        )
    else:
        for i in range(key_len - (ngram - uint64(1))):
            _add(
                lhh,
                lhh_count,
                key_lens,
                n_added_records,
                width,
                depth,
                max_key_len,
                uint_maxval,
                key[i : i + ngram],
            )


@njit(
    types.void(
        uint8[:, :, :],
        uint32[:, :],
        uint8[:, :],
        uint64[:],
        uint64,
        uint64,
        uint32,
        uint8[:, :, :],
        uint32[:, :],
        uint8[:, :],
        uint64[:],
    )
)
def _merge(
    lhh,
    lhh_count,
    key_lens,
    n_added_records,
    width,
    depth,
    uint_maxval,
    other_lhh,
    other_lhh_count,
    other_key_lens,
    other_n_added_records,
):
    """
    Numba function to merge the second heavy hitter sketch into the first
    """
    for row in range(depth):
        for col in range(width):
            keys_match = (np.all(lhh[row, col] == other_lhh[row, col])) and (
                key_lens[row, col] == other_key_lens[row, col]
            )
            if keys_match:
                if other_lhh_count[row, col] > uint_maxval - lhh_count[row, col]:
                    lhh_count[row, col] = uint_maxval
                else:
                    lhh_count[row, col] += other_lhh_count[row, col]
            else:
                if lhh_count[row, col] >= other_lhh_count[row, col]:
                    lhh_count[row, col] -= other_lhh_count[row, col]
                else:
                    lhh[row, col] = other_lhh[row, col]
                    key_lens[row, col] = other_key_lens[row, col]
                    lhh_count[row, col] = (
                        other_lhh_count[row, col] - lhh_count[row, col]
                    )

    # Merge the special counters
    n_added_records[0] += other_n_added_records[0]
    n_added_records[1] += other_n_added_records[1]


@njit(
    uint32(
        uint8[:, :, :],
        uint32[:, :],
        uint64,
        uint64,
        uint64,
        types.Bytes(uint8, 1, "C"),
        uint8,
    )
)
def _max_count(lhh, lhh_count, width, depth, max_key_len, key, key_len):
    """
    Numba function to provide the estimated count for the given `key`
    """
    if key_len == max_key_len:
        key_array = np.frombuffer(key, uint8)
    else:
        key_array = np.zeros(max_key_len, uint8)
        key_array[:key_len] = np.frombuffer(key, uint8)

    max_count = uint32(0)
    for row in range(depth):
        col = fasthash64(key, row) % width
        if np.all(key_array == lhh[row, col]) and lhh_count[row, col] > max_count:
            max_count = lhh_count[row, col]

    return max_count


class HeavyHitters:
    """
    Sketch implementation of the phi-heavy hitters algorithm which identifies all the
    keys in a data stream that are observed in at least phi fraction of the records.
    This assumes that keys have a fat-tailed distribution in the data stream. This is
    an implementation of the Topkapi algorithm.

    The parameter phi must be greater than 1 / width of the sketch for the theoretical
    guarantees to be valid. The theoretical guarantees use a count-min sketch to
    estimate the frequency of any given key. For practical reasons, the paper suggests
    dropping the count-min sketch to save space. In this case you use the lhh_count as
    an estimate of the frequency of occurence for a given key. Note that
    lhh_count <= true count <= cms. So when we call the query function we are doing a
    more conservative estimate which corresponds to a higher phi. If you are feeling
    like you want to squeeze more out of the sketch you can provide a lower threshold
    (= phi * n_added()) when calling the query() function.

    Parameters
    ----------
    width : int
        Width of the heavy hitters sketch. Must be non-negative
    depth : int, optional
        Depth of the heavy hitters sketch. Must be non-negative. Default is 4
    max_key_len : int, optional
        Maximum number of bytes any given key may have. Must be less than 256.
        Default is 16
    phi : float, optional
        When generating the candidate set of heavy hitters, only keys whose estimated
        frequency of occurrence (lhh_count) >= phi * n_added() will be added to the
        candidate set. Default of None is set to 1 / width.
    shared_memory : bool, optional
        If True, then sketch is placed in shared memory. Needed if performing
        multiprocessing as sketchnu.helpers.parallel_add() does. Default is False

    Attributes
    ----------
    width : np.uint64
        Width of the 2-d array of counters of the sketch
    depth : np.uint64
        Depth of the 2-d array of counters of the sketch
    max_key_len : np.uint64
        Maximum number of bytes any given key may have. Must be less than 256
    phi : np.float64
        When generating the candidate set of heavy hitters, only keys whose estimated
        frequency of occurrence (lhh_count) >= phi * n_added() will be added to the
        candidate set. Default of None is set to 1 / width.
    lhh : np.ndarray, shape=(depth, width, max_key_len), dtype=np.uint8
        Storing the keys associated with each bucket in the 2-d array of counters. Keys
        are stored as numpy arrays, as opposed to 2-d list of bytes, in order for numba
        to be able to process them. If a given key has fewer bytes than max_len_key,
        then right padded with 0s.
    lhh_count : np.ndarray, shape=(depth, width), dtype=np.uint32
        Store the counts associated with keys stored in lhh.
    key_lens : np.ndarray, shape=(depth, width), dtype=np.uint8
        The length of each of the keys stored in lhh
    n_added_records : np.ndarray, shape=(2,), dtype=np.uint64
        1-d array that holds two special counters. The first is the number of elements
        that have been added to the sketch. Useful for calculating error limits. The
        second is used by helpers.parallel_add() to keep track of the number of records
        that have been processed.
    """

    def __init__(
        self,
        width: int,
        depth: int = 4,
        max_key_len: int = 16,
        phi: float = None,
        shared_memory: bool = False,
    ) -> None:
        """
        Initialize a heavy hitters sketch

        Parameters
        ----------
        width : int
            Width of the heavy hitters sketch. Must be non-negative
        depth : int, optional
            Depth of the heavy hitters sketch. Must be non-negative. Default is 4
        max_key_len : int, optional
            Maximum number of bytes any given key may have. Must be less than 256.
            Default is 16
        phi : float, optional
            When generating the candidate set of heavy hitters, only keys whose
            estimated frequency of occurrence (lhh_count) >= phi * n_added() will be
            added to the candidate set. Default of None is set to 1 / width.
        shared_memory : bool, optional
            If True, then sketch is placed in shared memory. Needed if performing
            multiprocessing as sketchnu.helpers.parallel_add() does. Default is False

        Returns
        -------
        HeavyHitters
        """
        int_types = (int, np.uint8, np.int8, np.uint32, np.int32, np.uint64, np.int64)
        float_types = (float, np.float32, np.float64)
        if width <= 0 or not isinstance(width, int_types):
            raise ValueError(f"{width=:}. Must be an integer greater than 0")
        if depth <= 0 or not isinstance(depth, int_types):
            raise ValueError(f"{depth=:}. Must be an integer greater than 0")
        if (
            max_key_len <= 0
            or not isinstance(max_key_len, int_types)
            or max_key_len > 255
        ):
            raise ValueError(f"{max_key_len=:}. Must be an integer [1, 255]")
        if phi is not None and not isinstance(phi, float_types):
            raise ValueError(f"{phi=:}. Must be None or a positive float")
        if isinstance(phi, float_types) and (phi <= 0.0 or phi >= 1.0):
            raise ValueError(f"{phi=:}. Must be float between (0.0, 1.0)")
        if not isinstance(shared_memory, bool):
            raise ValueError(f"{type(shared_memory)=:}. Must be a boolean")

        self.width = np.uint64(width)
        self.depth = np.uint64(depth)
        self.max_key_len = np.uint64(max_key_len)
        if phi is None:
            self.phi = np.float64(1.0 / self.width)
        else:
            self.phi = np.float64(phi)
        self.uint_maxval = np.uint32(2 ** 32 - 1)

        self.args = {
            "width": width,
            "depth": depth,
            "max_key_len": max_key_len,
            "phi": phi,
        }

        # Store the value of n_added_records[0] when query was last run
        self.candidate_set = Counter()
        self.n_added_sort = 0
        self.threshold_sort = np.uint32(0)

        # Number of bytes needed for lhh, lhh_count, n_added_records
        lhh_nbytes = int(max_key_len * width * depth)
        lhh_count_nbytes = int(4 * width * depth)
        key_lens_nbytes = int(1 * width * depth)
        n_added_nbytes = 8 * 2

        if shared_memory:
            self.shm = SharedMemory(
                create=True,
                size=(lhh_nbytes + lhh_count_nbytes + key_lens_nbytes + n_added_nbytes),
            )
            start = 0
            end = lhh_nbytes
            self.lhh = np.frombuffer(self.shm.buf[start:end], np.uint8).reshape(
                self.depth, self.width, self.max_key_len
            )
            start = end
            end += lhh_count_nbytes
            self.lhh_count = np.frombuffer(self.shm.buf[start:end], np.uint32,).reshape(
                self.depth, self.width
            )
            start = end
            end += key_lens_nbytes
            self.key_lens = np.frombuffer(self.shm.buf[start:end], np.uint8,).reshape(
                self.depth, self.width
            )
            start = end
            self.n_added_records = np.frombuffer(self.shm.buf[start:], np.uint64,)
        else:
            self.lhh = np.zeros((self.depth, self.width, self.max_key_len), np.uint8)
            self.lhh_count = np.zeros((self.depth, self.width), np.uint32)
            self.key_lens = np.zeros((self.depth, self.width), np.uint8)
            self.n_added_records = np.zeros(2, np.uint64)

    def query(self, k: int, threshold: int = None) -> List[Tuple[bytes, int]]:
        """
        Return the top `k` heavy hitters. If new data has been added or if `threshold`
        is different from the last time a candidate set was generated, then this will
        generate a new candidate set before selecting the top `k`.

        Parameters
        ----------
        k : int
        threshold : int, optional
            Only include keys from lhh whose lhh_counts >= `threshold`. Default is None
            which then sets threshold to self.phi * self.n_added()
        
        Returns
        -------
        List[Tuple[bytes, int]]
            Sorted list of the (key, count) of the top `k`. Format is the same as
            collections.Counter().most_common().
        """
        if threshold is None:
            threshold = np.uint32(self.phi * self.n_added())
        else:
            threshold = np.uint32(threshold)

        if (self.n_added_sort < self.n_added()) or (self.threshold_sort != threshold):
            self.generate_candidate_set(threshold)

        return self.candidate_set.most_common(k)

    def add(self, key: bytes) -> None:
        """
        Add a single `key` to the heavy hitters sketch and update the counter tracking
        total number of keys added to the sketch.

        Parameters
        ----------
        key : bytes
            Element to be added to the sketch
        
        Returns
        -------
        None
        """
        _add(
            self.lhh,
            self.lhh_count,
            self.key_lens,
            self.n_added_records,
            self.width,
            self.depth,
            self.max_key_len,
            self.uint_maxval,
            key,
        )

    def update(self, keys: List[bytes]) -> None:
        """
        Add a list of `keys` to the sketch. This follows the convention of
        collections.Counter

        Parameters
        ----------
        keys : List[bytes]
            List of elements to add to the sketch

        Returns
        -------
        None

        """
        for key in keys:
            self.add(key)

    def add_ngram(self, key: bytes, ngram: int) -> None:
        """
        Take a given `key` and shingle it into ngrams of size `ngram` and then
        add the ngrams to the sketch. If the `key` length is less than `ngram`
        then add the whole `key`

        Parameters
        ----------
        key : bytes
            Element to be shingled before adding to the sketch
        ngram : int
            ngram size
        
        Returns
        -------
        None

        """
        ngram = np.uint64(ngram)
        _add_ngram(
            self.lhh,
            self.lhh_count,
            self.key_lens,
            self.n_added_records,
            self.width,
            self.depth,
            self.max_key_len,
            self.uint_maxval,
            key,
            ngram,
        )

    def update_ngram(self, keys: List[bytes], ngram: int) -> None:
        """
        Given a list of keys, shingle each into ngrams of size `ngram`, and then
        add them to the sketch.

        Parameters
        ----------
        keys : List[bytes]
            List of elements to be shingled before adding to the sketch.
        ngram : int
            ngram size

        Returns
        -------
        None

        """
        # Loop through the keys
        for key in keys:
            self.add_ngram(key, ngram)

    def merge(self, other) -> None:
        """
        Merge the HeavyHitter sketch `other` into this one.

        Parameters
        ----------
        other : HeavyHitters
            Another HeavyHitters with the same width, depth, max_key_len.
        
        Returns
        -------
        None

        Raises
        ------
        TypeError
            If `other` has different width, depth, or max_key_len

        """
        if (
            self.width != other.width
            or self.depth != other.depth
            or self.max_key_len != other.max_key_len
        ):
            raise TypeError("self and other have different width | depth | max_key_len")

        _merge(
            self.lhh,
            self.lhh_count,
            self.key_lens,
            self.n_added_records,
            self.width,
            self.depth,
            self.uint_maxval,
            other.lhh,
            other.lhh_count,
            other.key_lens,
            other.n_added_records,
        )

    def save(self, filename: Union[str, Path]) -> None:
        """
        Save the sketch to `filename` adding the .npz extension if not already part of
        `filename`
        
        Parameters
        ----------
        filename: str | Path
            File to save the sketch to disk. This will be a .npz file.
        
        Returns
        -------
        None

        """
        np.savez(
            filename,
            args=np.array(
                [self.width, self.depth, self.max_key_len, self.phi], np.float64
            ),
            lhh=self.lhh,
            lhh_count=self.lhh_count,
            key_lens=self.key_lens,
            n_added_records=self.n_added_records,
        )

    @staticmethod
    def load(filename: Union[str, Path], shared_memory: bool = False):
        """
        Load a saved HeavyHitters stored in `filename`

        Parameters
        ----------
        filename : str | Path
            File path to the saved .npz file
        shared_memory : bool, optional
            If True, load into shared memory. Default is False.
        
        Returns
        -------
        HeavyHitters
        """
        with np.load(filename) as npzfile:
            args = npzfile["args"]
            width = np.uint64(args[0])
            depth = np.uint64(args[1])
            max_key_len = np.uint64(args[2])
            phi = np.float64(args[3])
            hh = HeavyHitters(
                width, depth, max_key_len, phi, shared_memory=shared_memory
            )
            np.copyto(hh.lhh, npzfile["lhh"])
            np.copyto(hh.lhh_count, npzfile["lhh_count"])
            np.copyto(hh.key_lens, npzfile["key_lens"])
            np.copyto(hh.n_added_records, npzfile["n_added_records"])

        hh.generate_candidate_set()

        return hh

    def attach_existing_shm(self, existing_shm_name: str) -> None:
        """
        Attach this sketch to an existing shared memory block. Useful when working
        within a spawned child process. This creates self.existing_shm which gets
        closed when self.__del__ gets called.

        Parameters
        ----------
        existing_shm_name : str
            Name of an existing shared memory block to attach this sketch to
        
        Returns
        -------
        None
        """
        existing_shm = SharedMemory(name=existing_shm_name)

        start = 0
        end = self.lhh.nbytes
        self.lhh = np.frombuffer(existing_shm.buf[start:end], np.uint8).reshape(
            self.depth, self.width, self.max_key_len
        )
        start = end
        end += self.lhh_count.nbytes
        self.lhh_count = np.frombuffer(existing_shm.buf[start:end], np.uint32,).reshape(
            self.depth, self.width
        )
        start = end
        end += self.key_lens.nbytes
        self.key_lens = np.frombuffer(existing_shm.buf[start:end], np.uint8,).reshape(
            self.depth, self.width
        )
        start = end
        self.n_added_records = np.frombuffer(existing_shm.buf[start:], np.uint64,)

        # Now create class member to hold this so __del__ can clean up for us
        self.existing_shm = existing_shm

    def n_added(self) -> np.uint64:
        """
        This special counter is used to track the total number of elements
        that have been added to the sketch. Useful to check the error guarantees.

        Returns
        -------
        np.uint64
            The number of elements that have been added to the sketch.

        """
        return self.n_added_records[0]

    def n_records(self) -> np.uint64:
        """
        This special counter is used by the sketchnu.helpers.parallel_add() to
        keep track of the number of records that have been added to the sketch.

        Returns
        -------
        np.uint64
            The number of records that have been added to the sketch.

        """
        return self.n_added_records[1]

    def generate_candidate_set(self, threshold: int = None) -> None:
        """
        Generate a candidate set of heavy hitters. Only keys in `lhh` whose
        corresponding counts in `lhh_count` are greater the `threshold` are included.
        Contrary to the paper, we all the rows instead of just the first one. This
        seems like a small price to pay to not lose candidates due to hash collisions.
        The candidate set is a collections.Counter stored in self.candidate_set

        Parameters
        ----------
        threshold : int, optional
            If None (default), then uses `threshold` provided during instantiation
        
        Returns
        -------
        None
        """
        if threshold is None:
            threshold = np.uint32(self.phi * self.n_added())
        else:
            threshold = np.uint32(threshold)

        self.n_added_sort = self.n_added()
        self.threshold_sort = threshold
        self.candidate_set = Counter()

        # Generate candidate list
        for row in range(self.depth):
            for column in range(self.width):
                # No key associated with this column, so skip
                if self.lhh_count[row, column] == 0:
                    continue

                key_len = self.key_lens[row, column]
                key = bytes(self.lhh[row, column, :key_len])
                # Only need to do this once for each key
                if self.candidate_set[key] == 0:
                    max_count = _max_count(
                        self.lhh,
                        self.lhh_count,
                        self.width,
                        self.depth,
                        self.max_key_len,
                        key,
                        key_len,
                    )

                    if max_count >= threshold:
                        self.candidate_set[key] = max_count

    def __getitem__(self, key: bytes) -> int:
        """
        Return estimated number of times key was observed in the stream

        Parameters
        ----------
        key : bytes

        Returns
        -------
        int
        """
        key_len = len(key)
        max_count = _max_count(
            self.lhh,
            self.lhh_count,
            self.width,
            self.depth,
            self.max_key_len,
            key,
            key_len,
        )
        return max_count

    def __del__(self):
        try:
            if self.shm:
                try:
                    # Need to explicitly del the arrays since they are sharing the
                    # memory block. Without this you get the MemoryError
                    # "cannot close exported pointers exist"
                    del self.lhh
                    del self.lhh_count
                    del self.key_lens
                    del self.n_added_records
                    gc.collect()
                    sleep(0.25)
                    self.shm.close()
                    self.shm.unlink()
                except Exception as exc:
                    raise MemoryError(f"Failed to close & unlink: {exc}")
        except AttributeError:
            pass

        try:
            if self.existing_shm:
                try:
                    del self.lhh
                    del self.lhh_count
                    del self.key_lens
                    del self.n_added_records
                    gc.collect()
                    sleep(0.25)
                    self.existing_shm.close()
                except Exception as exc:
                    raise MemoryError(f"Failed to close existing_shm: {exc}")
        except AttributeError:
            pass
