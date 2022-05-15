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


Too many problems with the experimental Numba class. Converting this
to a series of Numba functions and a Python classes.

Implementation of count-min sketch using Numba. Three different types have been coded,
but all use conservative updating to help reduce errors. The three types are

* linear : Uses 32-bit linear counters
* log16 : Uses 16-bit log counters
* log8 : Uses 8-bit log counters

Each type is implemented as a Python class, but uses numba functions under the hood
where possible (and faster). The convenience function, :code:`CountMin()` is the
recommended way to instantiate a count-min sketch. The Python classes are
:code:`CountMinLinear`, :code:`CountMinLog16`, and :code:`CountMinLog8`.

Example
-------
To get a linear CountMin with width = 2\*\*20 ::

    from sketchnu.countmin import CountMin

    width = 2**20
    cms = CountMin("linear", width)
    key = "testing".encode("utf-8")
    cms.add(key)
    # To get an estimated count for that key use either
    cms.query(key)
    cms[key]

Each count-min sketch implementation has two special counters that are stored in the
attribute :code:`n_added_records` which is a 1-d array with two elements. The first
records the total number of elements added to the count-min sketch. This is useful when
calculating the error guarantees. Its value can be retrieved with :code:`n_added()`.
The second element is used by :code:`helpers.parallel_add()` to store the number of
records added to the sketch. This value is needed when using a count-min sketch to
calculate the tf-idf value for a given record. The idf piece needs the number of
records which have a given key, i.e. document frequency. This can be provided directly
by the count-min sketch. But you also need to know the total number of records. This
value can be retrieved by :code:`n_records()` if you use :code:`helpers.parallel_add()`
to create the count-min sketch.
"""
import gc
from multiprocessing.shared_memory import SharedMemory
from numba import njit, uint8, uint16, uint32, uint64, float64, types
import numpy as np
from pathlib import Path
from time import sleep
from typing import List, Union

from sketchnu.hashes import fasthash64


@njit(float64(float64, uint64, uint32, uint32))
def _func(base, max_count, num_reserved, uint_max):
    """
    Numba function used to determine the base of the log needed.
    """
    M = float64(max_count) - float64(num_reserved)
    return base ** (uint_max - num_reserved) - M * base + (M - 1.0)


@njit(float64(float64, uint64, uint32, uint32))
def _funcprime(base, max_count, num_reserved, uint_max):
    """
    Numba function used to determine the base of the log needed.
    """
    M = float64(max_count) - float64(num_reserved)
    return uint_max * base ** (uint_max - num_reserved) - M


@njit(float64(uint64, uint32, uint32))
def _find_base(max_count, num_reserved, uint_max):
    """
    Numba function to calcuate the correct base value for log counters using a simple
    Newton method.

    Parameters
    ----------
    max_count: uint64
        Maximum value we want the counter to return
    num_reserved: uint32
        Use linear counting up to this value. Log8 uses 15
    uint_max: uint32
        Maximum value that can be stored. For log8, this is 255
        For log16, this 65536 (2**16 -1)

    Returns
    -------
    float64
        The base to use for the log counters

    Raises
    ------
    ValueError
        If the base is 1.0

    """
    base = float64(np.exp(np.log(max_count) / (uint_max - num_reserved)))

    for i in range(200):
        base = base - _func(base, max_count, num_reserved, uint_max) / _funcprime(
            base, max_count, num_reserved, uint_max
        )
    if base < 1.000000001:
        raise ValueError("Calculated base is 1.0. Raise max_count")
    return base


@njit(float64(uint16, uint16, float64))
def _counter2value(counter, num_reserved, base):
    """
    Numba function to convert a log counter to its corresponding value. Used by both
    CountMinLog16 and CountMinLog8. For log8, the values just get cast to uint16.

    Parameters
    ----------
    counter : np.uint16
        Current value of the counter
    num_reserved : np.uint16
        Number of values reserved for linear counting. After that use log counting
    base : np.float64
        Log base used by the counter

    Returns
    -------
    float64
        Estimated value stored in the log counter
    """
    if counter <= num_reserved:
        return float64(counter)
    else:
        cprime = float64(counter - num_reserved)
        return (base ** cprime - 1.0) / (base - 1.0) + float64(num_reserved)


@njit(types.Tuple((float64, uint64))(float64[:], uint64))
def _rand(rand_batch, rand_ptr):
    """
    Numba function to retrieve a random value from the `rand_batch`. If the pointer
    `rand_ptr` has reached the end of the batch, then create a new batch of 2,048
    random values. When a new batch is created, `rand_batch` is updated as a
    side-effect.

    Parameters
    ----------
    rand_batch : np.ndarray
        1-d array of 2,048 random values with uniform distribution between [0, 1).
    rand_ptr : np.uint64
        Keeps track of where in the rand_batch you are.
    
    Returns
    -------
    rand_value : np.float64
        Random value taken from `rand_batch` found at position = `rand_ptr`
    rand_ptr : np.uint64
        Updated value of the pointer
    """
    if rand_ptr == uint64(2048):
        rand_batch[:] = np.random.rand(2048)
        rand_ptr = uint64(1)
    else:
        rand_ptr += uint64(1)
    return rand_batch[rand_ptr - uint64(1)], rand_ptr


@njit(
    types.Tuple((types.boolean, uint64))(
        uint16, uint16, uint16, float64, float64[:], uint64
    )
)
def _increase_counter(counter, num_reserved, uint_maxval, base, rand_nums, rand_ptr):
    """
    Numba function to randomly determine whether to increase a log counter based upon
    the current counter value. Used by both CountMinLog16 and CountMinLog8. For log8,
    the values just get cast to uint16. If `rand_ptr` is at the end of `rand_nums`,
    then a new batch of `rand_nums` is generated and `rand_ptr` resets to the
    beginning.

    Parameters
    ----------
    counter : np.uint16
        Current value of the counter
    num_reserved : np.uint16
        Number of values reserved for linear counting. After that use log counting
    uint_maxval : np.uint16
        Maximum value that a counter can store. Just the max value of uint of the
        appropriate type (uint8 | uint16)
    base : np.float64
        Log base used by the counter
    rand_nums : np.ndarray, dtype=np.float64
        1-d array of random values
    rand_ptr : np.uint64
        Stores the pointer location into `rand_nums`
    
    Returns
    -------
    bool
        True if counter should be increased. False if it shouldn't
    rand_ptr : uint64
        Updated pointer location into `rand_nums`
    """
    cprime = float64(counter) - float64(num_reserved)
    if counter >= uint_maxval:
        return False, rand_ptr
    elif cprime < 0:
        return True, rand_ptr
    else:
        rand, rand_ptr = _rand(rand_nums, rand_ptr)
        if rand < base ** (-cprime):
            return True, rand_ptr
        else:
            return False, rand_ptr


@njit(
    uint32(
        uint32[:, :],
        uint64[:],
        uint64,
        uint64,
        uint32,
        types.Bytes(types.uint8, 1, "C"),
    )
)
def _query_linear(cms, buckets, width, depth, uint_maxval, key):
    """
    Numba function to query a CountMinLinear for estimated number of times `key` has
    been added to the sketch. Alters the values stored in `buckets` to be the column
    ids that `key` maps to in the sketch.

    Parameters
    ----------
    cms : np.ndarray, dtype=np.uint32
        2-d array of shape (`depth`, `width`) containing the linear counters.
    buckets : np.ndarray, dtype=np.uint64
        1-d array of shape (`depth`,) holding column ids that `key` maps to in cms.
    width : np.uint64
        Number of columns in `cms`
    depth : np.uint64
        Number of rows in `cms`
    uint_maxval : np.uint32
        Maximum uint32 value. That is 2\*\*32 - 1
    key : bytes
    
    Returns
    -------
    int
        Estimated number of times that `key` has been added to the sketch
    """
    min_count = uint_maxval
    for row in range(depth):
        buckets[row] = fasthash64(key, row) % width
        count = cms[row, buckets[row]]
        if count < min_count:
            min_count = count
    return min_count


@njit(
    types.void(
        uint32[:, :],
        uint64[:],
        uint64[:],
        uint64,
        uint64,
        uint32,
        types.Bytes(types.uint8, 1, "C"),
    )
)
def _add_linear(cms, n_added_records, buckets, width, depth, uint_maxval, key):
    """
    Numba function to add `key` to the CountMinLinear sketch. Uses conservative
    updating, so it first queries the sketch for the estimated count of `key` which
    causes the buckets to be updated too. Updates `n_added_records[0]` which tracks
    the number of elements added to the sketch.

    Parameters
    ----------
    cms : np.ndarray, dtype=np.uint32
        2-d array of shape (`depth`, `width`) containing the linear counters.
    n_added_records : np.ndarray, dtype=np.uint64
        1-d array of shape (2,) holding special counters for number of elements added
        to the sketch and number of records.
    buckets : np.ndarray, dtype=np.uint64
        1-d array of shape (`depth`,) holding column ids that `key` maps to in cms.
    width : np.uint64
        Number of columns in `cms`
    depth : np.uint64
        Number of rows in `cms`
    uint_maxval : np.uint32
        Maximum uint32 value. That is 2\*\*32 - 1
    key : bytes

    Returns
    -------
    None
    """
    # Track total number of elements added to the sketch
    n_added_records[0] += uint64(1)

    # This gets min_count AND updates buckets
    min_count = _query_linear(cms, buckets, width, depth, uint_maxval, key)

    # Now update only those counters that have min_count
    for row in range(depth):
        count = cms[row, buckets[row]]
        if count == min_count and count < uint_maxval:
            cms[row, buckets[row]] += np.uint32(1)


@njit(
    types.void(
        uint32[:, :],
        uint64[:],
        uint64[:],
        uint64,
        uint64,
        uint32,
        types.Bytes(types.uint8, 1, "C"),
        uint64,
    )
)
def _add_ngram_linear(
    cms, n_added_records, buckets, width, depth, uint_maxval, key, ngram
):
    """
    Numba functino to take a given `key`, split it into ngrams of size `ngram`, and add
    the ngrams to a CountMinLinear sketch. If the `key` length is less than `ngram`
    then add the whole `key`.

    Parameters
    ----------
    cms : np.ndarray, dtype=np.uint32
        2-d array of shape (`depth`, `width`) containing the linear counters.
    n_added_records : np.ndarray, dtype=np.uint64
        1-d array of shape (2,) holding special counters for number of elements added
        to the sketch and number of records.
    buckets : np.ndarray, dtype=np.uint64
        1-d array of shape (`depth`,) holding column ids that `key` maps to in cms.
    width : np.uint64
        Number of columns in `cms`
    depth : np.uint64
        Number of rows in `cms`
    uint_maxval : np.uint32
        Maximum uint32 value. That is 2\*\*32 - 1
    key : bytes
        Element to be shingled before adding to the sketch
    ngram : uint64
        ngram size

    Returns
    -------
    None
    """
    key_len = uint64(len(key))
    if key_len <= ngram:
        _add_linear(cms, n_added_records, buckets, width, depth, uint_maxval, key)
    else:
        for i in range(key_len - (ngram - uint64(1))):
            _add_linear(
                cms,
                n_added_records,
                buckets,
                width,
                depth,
                uint_maxval,
                key[i : i + ngram],
            )


@njit(
    types.void(
        uint32[:, :], uint32[:, :], uint64, uint64, uint32, uint64[:], uint64[:],
    )
)
def _merge_linear(
    cms, other_cms, width, depth, uint_maxval, n_added_records, other_n_added_records
):
    """
    Merge the CountMinLinear sketch `other_cms` into the CountMinLinear sketch `cms`.

    Parameters
    ----------
    cms : np.ndarray, dtype=np.uint32
        2-d array of shape (`depth`, `width`) containing the linear counters.
    other_cms : np.ndarray, dtype=np.uint32
        2-d array of shape (`depth`, `width`) containing the linear counters.
    width : np.uint64
        Number of columns in `cms`
    depth : np.uint64
        Number of rows in `cms`
    uint_maxval : np.uint32
        Maximum uint32 value. That is 2\*\*32 - 1
    n_added_records : np.ndarray, dtype=np.uint64
        1-d array of shape (2,) holding special counters for number of elements added
        to the sketch and number of records.
    other_n_added_records : np.ndarray, dtype=np.uint64
        1-d array of shape (2,) holding special counters for number of elements added
        to the sketch and number of records.

    Returns
    -------
    None
    """
    # Merge the count-min sketches
    for row in range(depth):
        for col in range(width):
            if other_cms[row, col] > uint_maxval - cms[row, col]:
                cms[row, col] = uint_maxval
            else:
                cms[row, col] += other_cms[row, col]
    # Merge the special counters
    n_added_records[0] += other_n_added_records[0]
    n_added_records[1] += other_n_added_records[1]


class CountMinLinear:
    """
    Count-min sketch that uses 32-bit linear counters with conservative updating. A
    given element's maximum count is 2\*\*32 - 1

    Parameters
    ----------
    width : int
        Width of the count-min sketch. Must be non-negative
    depth : int, optional
        Depth of the count-min sketch. Must be non-negative. Default is 8
    shared_memory : bool, optional
        If True, then CountMinLinear is placed in shared memory. Needed if
        performing multiprocessing as sketchnu.helpers.parallel_add() does.
        Default is False.

    Attributes
    ----------
    width : np.uint64
        Width of the 2-d array of counters of the count-min sketch
    depth : np.uint64
        Depth of the 2-d array of counters of the count-min sketch
    shared_memory : bool
        Whether `cms` and `n_added_records` are attached to a shared memory block
    cms : np.uint32[:,:]
        2-d array of the counters. Shape = (depth, width)
    n_added_records : np.uint64[:]
        1-d array that holds two special counters. The first is the number of elements
        that have been added to the sketch. Useful for calculating error limits. The
        second is used by helpers.parallel_add() to keep track of the number of records
        that have been processed. Useful if you want to calculate a TF-IDF.
    """

    def __init__(self, width: int, depth: int = 8, shared_memory: bool = False) -> None:
        """
        Initialize a count-min sketch with 32-bit linear counters with conservative
        updating.

        Parameters
        ----------
        width : int
            Width of the count-min sketch. Must be non-negative
        depth : int, optional
            Depth of the count-min sketch. Must be non-negative. Default is 8
        shared_memory : bool, optional
            If True, then CountMinLinear is placed in shared memory. Needed if
            performing multiprocessing as sketchnu.helpers.parallel_add() does.
            Default is False.

        Returns
        -------
        CountMinLinear
        """
        if width <= 0:
            raise ValueError(f"{width=:}. Must be greater than 0")
        if depth <= 0:
            raise ValueError(f"{depth=:}. Must be greater than 0")

        self.width = np.uint64(width)
        self.depth = np.uint64(depth)
        self.uint_maxval = np.uint32(2 ** 32 - 1)

        self.args = {"cms_type": "linear", "width": width, "depth": depth}

        # Stores the column index for a given key. Used frequently, so declaring here
        self.buckets = np.zeros(depth, np.uint64)

        if shared_memory:
            cms_size = int(4 * width * depth)
            n_added_size = 8 * 2
            self.shm = SharedMemory(create=True, size=(cms_size + n_added_size))
            self.cms = np.frombuffer(self.shm.buf[:cms_size], np.uint32).reshape(
                depth, width
            )
            self.n_added_records = np.frombuffer(self.shm.buf[cms_size:], np.uint64)
        else:
            self.cms = np.zeros((depth, width), np.uint32)
            self.n_added_records = np.zeros(2, np.uint64)

    def query(self, key: bytes) -> int:
        """
        Return estimated number of times `key` was added into the count-min sketch

        Parameters
        ----------
        key : bytes
            Element whose estimated count you want returned

        Returns
        -------
        int

        """
        return _query_linear(
            self.cms, self.buckets, self.width, self.depth, self.uint_maxval, key
        )

    def add(self, key: bytes) -> None:
        """
        Add a single key to the count-min sketch and update the counter tracking total
        number of keys added to the count-min sketch. This is in n_added_records[0].

        Parameters
        ----------
        key : bytes
            Element to be added to the sketch

        Returns
        -------
        None
        """
        _add_linear(
            self.cms,
            self.n_added_records,
            self.buckets,
            self.width,
            self.depth,
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
        _add_ngram_linear(
            self.cms,
            self.n_added_records,
            self.buckets,
            self.width,
            self.depth,
            self.uint_maxval,
            key,
            ngram,
        )

    def update_ngram(self, keys: List[bytes], ngram: int) -> None:
        """
        Given a list of keys, split each into ngrams of size `ngram`, and then
        add them to the sketch.

        Note
        ----
        Current implementation loops through the keys in Python. Speed testing
        showed that converting the list to numba's typed list was very costly.
        Maybe if that gets faster a pure numba implementation can be done.

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
        Merge the count-min sketch `other` into this one.

        Parameters
        ----------
        other : CountMinLinear
            Another CountMinLinear with the same width and depth.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If `other` has different width, depth, or dtype

        """
        if (
            self.width != other.width
            or self.depth != other.depth
            or self.uint_maxval != other.uint_maxval
        ):
            raise TypeError("self and other have different width | depth | type")

        _merge_linear(
            self.cms,
            other.cms,
            self.width,
            self.depth,
            self.uint_maxval,
            self.n_added_records,
            other.n_added_records,
        )

    def save(self, filename: Union[str, Path]) -> None:
        """
        Save the sketch to `filename`

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
            args=np.array([self.width, self.depth], np.uint64),
            n_added_records=self.n_added_records,
            cms=self.cms,
            dtype=self.cms[0, 0],
        )

    @staticmethod
    def load(filename: Union[str, Path], shared_memory: bool = False):
        """
        Load a saved CountMinLinear stored in `filename`

        Parameters
        ----------
        filename : str | Path
            File path to the saved .npz file
        shared_memory : bool, optional
            If True, load into shared memory. Default is False.

        Returns
        -------
        CountMinLinear
        """
        with np.load(filename) as npzfile:
            args = npzfile["args"]
            cms_dtype = npzfile["dtype"].dtype
            if cms_dtype != np.uint32:
                raise TypeError("Saved sketch is not a CountMinLinear")

            cms = CountMinLinear(*args, shared_memory=shared_memory)
            np.copyto(cms.cms, npzfile["cms"])
            np.copyto(cms.n_added_records, npzfile["n_added_records"])

        return cms

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

        self.cms = np.frombuffer(
            existing_shm.buf[: self.cms.nbytes], self.cms.dtype
        ).reshape(int(self.depth), int(self.width))
        self.n_added_records = np.frombuffer(
            existing_shm.buf[self.cms.nbytes :], np.uint64
        )

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
        This can be used as the numerator of the idf piece of a tf-idf.

        Returns
        -------
        np.uint64
            The number of records that have been added to the sketch.

        """
        return self.n_added_records[1]

    def __getitem__(self, key: bytes) -> int:
        """
        Return estimated number of time `key` was added into the count-min sketch.
        Alias for query(key).
        """
        return self.query(key)

    def __del__(self):
        try:
            if self.shm:
                try:
                    # Need to explicitly del the arrays since they are sharing the
                    # memory block. Without this you get the MemoryError
                    # "cannot close exported pointers exist"
                    del self.cms
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
                    del self.cms
                    del self.n_added_records
                    gc.collect()
                    sleep(0.25)
                    self.existing_shm.close()
                except Exception as exc:
                    raise MemoryError(f"Failed to close existing_shm: {exc}")
        except AttributeError:
            pass


@njit(
    uint16(
        uint16[:, :],
        uint64[:],
        uint64,
        uint64,
        uint16,
        types.Bytes(types.uint8, 1, "C"),
    )
)
def _query_log16(cms, buckets, width, depth, uint_maxval, key):
    min_count = uint_maxval
    for row in range(depth):
        buckets[row] = fasthash64(key, row) % width
        count = cms[row, buckets[row]]
        if count < min_count:
            min_count = count
    return min_count


@njit(
    uint64(
        uint16[:, :],
        uint64[:],
        uint64[:],
        uint64,
        uint64,
        uint16,
        uint16,
        float64,
        float64[:],
        uint64,
        types.Bytes(types.uint8, 1, "C"),
    )
)
def _add_log16(
    cms,
    n_added_records,
    buckets,
    width,
    depth,
    uint_maxval,
    num_reserved,
    base,
    rand_nums,
    rand_ptr,
    key,
):
    # Track total number of elements added to the sketch
    n_added_records[0] += uint64(1)

    # This gets min_count AND updates buckets
    min_count = _query_log16(cms, buckets, width, depth, uint_maxval, key)
    increase, rand_ptr = _increase_counter(
        min_count, num_reserved, uint_maxval, base, rand_nums, rand_ptr
    )
    if increase:
        # Now update only those counter that have min_count
        for row in range(depth):
            count = cms[row, buckets[row]]
            if count == min_count:
                cms[row, buckets[row]] += np.uint16(1)

    return rand_ptr


@njit(
    uint64(
        uint16[:, :],
        uint64[:],
        uint64[:],
        uint64,
        uint64,
        uint16,
        uint16,
        float64,
        float64[:],
        uint64,
        types.Bytes(types.uint8, 1, "C"),
        uint64,
    )
)
def _add_ngram_log16(
    cms,
    n_added_records,
    buckets,
    width,
    depth,
    uint_maxval,
    num_reserved,
    base,
    rand_nums,
    rand_ptr,
    key,
    ngram,
):
    key_len = uint64(len(key))
    if key_len <= ngram:
        rand_ptr = _add_log16(
            cms,
            n_added_records,
            buckets,
            width,
            depth,
            uint_maxval,
            num_reserved,
            base,
            rand_nums,
            rand_ptr,
            key,
        )
    else:
        for i in range(key_len - (ngram - uint64(1))):
            rand_ptr = _add_log16(
                cms,
                n_added_records,
                buckets,
                width,
                depth,
                uint_maxval,
                num_reserved,
                base,
                rand_nums,
                rand_ptr,
                key[i : i + ngram],
            )
    return rand_ptr


@njit(
    types.void(
        uint16[:, :],
        uint16[:, :],
        uint64,
        uint64,
        uint64,
        uint16,
        uint16,
        float64,
        uint64[:],
        uint64[:],
    )
)
def _merge_log16(
    cms,
    other_cms,
    width,
    depth,
    max_count,
    uint_maxval,
    num_reserved,
    base,
    n_added_records,
    other_n_added_records,
):
    """
    Merge other_registers into registers

    Parameters
    ----------
    cms : np.ndarray, dtype=uint16, shape=(depth, width)
    other_cms : np.ndarray, dtype=uint16, shape=(depth, width)
    width : uint64
    depth : uint64
    max_count : uint64
    uint_maxval : uint16
    num_reserved : uint16
    base : float64
    n_added_records : np.ndarray, dtype=uint64, shape=(2,)
    other_n_added_records : np.ndarray, dtype=uint64, shape=(2,)

    Returns
    -------
    None
    """
    # Need to take care of various cases with log counters
    for row in range(depth):
        for col in range(width):
            # Get what the combined value should be
            v = _counter2value(cms[row, col], num_reserved, base) + _counter2value(
                other_cms[row, col], num_reserved, base
            )
            # If less than num_reserved, then c = v
            if v <= num_reserved:
                cms[row, col] = uint16(v)
            elif v >= max_count:
                cms[row, col] = uint_maxval
            else:
                cprime = np.log((v - num_reserved) * (base - 1.0) + 1.0) / np.log(base)
                cprime = uint16(cprime)
                clower = cprime + num_reserved
                vlower = _counter2value(clower, num_reserved, base)
                vhigher = _counter2value(clower + uint16(1), num_reserved, base)
                delta = v - vlower
                if delta / (vhigher - vlower) <= 0.5:
                    cms[row, col] = clower
                else:
                    cms[row, col] = clower + uint16(1)

    # Merge the special counters
    n_added_records[0] += other_n_added_records[0]
    n_added_records[1] += other_n_added_records[1]


class CountMinLog16(CountMinLinear):
    """
    Count-min sketch that uses 16-bit log counters with conservative updating.

    Parameters
    ----------
    width : int
        Width of the count-min sketch. Must be non-negative.
    depth : int, optional
        Depth of the count-min sketch. Must be non-negative. Default is 8.
    max_count : int, optional
        The maximum value we want to count up to for any given key. Default is
        2\*\*32 -1 (4,294,967,295).
    num_reserved : int, optional
        Perform linear counting for values [0, num_reserved]. After that use log
        counters. This gives more precise estimates for the number of times a key
        is seen for counts <= num_reserved. Default is 1023. This must be less than
        65,535 (2\*\*16 - 1).
    shared_memory : bool, optional
        If True, then CountMinLinear is placed in shared memory. Needed if
        performing multiprocessing as sketchnu.helpers.parallel_add() does.
        Default is False.

    Attributes
    ----------
    width : np.uint64
        Width of the 2-d array of counters of the count-min sketch
    depth : np.uint64
        Depth of the 2-d array of counters of the count-min sketch
    max_count : int, optional
        The maximum value we want to count up to for any given key. Default is
        2\*\*32 -1 (4,294,967,295).
    num_reserved : int, optional
        Perform linear counting for values [0, num_reserved]. After that use log
        counters. This gives more precise estimates for the number of times a key
        is seen for counts <= num_reserved. Default is 1023. This must be less than
        65,535 (2\*\*16 - 1).
    shared_memory : bool
        Whether `cms` and `n_added_records` are attached to a shared memory block
    cms : np.uint32[:,:]
        2-d array of the counters. Shape = (depth, width)
    n_added_records : np.uint64[:]
        1-d array that holds two special counters. The first is the number of elements
        that have been added to the sketch. Useful for calculating error limits. The
        second is used by helpers.parallel_add() to keep track of the number of records
        that have been processed. Useful if you want to calculate a TF-IDF.
    base : np.float64
        Calculated base for the log counters. Depends affected by max_count and
        num_reserved.
    """

    def __init__(
        self,
        width: int,
        depth: int = 8,
        max_count=4294967295,
        num_reserved=1023,
        shared_memory: bool = False,
    ) -> None:
        """
        Initialize a count-min sketch with 16-bit log counters with conservative
        updating.

        Parameters
        ----------
        width : int
            Width of the count-min sketch. Must be non-negative.
        depth : int, optional
            Depth of the count-min sketch. Must be non-negative. Default is 8.
        max_count : int, optional
            The maximum value we want to count up to for any given key. Default is
            2\*\*32 -1 (4,294,967,295).
        num_reserved : int, optional
            Perform linear counting for values [0, num_reserved]. After that use log
            counters. This gives more precise estimates for the number of times a key
            is seen for counts <= num_reserved. Default is 1023. This must be less than
            65,535 (2\*\*16 - 1).
        shared_memory : bool, optional
            If True, then CountMinLinear is placed in shared memory. Needed if
            performing multiprocessing as sketchnu.helpers.parallel_add() does.
            Default is False.

        Returns
        -------
        CountMinLog16
        """
        if width <= 0:
            raise ValueError(f"{width=:}. Must be greater than 0")
        if depth <= 0:
            raise ValueError(f"{depth=:}. Must be greater than 0")
        if num_reserved >= 65535:
            raise ValueError(f"{num_reserved=:,}. Must be less than 65,535")

        self.width = np.uint64(width)
        self.depth = np.uint64(depth)
        self.uint_maxval = np.uint16(2 ** 16 - 1)
        self.max_count = np.uint64(max_count)
        self.num_reserved = np.uint16(num_reserved)

        self.args = {
            "cms_type": "log16",
            "width": width,
            "depth": depth,
            "max_count": max_count,
            "num_reserved": num_reserved,
        }

        # Determine the base of the log counters
        self.base = _find_base(self.max_count, self.num_reserved, self.uint_maxval)

        # Stores the column index for a given key. Used frequently, so declaring here
        self.buckets = np.zeros(depth, np.uint64)

        # Create random numbers in batches of 2048
        rng = np.random.default_rng()
        self.rng = np.random.default_rng(rng.integers(0, 2 ** 63))
        self.rand_ptr = 0
        self.rand_nums = self.rng.random(2048)

        if shared_memory:
            cms_size = int(2 * width * depth)
            n_added_size = 8 * 2
            self.shm = SharedMemory(create=True, size=(cms_size + n_added_size))
            self.cms = np.frombuffer(self.shm.buf[:cms_size], np.uint16).reshape(
                depth, width
            )
            self.n_added_records = np.frombuffer(self.shm.buf[cms_size:], np.uint64)
        else:
            self.cms = np.zeros((depth, width), np.uint16)
            self.n_added_records = np.zeros(2, np.uint64)

    def query(self, key: bytes) -> float:
        """
        Return estimated number of times `key` was added into the count-min sketch

        Parameters
        ----------
        key : bytes
            Element whose estimated count you want returned

        Returns
        -------
        float

        """
        min_counter = _query_log16(
            self.cms, self.buckets, self.width, self.depth, self.uint_maxval, key
        )

        return _counter2value(min_counter, self.num_reserved, self.base)

    def add(self, key: bytes) -> None:
        """
        Add a single key to the count-min sketch and update the counter tracking total
        number of keys added to the count-min sketch. This is in n_added_records[0].

        Parameters
        ----------
        key : bytes
            Element to be added to the sketch
        
        Returns
        -------
        None
        """
        self.rand_ptr = _add_log16(
            self.cms,
            self.n_added_records,
            self.buckets,
            self.width,
            self.depth,
            self.uint_maxval,
            self.num_reserved,
            self.base,
            self.rand_nums,
            self.rand_ptr,
            key,
        )

    def add_ngram(self, key: bytes, ngram: int) -> None:
        """
        Take a given `key` and split it into ngrams of size `ngram` and then
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
        self.rand_ptr = _add_ngram_log16(
            self.cms,
            self.n_added_records,
            self.buckets,
            self.width,
            self.depth,
            self.uint_maxval,
            self.num_reserved,
            self.base,
            self.rand_nums,
            self.rand_ptr,
            key,
            ngram,
        )

    def merge(self, other) -> None:
        """
        Merge the count-min sketch `other` into this one.

        Parameters
        ----------
        other : CountMinLog16
            Another CountMinLog16 with the same parameters.
        
        Returns
        -------
        None

        Raises
        ------
        TypeError
            If `other` has different parameters

        """
        if (
            self.width != other.width
            or self.depth != other.depth
            or self.uint_maxval != other.uint_maxval
            or self.max_count != other.max_count
            or self.num_reserved != other.num_reserved
        ):
            raise TypeError(
                "self and other have different width|depth|type|max_count|num_reserved"
            )

        _merge_log16(
            self.cms,
            other.cms,
            self.width,
            self.depth,
            self.max_count,
            self.uint_maxval,
            self.num_reserved,
            self.base,
            self.n_added_records,
            other.n_added_records,
        )

    def save(self, filename: Union[str, Path]) -> None:
        """
        Save the count-min sketch to `filename`
        
        Parameters
        ----------
        filename: str | Path
            File to save the hll to disk. This will be a .npz file.
        
        Returns
        -------
        None

        """
        np.savez(
            filename,
            args=np.array([self.width, self.depth, self.max_count, self.num_reserved]),
            n_added_records=self.n_added_records,
            cms=self.cms,
            dtype=self.cms[0, 0],
        )

    @staticmethod
    def load(filename: Union[str, Path], shared_memory: bool = False):
        """
        Load a saved CountMinLog16 stored in `filename`

        Parameters
        ----------
        filename : str | Path
            File path to the saved .npz file
        shared_memory : bool, optional
            If True, load into shared memory. Default is False.
        
        Returns
        -------
        CountMinLog16
        """
        with np.load(filename) as npzfile:
            args = npzfile["args"]
            cms_dtype = npzfile["dtype"].dtype
            if cms_dtype != np.uint16:
                raise TypeError("Saved sketch is not a CountMinLog16")

            cms = CountMinLog16(*args, shared_memory=shared_memory)
            np.copyto(cms.cms, npzfile["cms"])
            np.copyto(cms.n_added_records, npzfile["n_added_records"])

        return cms


@njit(
    uint8(
        uint8[:, :], uint64[:], uint64, uint64, uint8, types.Bytes(types.uint8, 1, "C"),
    )
)
def _query_log8(cms, buckets, width, depth, uint_maxval, key):
    min_count = uint_maxval
    for row in range(depth):
        buckets[row] = fasthash64(key, row) % width
        count = cms[row, buckets[row]]
        if count < min_count:
            min_count = count
    return min_count


@njit(
    uint64(
        uint8[:, :],
        uint64[:],
        uint64[:],
        uint64,
        uint64,
        uint8,
        uint8,
        float64,
        float64[:],
        uint64,
        types.Bytes(types.uint8, 1, "C"),
    )
)
def _add_log8(
    cms,
    n_added_records,
    buckets,
    width,
    depth,
    uint_maxval,
    num_reserved,
    base,
    rand_nums,
    rand_ptr,
    key,
):
    # Track total number of elements added to the sketch
    n_added_records[0] += uint64(1)

    # This gets min_count AND updates buckets
    min_count = _query_log8(cms, buckets, width, depth, uint_maxval, key)
    increase, rand_ptr = _increase_counter(
        min_count, num_reserved, uint_maxval, base, rand_nums, rand_ptr
    )
    if increase:
        # Now update only those counter that have min_count
        for row in range(depth):
            count = cms[row, buckets[row]]
            if count == min_count:
                cms[row, buckets[row]] += np.uint8(1)

    return rand_ptr


@njit(
    uint64(
        uint8[:, :],
        uint64[:],
        uint64[:],
        uint64,
        uint64,
        uint8,
        uint8,
        float64,
        float64[:],
        uint64,
        types.Bytes(types.uint8, 1, "C"),
        uint64,
    )
)
def _add_ngram_log8(
    cms,
    n_added_records,
    buckets,
    width,
    depth,
    uint_maxval,
    num_reserved,
    base,
    rand_nums,
    rand_ptr,
    key,
    ngram,
):
    key_len = uint64(len(key))
    if key_len <= ngram:
        rand_ptr = _add_log8(
            cms,
            n_added_records,
            buckets,
            width,
            depth,
            uint_maxval,
            num_reserved,
            base,
            rand_nums,
            rand_ptr,
            key,
        )
    else:
        for i in range(key_len - (ngram - uint64(1))):
            rand_ptr = _add_log8(
                cms,
                n_added_records,
                buckets,
                width,
                depth,
                uint_maxval,
                num_reserved,
                base,
                rand_nums,
                rand_ptr,
                key[i : i + ngram],
            )
    return rand_ptr


@njit(
    types.void(
        uint8[:, :],
        uint8[:, :],
        uint64,
        uint64,
        uint64,
        uint8,
        uint8,
        float64,
        uint64[:],
        uint64[:],
    )
)
def _merge_log8(
    cms,
    other_cms,
    width,
    depth,
    max_count,
    uint_maxval,
    num_reserved,
    base,
    n_added_records,
    other_n_added_records,
):
    """
    Merge other_registers into registers

    Parameters
    ----------
    cms : np.ndarray, dtype=uint8, shape=(depth, width)
    other_cms : np.ndarray, dtype=uint8, shape=(depth, width)
    width : uint64
    depth : uint64
    max_count : uint64
    uint_maxval : uint8
    num_reserved : uint8
    base : float64
    n_added_records : np.ndarray, dtype=uint64, shape=(2,)
    other_n_added_records : np.ndarray, dtype=uint64, shape=(2,)
    
    Returns
    -------
    None
    """
    # Need to take care of various cases with log counters
    for row in range(depth):
        for col in range(width):
            # Get what the combined value should be
            v = _counter2value(cms[row, col], num_reserved, base) + _counter2value(
                other_cms[row, col], num_reserved, base
            )
            # If less than num_reserved, then c = v
            if v <= num_reserved:
                cms[row, col] = uint8(v)
            elif v >= max_count:
                cms[row, col] = uint_maxval
            else:
                cprime = np.log((v - num_reserved) * (base - 1.0) + 1.0) / np.log(base)
                cprime = uint8(cprime)
                clower = cprime + num_reserved
                vlower = _counter2value(clower, num_reserved, base)
                vhigher = _counter2value(clower + uint8(1), num_reserved, base)
                delta = v - vlower
                if delta / (vhigher - vlower) <= 0.5:
                    cms[row, col] = clower
                else:
                    cms[row, col] = clower + uint8(1)

    # Merge the special counters
    n_added_records[0] += other_n_added_records[0]
    n_added_records[1] += other_n_added_records[1]


class CountMinLog8(CountMinLog16):
    """
    Count-min sketch that uses 8-bit log counters with conservative updating.

    Parameters
    ----------
    width : int
        Width of the count-min sketch. Must be non-negative.
    depth : int, optional
        Depth of the count-min sketch. Must be non-negative. Default is 8.
    max_count : int, optional
        The maximum value we want to count up to for any given key. Default is
        2\*\*32 -1 (4,294,967,295).
    num_reserved : int, optional
        Perform linear counting for values [0, num_reserved]. After that use log
        counters. This gives more precise estimates for the number of times a key
        is seen for counts <= num_reserved. Default is 15. This must be less than
        255 (2\*\*8 - 1).
    shared_memory : bool, optional
        If True, then CountMinLinear is placed in shared memory. Needed if
        performing multiprocessing as sketchnu.helpers.parallel_add() does.
        Default is False.

    Attributes
    ----------
    width : np.uint64
        Width of the 2-d array of counters of the count-min sketch
    depth : np.uint64
        Depth of the 2-d array of counters of the count-min sketch
    max_count : int, optional
        The maximum value we want to count up to for any given key. Default is
        2\*\*32 -1 (4,294,967,295).
    num_reserved : int, optional
        Perform linear counting for values [0, num_reserved]. After that use log
        counters. This gives more precise estimates for the number of times a key
        is seen for counts <= num_reserved. Default is 15. This must be less than
        255 (2\*\*8 - 1).
    shared_memory : bool
        Whether `cms` and `n_added_records` are attached to a shared memory block
    cms : np.uint32[:,:]
        2-d array of the counters. Shape = (depth, width)
    n_added_records : np.uint64[:]
        1-d array that holds two special counters. The first is the number of elements
        that have been added to the sketch. Useful for calculating error limits. The
        second is used by helpers.parallel_add() to keep track of the number of records
        that have been processed. Useful if you want to calculate a TF-IDF.
    base : np.float64
        Calculated base for the log counters. Depends affected by max_count and
        num_reserved.
    """

    def __init__(
        self,
        width: int,
        depth: int = 8,
        max_count=4294967295,
        num_reserved=15,
        shared_memory: bool = False,
    ) -> None:
        """
        Initialize a count-min sketch with 8-bit log counters with conservative
        updating.

        Parameters
        ----------
        width : int
            Width of the count-min sketch. Must be non-negative.
        depth : int, optional
            Depth of the count-min sketch. Must be non-negative. Default is 8.
        max_count : int, optional
            The maximum value we want to count up to for any given key. Default is
            2\*\*32 -1 (4,294,967,295).
        num_reserved : int, optional
            Perform linear counting for values [0, num_reserved]. After that use log
            counters. This gives more precise estimates for the number of times a key
            is seen for counts <= num_reserved. Default is 15. This must be less than
            255 (2\*\*8 - 1).
        shared_memory : bool, optional
            If True, then CountMinLinear is placed in shared memory. Needed if
            performing multiprocessing as sketchnu.helpers.parallel_add() does.
            Default is False.
        
        Returns
        -------
        CountMinLog8
        """
        if width <= 0:
            raise ValueError(f"{width=:}. Must be greater than 0")
        if depth <= 0:
            raise ValueError(f"{depth=:}. Must be greater than 0")
        if num_reserved >= 255:
            raise ValueError(f"{num_reserved=:,}. Must be less than 255")

        self.width = np.uint64(width)
        self.depth = np.uint64(depth)
        self.uint_maxval = np.uint8(2 ** 8 - 1)
        self.max_count = np.uint64(max_count)
        self.num_reserved = np.uint8(num_reserved)

        self.args = {
            "cms_type": "log8",
            "width": width,
            "depth": depth,
            "max_count": max_count,
            "num_reserved": num_reserved,
        }

        # Determine the base of the log counters
        self.base = _find_base(self.max_count, self.num_reserved, self.uint_maxval)

        # Stores the column index for a given key. Used frequently, so declaring here
        self.buckets = np.zeros(depth, np.uint64)

        # Create random numbers in batches of 2048
        rng = np.random.default_rng()
        self.rng = np.random.default_rng(rng.integers(0, 2 ** 63))
        self.rand_ptr = 0
        self.rand_nums = self.rng.random(2048)

        if shared_memory:
            cms_size = int(1 * width * depth)
            n_added_size = 8 * 2
            self.shm = SharedMemory(create=True, size=(cms_size + n_added_size))
            self.cms = np.frombuffer(self.shm.buf[:cms_size], np.uint8).reshape(
                depth, width
            )
            self.n_added_records = np.frombuffer(self.shm.buf[cms_size:], np.uint64)
        else:
            self.cms = np.zeros((depth, width), np.uint8)
            self.n_added_records = np.zeros(2, np.uint64)

    def query(self, key: bytes) -> float:
        """
        Return estimated number of times `key` was added into the count-min sketch

        Parameters
        ----------
        key : bytes
            Element whose estimated count you want returned

        Returns
        -------
        float

        """
        min_counter = _query_log8(
            self.cms, self.buckets, self.width, self.depth, self.uint_maxval, key
        )

        return _counter2value(min_counter, self.num_reserved, self.base)

    def add(self, key: bytes) -> None:
        """
        Add a single key to the count-min sketch and update the counter tracking total
        number of keys added to the count-min sketch. This is in n_added_records[0].

        Parameters
        ----------
        key : bytes
            Element to be added to the sketch
        
        Returns
        -------
        None
        """
        self.rand_ptr = _add_log8(
            self.cms,
            self.n_added_records,
            self.buckets,
            self.width,
            self.depth,
            self.uint_maxval,
            self.num_reserved,
            self.base,
            self.rand_nums,
            self.rand_ptr,
            key,
        )

    def add_ngram(self, key: bytes, ngram: int) -> None:
        """
        Take a given `key` and split it into ngrams of size `ngram` and then
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
        self.rand_ptr = _add_ngram_log8(
            self.cms,
            self.n_added_records,
            self.buckets,
            self.width,
            self.depth,
            self.uint_maxval,
            self.num_reserved,
            self.base,
            self.rand_nums,
            self.rand_ptr,
            key,
            ngram,
        )

    def merge(self, other) -> None:
        """
        Merge the count-min sketch `other` into this one.

        Parameters
        ----------
        other : CountMinLog8
            Another CountMinLog8 with the same parameters.
        
        Returns
        -------
        None

        Raises
        ------
        TypeError
            If `other` has different parameters

        """
        if (
            self.width != other.width
            or self.depth != other.depth
            or self.uint_maxval != other.uint_maxval
            or self.max_count != other.max_count
            or self.num_reserved != other.num_reserved
        ):
            raise TypeError(
                "self and other have different width|depth|type|max_count|num_reserved"
            )

        _merge_log8(
            self.cms,
            other.cms,
            self.width,
            self.depth,
            self.max_count,
            self.uint_maxval,
            self.num_reserved,
            self.base,
            self.n_added_records,
            other.n_added_records,
        )

    @staticmethod
    def load(filename: Union[str, Path], shared_memory: bool = False):
        """
        Load a saved CountMinLog8 stored in `filename`

        Parameters
        ----------
        filename : str | Path
            File path to the saved .npz file
        shared_memory : bool, optional
            If True, load into shared memory. Default is False.
        
        Returns
        -------
        CountMinLog8
        """
        with np.load(filename) as npzfile:
            args = npzfile["args"]
            cms_dtype = npzfile["dtype"].dtype
            if cms_dtype != np.uint8:
                raise TypeError("Saved sketch is not a CountMinLog8")

            cms = CountMinLog8(*args, shared_memory=shared_memory)
            np.copyto(cms.cms, npzfile["cms"])
            np.copyto(cms.n_added_records, npzfile["n_added_records"])

        return cms


def CountMin(
    cms_type: str,
    width: int,
    depth: int = 8,
    max_count: int = 4294967295,
    num_reserved: int = None,
    shared_memory: bool = False,
) -> Union[CountMinLinear, CountMinLog16, CountMinLog8]:
    """
    Convenience function to instantiate a count-min sketch of the given type.

    Parameters
    ----------
    cms_type : str
        Must be 'linear' | 'log16' | 'log8'
    width : int
        Width of the count-min sketch. Best if you keep width >= n_unique/2
    depth : int, optional
        Depth of the count-min sketch. Sets the number of different hash
        functions used. Probability of exceeding error limits is determined by
        the depth, exp(-depth). Default is 8 which should be fine for most
        circumstances.
    max_count : int, optional
        Maximum value the count a given element may reach. Not used by 'linear'
        type. Default is 2\*\*32 - 1 (4,294,967,295) to match linear type.
    num_reserved : int, optional
        Perform linear counting for values [0, num_reserved]. After that use
        log counters. This gives more precise estimates for the number of times
        an element is seen for counts <= num_reserved. Not used by the 'linear'
        type. Default is None which uses 15 for log8 and 1023 for log16. This
        must be less than the maximum uint value: 65,535 for log16 and 255
        for log8.
    shared_memory : bool, optional
        If True, then count-min sketch is placed in shared memory. Needed if
        performing multiprocessing as sketchnu.helpers.parallel_add() does.
        Default is False.
    
    Returns
    -------
    cms : CountMinLinear | CountMinLog16 | CountMinLog8
        A count-min sketch of the requested type and size
    """
    if cms_type == "linear":
        cms = CountMinLinear(width, depth, shared_memory)
    elif cms_type == "log16":
        if num_reserved is None:
            cms = CountMinLog16(width, depth, max_count, shared_memory=shared_memory)
        else:
            cms = CountMinLog16(width, depth, max_count, num_reserved, shared_memory)
    elif cms_type == "log8":
        if num_reserved is None:
            cms = CountMinLog8(width, depth, max_count, shared_memory=shared_memory)
        else:
            cms = CountMinLog8(width, depth, max_count, num_reserved, shared_memory)
    else:
        raise ValueError(f"{cms_type=:}. Must be linear | log16 | log8")

    return cms


def load(
    filename: Union[str, Path], shared_memory: bool = False
) -> Union[CountMinLinear, CountMinLog16, CountMinLog8]:
    """
    Load a saved count-min sketch stored in `filename`

    Parameters
    ----------
    filename: str | Path
        File path to the saved .npz file
    shared_memory : bool
        If True, load into shared memory
        
    Returns
    -------
    CountMinLinear | CountMinLog16 | CountMinLog8
    """
    with np.load(filename) as npzfile:
        cms_dtype = npzfile["dtype"].dtype

    if cms_dtype == np.uint32:
        return CountMinLinear.load(filename, shared_memory)
    elif cms_dtype == np.uint16:
        return CountMinLog16.load(filename, shared_memory)
    elif cms_dtype == np.uint8:
        return CountMinLog8.load(filename, shared_memory)
