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
to a series of Numba functions and a Python class.

Example
-------
To get a HyperLogLog with precision of 16 and seed 0::

    from sketchnu.hyperloglog import HyperLogLog

    hll = HyperLogLog()
    key = 'testing'.encode('utf-8')
    hll.add(key)
    # To get a estimated cardinality
    hll.query()

"""
from multiprocessing.shared_memory import SharedMemory
from numba import njit, uint8, uint64, float64, types
import numpy as np
from pathlib import Path
from typing import List, Union

from sketchnu.hashes import fasthash64
from sketchnu.hll_constants import sub_algorithm_threshold, raw_estimate, bias_data


@njit(float64(uint64, uint64))
def _linear_counting(m, n_zero):
    """
    If there are registers that are zero, then we do linear_counting to
    estimate the cardinality
    
    Parameters
    ----------
    m : int
        Number of registers
    n_zero : int
        Number of registers that equal 0

    Returns
    -------
    float64
        Estimated cardinality when some registers are still zero

    """
    return float64(m) * np.log(float64(m) / float64(n_zero))


@njit(float64(uint8[:], uint64, float64))
def _estimation_function(registers, m, alpha):
    """
    Estimated cardinality with no bias correction

    Returns
    -------
    float64

    """
    total = float64(0.0)
    for r in registers:
        total += 2.0 ** (-float64(r))
    return alpha * float64(m ** 2) / total


@njit(uint8(uint64))
def _n_leading_zeros64(x):
    """
    Calculate the number of leading zeros for a given uint64 integer.
    If numba ever gets a uint6 type, then we could replace uint8.

    Parameters
    ----------
    x : uint64

    Returns
    -------
    uint8
        Number of leading zeros in the bit representation of `x`

    """
    zero = uint64(0)
    n = uint8(64)
    y = x >> uint64(32)
    if y != zero:
        n = n - uint8(32)
        x = y
    y = x >> uint64(16)
    if y != zero:
        n = n - uint8(16)
        x = y
    y = x >> uint64(8)
    if y != zero:
        n = n - uint8(8)
        x = y
    y = x >> uint64(4)
    if y != zero:
        n = n - uint8(4)
        x = y
    y = x >> uint64(2)
    if y != zero:
        n = n - uint8(2)
        x = y
    y = x >> uint64(1)
    if y != zero:
        return n - uint8(2)

    return n - uint8(x)


@njit(float64(uint8[:], uint64, uint64, float64, float64[:], float64[:]))
def _query(registers, m, threshold, alpha, raw_estimate, bias_data):
    """
    Return the estimated cardinality

    Returns
    -------
    float
        Estimated cardinality of elements added to the HyperLogLog so far

    """
    # Get the number of registers that are still zero
    n_zero = m - uint64(np.count_nonzero(registers))

    if n_zero > 0:
        # Start by trying linear counting
        cardinality = _linear_counting(m, n_zero)

        # If above sub-algorithm threshold, switch to estimation function
        # with bias correction
        if cardinality > threshold:
            est = _estimation_function(registers, m, alpha)
            bias = np.interp(est, raw_estimate, bias_data)
            cardinality = est - bias
    else:
        cardinality = _estimation_function(registers, m, alpha)
        # If cardinality is relatively low, then remove estimated bias
        if cardinality <= float64(5 * m):
            bias = np.interp(cardinality, raw_estimate, bias_data)
            cardinality = cardinality - bias

    return cardinality


@njit(types.void(uint8[:], uint64, uint64, uint64, types.Bytes(types.uint8, 1, "C")))
def _add(registers, seed, p, m, key):
    """
    Add a single `key` to the HyperLogLog.
    
    Parameters
    ----------
    registers : uint8[:]
        HyperLogLog's registers
    seed : np.uint64
        Seed for hashing
    p : np.uint64
        HyperLogLog's precision
    m : np.uint64
        Size of HyperLogLog's registers
    key : bytes
        Element to add to the HyperLogLog

    Returns
    -------
    None
        registers has a side-effect of being updated
    """
    hash_val = fasthash64(key, seed)
    # Get the index of the register using the first p bits of the hash
    reg_idx = hash_val & uint64(m - 1)
    # Get the remaining bits for getting number of leading zeros
    bits = hash_val >> p

    # Get number of leading zeros of the 64-bit bits
    # subtract p since the bits is padded with zeros since it is
    # just a 64-p bit value.
    rank = _n_leading_zeros64(bits) - p + 1
    registers[reg_idx] = max(registers[reg_idx], rank)

    return None


@njit(
    types.void(
        uint8[:], uint64, uint64, uint64, types.Bytes(types.uint8, 1, "C"), uint64
    )
)
def _add_ngram(registers, seed, p, m, key, ngram):
    """
    Take a given `key` and split it into ngrams of size `ngram` and then
    add the ngrams to the sketch. If the `key` length is less than `ngram`
    then add the whole `key`

    Parameters
    ----------
    registers : uint8[:]
    seed : uint64
    p : uint64
    m : uint64
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
        _add(registers, seed, p, m, key)
    else:
        for i in range(key_len - (ngram - uint64(1))):
            _add(registers, seed, p, m, key[i : i + ngram])


@njit(types.void(uint8[:], uint8[:], uint64))
def _merge(registers, other_registers, m):
    """
    Merge other_registers into registers

    Parameters
    ----------
    registers : np.ndarray, dtype=uint8
        Registers for a HyperLogLog
    other_registers : np.ndarray, dtype=uint8
        Registers for another HyperLogLog that is merged into the first
    m : uint64
        Number of elements in both registers and other_registers
    
    Returns
    -------
    None
        Side-effect that updates `registers`
    """
    for i in range(m):
        registers[i] = max(registers[i], other_registers[i])


class HyperLogLog:
    """
    Implementation of the HyperLogLog++ algorithm which uses a 64-bit hash function and
    corrects bias when cardinality is low.

    Parameters
    ----------
    p : int
        Precision specifies the number of registers (2\*\*p). The larger the p,
        the more accurate the estimated cardinality. Must be between [7, 16]
    seed : int, optional
        Seed passed to the fasthash64 function. Default is 0
    shared_memory : bool, optional
        If True, then HyperLogLog is placed in shared memory. Needed if
        performing multiprocessing as sketchnu.helpers.parallel_add() does.
        Default is False.
    
    Attributes
    ----------
    p : int
        Precision of the HyperLogLog. Must be between [7, 16]
    seed : int
        Seed passed to hash function
    m : int
        Number of registers. Equal to 2\*\*p
    shared_memory : bool, optional
        If True, then HyperLogLog is placed in shared memory. Needed if
        performing multiprocessing as sketchnu.helpers.parallel_add() does.
        Default is False.

    """
    def __init__(self, p: int = 16, seed: int = 0, shared_memory: bool = False):
        """
        Instantiate a HyperLogLog. This is an implementation of the HyperLogLog++
        algorithm which uses a 64-bit hash function and corrects bias when cardinality
        is low.

        Parameters
        ----------
        p : int
            Precision specifies the number of registers (2**p). The larger the p,
            the more accurate the estimated cardinality. Must be between [7, 16]
        seed : int, optional
            Seed passed to the fasthash64 function. Default is 0
        shared_memory : bool, optional
            If True, then HyperLogLog is placed in shared memory. Needed if
            performing multiprocessing as sketchnu.helpers.parallel_add() does.
            Default is False.

        Returns
        -------
        HyperLogLog
            The Python class for the HyperLogLog with specified p and seed.

        """
        self.p = np.uint64(p)
        self.seed = np.uint64(seed)

        if self.p > np.uint64(16) or self.p < np.uint64(7):
            raise ValueError("p must be within [7, 16]")

        self.args = {"p": p, "seed": seed}

        # Total number of registers, 2**p
        self.m = np.uint64(1) << self.p
        # Registers will hold the maximum number of (leading zeros + 1) seen
        # for each of the m registers
        if shared_memory:
            self.shm = SharedMemory(create=True, size=int(self.m))
            self.registers = np.frombuffer(self.shm.buf, np.uint8)
        else:
            self.registers = np.zeros(self.m, np.uint8)

        self.alpha = np.float64(0.7213 / (1.0 + 1.079 / self.m))

        # Using [p-7] since we start with p = 7.
        # Gave IndexError: only integers are valid indices
        # Didn't like having np.uint64 and int mixed together
        self.threshold = sub_algorithm_threshold[int(self.p) - 7]
        self.bias_data = bias_data[int(self.p) - 7, :]
        self.raw_estimate = raw_estimate[int(self.p) - 7, :]

    def add(self, key: bytes) -> None:
        """
        Add a single key to the HyperLogLog.
        
        Note
        ----
        Currently, the numba.typed.List needed for the update() are quite slow.
        Speedtest show that you are better off looping through the keys in
        python and add(key) instead of update(numba.typed.List(keys)). Taking
        just over twice as long.

        Parameters
        ----------
        key : bytes
            Element to add to the HyperLogLog

        Returns
        -------
        None
        """
        _add(self.registers, self.seed, self.p, self.m, key)

    def update(self, keys: List[bytes]) -> None:
        """
        Given a list of keys, update the HyperLogLog. This follows the
        convention of collections.Counter in that keys should be a list of
        keys.

        Note
        ----
        Currently, the numba.typed.List needed for the update() are quite slow.
        Speedtest show that you are better off looping through the keys in
        python and add(key) instead of update(numba.typed.List(keys)). Taking
        just over twice as long. If repeated adding the same keys, then update
        is faster.

        Parameters
        ----------
        keys : List[bytes]
            List of elements to add to the HyperLogLog at once

        Returns
        -------
        None

        """
        for key in keys:
            self.add(key)

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
        ngram = np.uint64(ngram)
        _add_ngram(self.registers, self.seed, self.p, self.m, key, ngram)

    def update_ngram(self, keys: bytes, ngram: int) -> None:
        """
        Given a list of keys, split each into ngrams of size `ngram`, and then
        add them to the sketch.

        Note
        ----
        Speed tests show that you are slightly better off looping through the
        keys in Python and calling add_ngram() instead of update_ngram(). Used
        50k keys, each of 200 bytes, and ngram=4 to find that update_ngram()
        was ~ 6% slower. This is significantly better than add() vs. update().
        If repeated adding the same keys (for testing), then update_ngram() is
        just a little bit faster than add_ngram().

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

    def query(self) -> float:
        """
        Return the estimated cardinality

        Returns
        -------
        float
            Estimated cardinality of elements added to the HyperLogLog so far
        """
        return _query(
            self.registers,
            self.m,
            self.threshold,
            self.alpha,
            self.raw_estimate,
            self.bias_data,
        )

    def merge(self, other) -> None:
        """
        Merge the HyperLogLog `other` into this one.

        Parameters
        ----------
        other : HyperLogLog
            Another HyperLogLog with the same precision and seed
        
        Returns
        -------
        None

        Raises
        ------
        TypeError
            If `other` has different precision or seed from this one

        """
        if self.p != other.p or self.seed != other.seed:
            raise TypeError(f"Precision & seed in other must match self")

        _merge(self.registers, other.registers, self.m)

    def attach_existing_shm(self, existing_shm_name: str) -> SharedMemory:
        """
        Attach this sketch to an existing shared memory block. Useful when working
        within a spawned child process. This creates self.existing_shm which gets
        closed when self.__del__ gets called.

        Parameters
        ----------
        existing_shm_name : str
            Name an existing shared memory block to attach this sketch to
        
        Returns
        -------
        None
        """
        existing_shm = SharedMemory(name=existing_shm_name)
        self.registers = np.frombuffer(existing_shm.buf, np.uint8)

        self.existing_shm = existing_shm

    def save(self, filename: Union[str, Path]) -> None:
        """
        Save the HyperLogLog sketch, hll, to the file, filename
        
        Parameters
        ----------
        filename: str | Path
            File to save the hll to disk. This will be a .npz file.
        
        Returns
        -------
        None

        """
        np.savez(
            filename, args=np.array([self.p, self.seed], np.uint64), hll=self.registers
        )

    @staticmethod
    def load(filename: Union[str, Path], shared_memory: bool = False):
        """
        Load a saved HyperLogLog sketch stored in filename

        Parameters
        ----------
        filename: str | Path
            File path to the saved .npz file
        shared_memory : bool
            If True, copy registers into shared memory
            
        Returns
        -------
        HyperLogLog
        """
        with np.load(filename) as npzfile:
            args = npzfile["args"]
            hll = HyperLogLog(*args, shared_memory=shared_memory)
            np.copyto(hll.registers, npzfile["hll"])

        return hll

    def __del__(self):
        try:
            if self.shm:
                try:
                    self.shm.close()
                    self.shm.unlink()
                except Exception as exc:
                    raise MemoryError(f"Failed to close & unlink: {exc}")
        except AttributeError:
            pass

        try:
            if self.existing_shm:
                try:
                    self.existing_shm.close()
                except Exception as exc:
                    raise MemoryError(f"Failed to close existing_shm: {exc}")
        except AttributeError:
            pass
