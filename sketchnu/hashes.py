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


Numba implementations of the **non-cryptographic** hashing functions
FastHash (32 & 64-bit) and MurmurHash3 (32-bit). The 64-bit FastHash is used
in both the HyperLogLog and count-min sketch code.

"""
from numba import njit, uint8, uint32, uint64, types
import numpy as np

########## FastHash ##########
@njit(uint64(uint64, uint64, uint64))
def _xor_shiftl(v, t, l):
    return v ^ (t << l)


@njit(uint64(uint64))
def _fhmix64(h):
    h ^= h >> 23
    h *= uint64(0x2127599BF4325C37)
    h ^= h >> 47

    return h


@njit(uint64(types.Bytes(types.uint8, 1, "C"), uint64))
def fasthash64(key, seed):
    """
    Calculate the unsigned 64-bit integer FastHash value of `key` with the
    given `seed`. Code adapted from https://github.com/rurban/smhasher/

    Parameters
    ----------
    key : bytes
        Bytes to be hashed
    seed : uint64
        Seed to use when hashing
    
    Returns
    -------
    uint64
        Unsigned 64-bit integer hash value of `key`

    """
    m = uint64(0x880355F21E6D1965)

    key_len = uint64(len(key))
    nblocks = key_len // 8  # How many 8-byte blocks are there

    h = seed ^ (key_len * m)

    if nblocks > 0:
        # Cast complete 64-bit chunks into blocks array
        blocks = np.frombuffer(key[: nblocks * 8], np.uint64)
        # Process the 64-bit blocks
        for v in blocks:
            h ^= _fhmix64(v)
            h *= m

    # The tail should have no more than 7 bytes remaining.
    # How many bytes remain after 8-byte block processing
    switch_case = key_len & 7
    if switch_case == 7:
        tail = key[nblocks * 8 :]
        v = uint64(0)
        v = _xor_shiftl(v, tail[6], 48)
        v = _xor_shiftl(v, tail[5], 40)
        v = _xor_shiftl(v, tail[4], 32)
        v = _xor_shiftl(v, tail[3], 24)
        v = _xor_shiftl(v, tail[2], 16)
        v = _xor_shiftl(v, tail[1], 8)
        v ^= uint64(tail[0])
        h ^= _fhmix64(v)
        h *= m
    elif switch_case == 6:
        tail = key[nblocks * 8 :]
        v = uint64(0)
        v = _xor_shiftl(v, tail[5], 40)
        v = _xor_shiftl(v, tail[4], 32)
        v = _xor_shiftl(v, tail[3], 24)
        v = _xor_shiftl(v, tail[2], 16)
        v = _xor_shiftl(v, tail[1], 8)
        v ^= uint64(tail[0])
        h ^= _fhmix64(v)
        h *= m
    elif switch_case == 5:
        tail = key[nblocks * 8 :]
        v = uint64(0)
        v = _xor_shiftl(v, tail[4], 32)
        v = _xor_shiftl(v, tail[3], 24)
        v = _xor_shiftl(v, tail[2], 16)
        v = _xor_shiftl(v, tail[1], 8)
        v ^= uint64(tail[0])
        h ^= _fhmix64(v)
        h *= m
    elif switch_case == 4:
        tail = key[nblocks * 8 :]
        v = uint64(0)
        v = _xor_shiftl(v, tail[3], 24)
        v = _xor_shiftl(v, tail[2], 16)
        v = _xor_shiftl(v, tail[1], 8)
        v ^= uint64(tail[0])
        h ^= _fhmix64(v)
        h *= m
    elif switch_case == 3:
        tail = key[nblocks * 8 :]
        v = uint64(0)
        v = _xor_shiftl(v, tail[2], 16)
        v = _xor_shiftl(v, tail[1], 8)
        v ^= uint64(tail[0])
        h ^= _fhmix64(v)
        h *= m
    elif switch_case == 2:
        tail = key[nblocks * 8 :]
        v = uint64(0)
        v = _xor_shiftl(v, tail[1], 8)
        v ^= uint64(tail[0])
        h ^= _fhmix64(v)
        h *= m
    elif switch_case == 1:
        tail = key[nblocks * 8 :]
        v = uint64(0)
        v ^= uint64(tail[0])
        h ^= _fhmix64(v)
        h *= m

    return _fhmix64(h)


@njit(uint32(types.Bytes(uint8, 1, "C"), uint64))
def fasthash32(key, seed):
    """
    Calculate the unsigned 32-bit integer FastHash value of `key` with the
    given `seed`. Code adapted from https://github.com/rurban/smhasher/

    Parameters
    ----------
    key : bytes
        Bytes to be hashed
    seed : uint64
        Seed to use when hashing
    
    Returns
    -------
    uint32
        Unsigned 32-bit integer hash value of `key`

    """
    h = fasthash64(key, seed)
    return uint32(h - (h >> 32))


########## End FastHash ##########


########## Murmur3 ##########
@njit(uint32(uint32, uint32))
def _xor32(x, y):
    return x ^ y


@njit(uint32(uint32, uint32))
def _shift32r(x, y):
    return x >> y


@njit(uint32(uint32, uint32))
def _shift32l(x, y):
    return x << y


@njit(uint32(uint32, uint32))
def _rotl32(x, r):
    return _shift32l(x, r) | _shift32r(x, 32 - r)


@njit(uint32(uint32))
def _fmix32(h):
    """
    Force all bits of a hash block to avalance.
    """
    h = _xor32(h, _shift32r(h, 16))
    h *= uint32(0x85EBCA6B)
    h = _xor32(h, _shift32r(h, 13))
    h *= uint32(0xC2B2AE35)
    h = _xor32(h, _shift32r(h, 16))

    return h


@njit(uint32(types.Bytes(uint8, 1, "C"), uint32))
def murmur3(key, seed):
    """
    Calculate the unsigned 32-bit integer MurmurHash3 value of `key` with the
    given `seed`. Code adapted from https://github.com/rurban/smhasher/

    Parameters
    ----------
    key : bytes
        Bytes to be hashed
    seed : uint32
        Seed to use when hashing

    Returns
    -------
    uint32
        Unsigned 32-bit integer hash value of `key`

    """
    key_len = uint32(len(key))
    nblocks = key_len // 4  # How many 4-byte blocks are there

    # Cast whole 4-byte blocks to uint32
    blocks = np.frombuffer(key[: nblocks * 4], np.uint32)
    # Leave any remaining as uint8
    tail = key[nblocks * 4 :]

    h = seed
    c1 = uint32(0xCC9E2D51)
    c2 = uint32(0x1B873593)
    c3 = uint32(0xE6546B64)

    for i in range(nblocks):
        k1 = blocks[i]
        k1 *= c1
        k1 = _rotl32(k1, 15)
        k1 *= c2

        h = _xor32(h, k1)
        h = _rotl32(h, 13)
        h = h * uint32(5) + c3

    k1 = uint32(0)
    switch_len = key_len & 3
    if switch_len == 3:
        k1 = _xor32(k1, _shift32l(tail[2], 16))
        k1 = _xor32(k1, _shift32l(tail[1], 8))
        k1 = _xor32(k1, tail[0])
        k1 *= c1
        k1 = _rotl32(k1, 15)
        k1 *= c2
        h = _xor32(h, k1)
    elif switch_len == 2:
        k1 = _xor32(k1, _shift32l(tail[1], 8))
        k1 = _xor32(k1, tail[0])
        k1 *= c1
        k1 = _rotl32(k1, 15)
        k1 *= c2
        h = _xor32(h, k1)
    elif switch_len == 1:
        k1 = _xor32(k1, tail[0])
        k1 *= c1
        k1 = _rotl32(k1, 15)
        k1 *= c2
        h = _xor32(h, k1)

    h = _xor32(h, key_len)
    h = _fmix32(h)

    return h


########## End Murmur3 ##########
