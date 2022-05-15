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
from sketchnu.hashes import fasthash32, murmur3


def test_fasthash32():
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
