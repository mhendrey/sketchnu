Two different hashing algorithms are implemented, FastHash and MurmurHash3, but
only the FastHash is used by the sketch algorithms.

Hash Functions
==============

FastHash
--------

An implementation of both the 32-bit and 64-bit versions of the FastHash
algorithm implemented and checked against the C++ version found at
https://github.com/rurban/smhasher/blob/master/fasthash.cpp.
The FastHash64 is used extensively throughout sketchnu. Both functions take
bytes and a seed value and return an unsigned integer of 32 or 64-bits. The
32-bit version first calls the 64-bit version and then does some bit mixing
to reduce down to 32-bit.

Usage
~~~~~
The FastHash functions take in bytes and return unsigned integers::

    from sketchnu.hashes import fasthash64, fasthash32
    hv_64 = fasthash64(b'one_key', 0)
    hv_32 = fasthash32(b'one_key', 1)

Testing
~~~~~~~

Compare the fasthash32 against the smhasher C++ version. Since the
fasthash32() calls fasthash64() and then does some bit mixing, by testing
the fasthash32() we are also testing the fasthash64().

Values we assert against are from running the C++ version with the given
keys and seeds. C++ code was taken from
https://github.com/rurban/smhasher/blob/master/fasthash.cpp
https://github.com/rurban/smhasher/blob/master/fasthash.h

MurmurHash3
-----------

An implementation of the 32-bit version of the MurmurHash3 algorithm
implemented and checked against the C++ version found at
https://github.com/rurban/smhasher/blob/master/MurmurHash3.cpp

Usage
~~~~~
The murmur3 function takes in bytes and returns unsigned 32-bit integer::

    from sketchnu.hashes import murmur3
    hv = murmur3(b'one_key', 0)

Testing
~~~~~~~
Compare the murmur3 against the smhasher C++ version found in the
MurmurHash3_x86_32() function.

Values we assert against are from running the C++ version with the given
keys and seeds. C++ code was taken from
https://github.com/rurban/smhasher/blob/master/MurmurHash3.cpp
https://github.com/rurban/smhasher/blob/master/MurmurHash3.h
