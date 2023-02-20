HyperLogLog++
=============

The Basics
----------

A sketch that estimates the number of unique elements, cardinality, that have
been added into the sketch.

A HyperLogLog is initialized with a given precision, `p`, which specifies the
number of registers, `m=2\*\*p`, used. The larger the `p`, the more accurate the
estimated cardinality. This implementation allows for precision between [7,16].

When adding an element (bytes), the element is hashed to a 64-bit unsigned
integer. The first `p` bits are used to determine which register to use and the
remaining 64-`p` bits are used to update the register if needed. Each register
simply keeps track of the maximum number of leading zero bits seen so far. This
value is what allows for the estimation of the cardinality.

The HyperLogLog is implemented as a Numba class. The convenience function
:code:`HyperLogLog()` is the recommended way to instantiate a HyperLogLog object. The
Numba class itself is called :code:`HyperLogLog_nu`.

Usage
-----

::

    from sketchnu.hyperloglog import HyperLogLog

    hll = HyperLogLog(p=16, seed=0)  # Default settings

    # Add one key
    hll.add(b"abcd")

    # Add multiple keys as an iterable
    hll.update([b"1234", b"4321"])

    # Add multiple keys as a dict
    # Dict values are irrelevant for this sketch and ignored
    hll.update({b"4321": 4, b"ab23": 10})

    # Query for the current estimated cardinality
    est_cardinality = hll.query()
    print(f"{est_cardinality=:.2f}. True cardinality = 4")

    hll2 = HyperLogLog()
    hll2.update([b"12cd", b"1234"])

    # Merge two HyperLogLog's together; must have same p & seed
    hll.merge(hll2)
    est_cardinality = hll.query()
    print(f"After merging, {est_cardinality=:.2f}. True cardinality = 5")

    # Save to disk; load from disk
    hll.save("/path/to/save/hll.npz")
    hll_load = HyperLogLog.load("/path/to/save/hll.npz")

The Details
-----------

HyperLogLog is a sketch algorithm that estimates the number of distinct elements,
cardinality, of large datasets. The HyperLogLog algorithm was described in
`"HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm"
<http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf>`_ by Flajolet, Fusy,
Gandouet, and Meunier in 2007, which itself was an improvement over the LogLog
algorithm. The authors prove that the HyperLogLog’s estimated cardinality
“is asymptotically almost unbiased”.

An improvement to HyperLogLog, HyperLogLog++, was proposed in the 2018 paper
`"HyperLogLog in Practice: Algorithmic Engineering of a State of the Art
Cardinality Estimation Algorithm" <https://stefanheule.com/papers/edbt13-hyperloglog.pdf>`_
by Heule, Nunkesser, and Hall. This paper is the basis for the implementation
found here.

The HyperLogLog++ algorithm addressed two shortcomings found in HyperLogLog. The
original algorithm used a 32-bit hash function which then resulted in hash
collisions when the cardinality of the dataset approached 2\*\*32 ~ 4 billion. This was
easily addressed by switching to a 64-bit hash function. We use FastHash64 which
has also been implemented in Numba as a part of this package.

The second issue was a bias in the estimated cardinality when the cardinality is
low. The HyperLogLog++ algorithm fixes this by experimentally estimating the
bias and then using this bias correction when the cardinality is within an
experimentally determined range. We use the predetermined threshold given in the
Heule et al. paper, but we have run our own experiments to determine the bias.
You can find this code in the hll_bias_experiment.py. The estimate and bias
values are stored in the hll_constants.py file. This file contains the new
values we have determined as well as the original values from the paper. The
authors' provided value were taken from http://goo.gl/iU8Ig.

The authors' had also implemented a sparse representation which can reduce the
required memory when the cardinality is low. That has **not** been implemented
in this package.

We thank the people behind the Python package,
`datasketch <http://ekzhu.com/datasketch/index.html>`_, which was an inspiration
to our efforts and helpful in seeing a Python implementation of the algorithm.

A simple timeit comparison resulted in:

* Adding 1M keys: We are 3.8x faster than datasketch's HyperLogLogPlusPlus
* Estimating: We are 4.8x faster than datasketch's HyperLogLogPlusPlus

Testing
-------

Given that these are probablistic in nature, writing traditional software tests
is a bit challenging. We have written statistical tests that should pass the
vast majority of the time. The tests can be found in tests/test_hyperloglog.py

We test the low, medium, and high range of HyperLogLog using a t-test to test the
null hypothesis that the mean of the difference between the true cardinality and the
estimated (bias corrected if within the appropriate range) is 0. The HyperLogLog's
estimated cardinality "is asymptotically almost unbiased". Despite this, even
low values typically pass the test. We use a confidence level of 99.9% to reject
the null hypothesis. The tests assert that we should fail to reject the null
hypothesis.

We also use a t-test to test the null hypothesis that the mean of the difference
between the true count and the estimated count after merging two HyperLogLogs is 0.
A confidence level of 99.9% is used to reject the null hypothesis. The test asserts
that we should fail to reject the null hypothesis.