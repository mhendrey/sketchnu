Count-Min Sketch
================

The Basics
----------

A sketch that estimates the number of times a given element has been added into
the sketch. The key benefits for using a count-min sketch are:

* Fixed memory size no matter the number of distinct elements added
* Saves memory by not storing the keys
* Error guarantees (See `The Details`_ section)
* Easily parallelized across multiple workers, each with their own count-min sketch

A count-min sketch (cms) is a 2-d array of counters with a width `w` (number of
columns) and depth `d` (number of rows). In this implementation, we provide
three different cms types which determine the number of bytes used for each
counter:

* linear, 4-bytes
* log16, 2-bytes
* log8, 1-byte

Each element is associated with `d` counters, one in each of the
`d` rows, by `d` independent hash functions that map the element to
a particular counter (column) in a given row. We use FastHash64 as our hash
function.

For a basic count-min sketch (we implement a modified version), when an element
is added to the sketch simply increment its associated `d` counters by one.
To get an element's estimated count, simply return the minimum value stored in
the element's `d` counters.

The width sets the collision rate where two elements map to the same counter in
a given row. A larger width reduces the collision rate. When a collision occurs
the counter will report an estimated count that is higher than the true count.
By using more rows, we ensure that the same two elements will not collide in
the other rows due to the indepence of the hash functions. Therefore, by taking
the minimum of an element's counters we ensure that the error due to collisions
is minimized. In practice, a good rule of thumb is to have the width greater
than or equal to half the cardinality of the data.

The count-min sketches are implemented as Numba classes. The convenience
function :code:`CountMin()` is the recommended way to instantiate a count-min sketch
object. The Numba classes are called :code:`CountMinLinear`, :code:`CountMinLog16`,
:code:`CountMinLog8`.

Usage
-----

::

    import numpy as np
    from sketchnu.countmin import CountMin, save, load

    # Create 100k random 2-byte keys
    keys = [bytes(r) for r in np.random.randint(0, 256, (100000, 2), np.uint8)]

    # Instantiate a linear cms with width = 2**17 and depth = 8
    cms = CountMin('linear', width=2**17, depth=8)
    for key in keys:
        cms.add(key)

    # Get the number of times the first element was seen
    n_key_0 = cms.query(keys[0])

    # You can also get the estimated count with
    n_key_0 = cms[keys[0]]

    print(f'{keys[0]} was seen {n_key_0} times')
    
    # Instantiate a second count-min sketch with same parameters
    cms2 = CountMin('linear', width=2**17)
    for key in keys:
        cms2.add(key)
    
    # Merge the second into the first
    cms.merge(cms2)
    n_key_0_merge = cms.query(keys[0])
    print(f'{keys[0]} now seen {n_key_0_merge} times after merging')

    # Get the total number of elements added to the sketches
    print(f'Total elements added = {cms.n_added()}')

    # Save to disk
    save(cms, '/path/to/save/cms.npz')
    cms_load = load('/path/to/save/cms.npz')

The Details
-----------

The original algorithm was published in "An Improved Data Stream Summary:
The Count-Min Sketch and its Applications" by Cormode and Muthukrishnan in 2005.
In this paper the authors prove that the estimated count for a given element has
the following guarantees:

* The true count is less than or equal to the estimated count
* With probability of at least 1 - exp(-d), the estimate <= true + N * exp(1) / w

where `d` is the depth, `w` is the width, and `N` is the total number of elements
(including duplicates) added to the sketch. So by increasing the width you
reduce the error and by increasing the depth you increase the probability that
you do not exceed the specified error limit.

Over time numerous variations on the original algorithm were developed. See the
2012 paper "Sketch Algorithms for Estimating Point Queries in NLP" by Goyal,
Daume, and Cormode for a good summary.

We have chosen to implement a variation published in the 2015 paper
"Count-Min-Log sketch: Approximating counting with approximate counters" by
Pitel and Fouquier. In this paper, the authors use a variant of the count-min
sketch with conservative updating that uses log-based, approximate counters
instead of linear counters to improve the average relative error at a constant
memory footprint.

A conservative update only increases the counts in the sketch by the minimum
amount needed to ensure the estimate remains accurate. This was applied to
count-min sketches by Goyal and Daume in "Approximate scalable bounded space
sketch for large data NLP" in 2011. Instead of incrementing each of the `d`
counters associated with an element, only those counters that equal the minimum
value of the `d` counters is updated. This helps to reduce the error at the
expense of finding the minimum count value for the `d` counters before
incrementing. This seems a small price to pay for improved performance.
Unfortunately, there are no guarantees that have been proven with conservative
updating, but at least the error is never increased and empirical studies show
a reduction in the error.

The paper "Approximate counting: a detailed analysis" by Flajolet in 1985
describes the behavior of the `w*d` log-based counters used in the count-min
sketch (log8 & log16 variants). When adding an element to the sketch, if any of
the `d` counters associated with the element are to be updated, then those
counters are incremented with probability `x\*\*(-c)` where `x` is the base of the
log counter and `c` is the current value stored in that log counter. The paper
shows that there is an unbiased estimate of the count `N` given by
`(x\*\*c - 1)/(x - 1)` that has a variance of `(x-1)N(N-1)/2`. A more recent
analysis (2020) in "Optimal bounds for approximate counting" by Nelson and Yu
show that these counters are optimal in terms of space (memory) required.

We have also added one additional feature that is not discussed in the paper.
We split the values of our log-based counters into two parts. At the low end,
[0, num_reserved], we use linear counting and from (num_reserved, max_count] we
use the log-based counters. This allows for more accurate counting at the low
end at the expense of less accuracy on the high end. The default values for
num_reserved are 15 and 1023 for the 1-byte and 2-byte versions, respectively.

For the log-based counters we also set a max_count. This is the maximum value
that you want to be able to count to for any given element. By default, this is
set to 2\*\*32 - 1 to match the limit of the linear version. This is useful if you
plan to filter out elements that have too high of a count. For example, if you
are going to filter out elements that have been observed 100M times or more,
then there is no need for the max_value to be 2\*\*32-1. Instead you can set it to
be 100M which will improve the accuracy of your log counters, by using a smaller
base `x`, since a smaller range of values is now covered by the same number of
bits. The base x of the log counters is determined by the num_reserved and the
max_count.

To get an estimated count for an element, the following steps are done::

    # Return the minimum value stored in the d counters associated with key
    c = get_min_c(key)
    if c <= num_reserved:
        return c
    else:
        cprime = c - num_reserved
        return (x**cprime - 1) / (x - 1) + num_reserved

where `x` is the base of the log counters.

For the linear count-min sketch, merging two sketches (must have the same `w`
& `d`) is simply a matter of an element-wise sum between the two arrays of
counters; ensuring that you do not exceed the 4-byte maximum value of 2\*\*32 - 1.
For the log-based count-min sketch, it is a bit more complicated. For each of
the `w*d` counters, convert the stored log value into the corresponding
estimated count value. Add the two values together to get the value v that is
to be estimated in the merged log counter:

* If v <= num_reserved
    * store v in the log counter
* If v >= max_count
    * store the max uint value in the log counter
* Else
    * find the corresponding log values that bound v and round to the nearest log value

Testing
-------

Given that these are probablistic in nature, writing traditional software tests
is a bit challenging. We have written statistical tests that should pass the
vast majority of the time. The tests can be found in tests.py

We start by testing that the error guarantees are met for the linear cms. This
test is found in :code:`test_cms_linear()`.

For the log8 and log16 versions, we have two separate tests. The first is checking
that the log updating provides an unbiased estimate. In order to limit the biased
errors introduced by collisions in the count-min sketch, we set the width equal
to the total number of elements inserterd times 3. This means that with a
probability of at least 1 - exp(-d) that the estimate <= true + exp(1) / 3. We
use a t-test to test the null hypothesis that the mean of the difference
between the true count and the estimated count is 0. The test asserts that we
should fail to reject the null hypothesis at a 99% confidence level. These
tests are located in :code:`test_cms_log8_update()` and
:code:`test_cms_log16_update()`.

The second test ensures that merging two count-min sketches together give an
unbiased estimate. Again, to limit the biased errors introduced by collisions
in the count-min sketch, we set the width equal to the total number of elements
inserted times 3. We use a t-test to test the null hypothesis that the mean
of the difference between the true count and estimated count is 0. The test
asserts that we should fail to reject the null hypothesisat a 99% confidence
level. These tests are in :code:`test_cms_log8_merge()` and
:code:`test_cms_log16_merge()`.