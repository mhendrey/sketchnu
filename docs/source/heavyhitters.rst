Heavy-Hitters
=============

The Basics
----------

Sketch implementation of the phi-heavy hitters algorithm which identifies all the
elements in a data stream that are observed in at least phi fraction of the records.
This assumes that elements have a fat-tailed distribution in the data stream. This is
an implementation of the Topkapi algorithm.

Similar to a count-min sketch, a heavy-hitter uses a 2-d array with a given width `w`
and depth `d`, but unlike a count-min sketch the heavy-hitter must also keep track of
which elements are associated with a given cell in the 2-d array. The Topkapi algorithm
does this by adding a byte string (storing the corresponding element) and another
counter to each cell in the 2-d array. This is in addition to counter associated with
the count-min sketch.

When an element is to be added to the sketch, you do the following for each of the `d`
rows. The element is hashed which identifies which column in the row the element is
associated with. Increment the count-min sketch counter at that row/column. If the element
matches the byte string stored in that row/column, then increment the secondary
counter. If the element does not match the byte string stored in that row/column, then
decrement the counter. If the counter goes to zero, then set the byte string to be the
element and increment the secondary counter.

To identify which elements are the most frequent, a candidate set of elements are taken
from the 2-d array of elements stored in the byte strings. For each of these elements,
you query the count-min sketch to determine their relative ranks to each other. Return
only those elements whose frequency of occurency is greater than or equal to phi in
sorted decreasing order.

For practical reasons, the count-min sketch counters are removed in order to reduce the
memory footprint even more. The secondary counter is then used to estimate the count a
given element appears in the data stream. This secondary counter's
estimate <= true count <= cms estimate. This leads to returning fewer top k elements
than if the cms was retained. Thus this implementation is a more conservative estimate.

Usage
-----

::

    from collections import Counter
    import numpy as np
    from sketchnu.heavyhitters import HeavyHitters

    # Generate probabilities from zip distribution
    # with slope = -s
    def zipf(vocab_size:int, s:int=1):
        denom = 0.0
        for i in range(vocab_size):
            denom += 1 / (i + 1) ** s
        
        p = np.zeros(vocab_size, np.float64)
        for k in range(1, vocab_size + 1):
            p[k-1] = 1 / k ** s / denom
        
        return p / p.sum()
    
    # Number of distinct elements in stream
    vocab_size = 10_000
    # Size of the data stream
    N = 100_000
    # Zipf distribution with slope of -1
    probs = zipf(vocab_size)
    vocab = [f"{i:04}".encode("utf-8") for i in range(vocab_size)]

    stream = np.random.choice(vocab, N, p=probs).tolist()

    hh = HeavyHitters(
        width=100,
        depth=4,        # Default is 4
        max_key_len=4,  # Each vocab member is just 4-bytes
        phi=0.01,       # Defaults to 1 / width
    )

    # Add a single key to the sketch
    hh.add(vocab[0])

    # Add a single key multiple times to the sketch
    hh.add(vocab[0], 3)

    # Add a stream as a Dict; 
    hh.update(Counter(stream))
    # You could also just do hh.update(stream), but calling Counter is nearly
    # 9x faster in this example.

    # Get the top 10 most frequent elements whose counts >= phi * N
    result = hh.query(10)

    print(result)
    
    # Get the top 10 whose counts >= 800
    r_threshold = hh.query(10, 800)

    print(r_threshold)

    # Instantiate a second heavy-hitter with same parameters
    hh2 = HeavyHitters(**hh.args)

    # Add some more elements from a similar stream
    stream = np.random.choice(vocab, N, p=probs).tolist()

    hh2.update(Counter(stream))
    # Now merge
    hh.merge(hh2)

    # Notice counts ~2x from before merge
    result_merge = hh.query(5) 

    print(result_merge)

    # Save to disk
    hh.save("/path/to/save/hh.npz")
    hh_load = HeavyHitters.load("/path/to/save/hh.npz")


The Details
-----------

This is an implementation of the topkapi algorithm published from

A\. Mandal, H\. Jiang, A\. Shrivastava, and V\. Sarkar, "Topkapi: Parallel and Fast
Sketches for Finding Top-K Frequent Elements", Advances in Neural Information
Processing Systems **31**, (2018).

In this paper the authors prove that the topkapi algorithm satisfies the
phi-heavy hitters with the following guarantees:

* Misses to report an element with frequency >= phi * N with probability at most
  delta / 2
* Reports an element with frequency <= (phi - eps) * N with probability at most
  delta / 2

where eps = 1 / `w`, eps < phi, delta / 2 = exp(-`d`), and `N` is the total number of
records added to the sketch. By increasing the width, and thus decreasing eps, you may
track more elements. By increasing the depth, you decrease the probability that you
either fail to report an element, or that you include an element that you shouldn't.

The proofs of these guarantees used the additional count-min sketch that is part of the
topkapi algorithm. For practical considerations, the authors note that you can remove
the count-min sketch to reduce the memory footprint. Instead they suggest using the
incrementing & decrementing secondary counter as an estimate for the frequency of
occurrence of any given element. That is how the algorithm has been implemented in this
package.

A second practical consideration, that has **not** been implemented here, is to only
use the elements stored in the first row of the sketch, instead of all the rows, when
constructing the candidate set of elements. I found that this results in too many
elements dropping out of the candidate set. This happens because of common elements
colliding in the first row causing one to beat out the other for that particular column
bucket. By using all the rows, this reduces the likelihood of this type of error and
given that the typical depth is 4, the computational cost seems minimal.