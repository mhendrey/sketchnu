Heavy-Hitters
=============

The Basics
----------

A sketch that estimates the top k most frequently observed elements that have been
added into the sketch.

Usage
-----

::

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
    
    top_k = 5

    # Number of distinct elements in stream
    vocab_size = 100000
    # Need to see enough for the top k to appear
    N = 1000000
    # Zipf distribution with slope of -1
    probs = zipf(vocab_size)
    vocab = [f"{i:05}".encode("utf-8") for i in range(vocab_size)]

    stream = np.random.choice(vocab, N, p=probs).tolist()
    hh = HeavyHitters(
        width=top_k*10, # Padding a little to reduce collisions 
        max_key_len=5,  # Each vocab member is just 5-bytes
    )
    hh.update(stream)

    # Get results for the approximate top k most common elements
    result = hh.query(top_k)

    print(result)
    
    # Get top k, but only if estimated count exceeds 10,000
    r_threshold = hh.query(top_k, 10000)

    print(r_threshold)

    # Instantiate a second heavy-hitter with same parameters
    hh2 = HeavyHitters(**hh.args)
    # Add same elements more elements from a similar stream
    stream = np.random.choice(vocab, N, p=probs).tolist()
    hh2.update(stream)
    # Now merge
    hh.merge(hh2)

    # Notice counts ~2x from before merge
    result_merge = hh.query(top_k) 

    print(result_merge)

The Details
-----------

This is an implementation of the topkapi algorithm published from

A\. Mandal, H\. Jiang, A\. Shrivastava, and V\. Sarkar, "Topkapi: Parallel and Fast
Sketches for Finding Top-K Frequent Elements", Advances in Neural Information
Processing Systems **31**, (2018).