Helper Functions
================

We have written a few helper functions to aid in parallelizing the creation of one
or more of the sketches. The functions will spin up multiple independent workers, each
of which has its own sketch(s) to add its portion of the data to be processed. Once all
the data has been processed, then the individual sketches will be merged efficiently in
successive rounds of merging to achieve the final sketch for each type.

The main helper function, :code:`parallel_add`, can add the data to one or more of the
sketches at the same time. You just need to specify one or more of :code:`cms_args`,
:code:`hh_args`, or :code:`hll_args`, which provide the necessary parameters to
initialize the respective sketch. Simply leave one or two as the default :code:`None`
if you don't want that type of sketch.

The :code:`parallel_add` function takes a user-defined generator function,
:code:`process_q_item`. The generator takes an item from the queue and yields a tuple.
The first element is a Dict[bytes, int] | Iterable[bytes] to be added to the sketch(s).
The second element of the tuple is the number of records in the item. Using a Dict is
often faster if the data has multiple entries for a given key (bytes).

When using :code:`parallel_add`, the sketches are placed into shared memory to allow
the spawned processes to access the sketches during data processing. The subsequent
call to :code:`parallel_merge` also leverages the shared memory in order to combine the
parallel sketches into a single final sketch which is what gets returned.

Usage
-----

Let's assume that we have a bunch of text files in a directory. Each line in
a file represents a single record. Each line (record) will be split by white
space which will then be added to the sketches. Remember that only bytes can
be added to a sketch, hence the :code:`word.encode('utf-8')`.

::

    from collections import Counter
    import logging
    from pathlib import Path
    from sketchnu.helpers import parallel_add

    input_dir = Path('/path/to/text/files')
    files = input_dir.iterdir()

    # Define the generator function
    # Likely faster to combine keys to add to the sketches all together as done here
    def process_q_item(filepath:str):
        with open(filepath) as f:
            n_records = 0
            counter = Counter()
            for line in f:
                words = line.strip().split()
                counter.update([w.encode("utf-8") for w in words])
                n_records += 1
            yield counter, n_records
    
    cms_args = {"cms_type": "linear", "width": 2**20}
    hh_args = {"width": 50, "max_key_len": 16}
    hll_args = {"p": 16}

    cms, hh, hll = parallel_add(
        files,
        process_q_item,
        n_workers=4,
        cms_args=cms_args,
        hh_args=hh_args,
        hll_args=hll_args,
    )

    # To see total number of elements added
    print(f'A total of {cms.n_added():,} elements have been added')
    # To see total number of records added
    print(f'{cms.n_records():,} records were processed')
    # To see how many times had "the" appears in the text
    print(f"'the' appears {cms['the'.encode('utf-8')]} times")
    # To see how many distinct words in all the data
    print(f"{hll.query():.1f} unique words")
    # To see the most common word
    print(f"{hh.query(1)} is the most commonly seen word")

