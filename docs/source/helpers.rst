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
initialize the respective sketch(es). Simply leave one or two as the default
:code:`None` if you don't want that type of sketch.

The :code:`parallel_add` function takes a user-defined function,
:code:`process_q_item`. This function will take an item from the queue, process it as
desired, and then add the keys to the sketch(es). The function must take the following
arguments (q_item, \*sketches, \*\*kwargs) and return the number of records that were
processed. If you want to add to more than one sketch, then those must be listed in
alphabetical order since that is how :code:`parallel_add` will pass them.

When using :code:`parallel_add`, the sketches are placed into shared memory to allow
the spawned processes to access the sketches during data processing. The subsequent
call to :code:`parallel_merge` also leverages the shared memory in order to combine the
parallel sketches into a single final sketch which is what gets returned.

Usage
-----

Let's assume that we have a bunch of text files in a directory. Each line in
a file represents a single record. Each line (record) will be split by white
space which will then be added to the sketches. Remember that only bytes can
be added to a sketch, hence the :code:`w.encode('utf-8')`.

::

    from collections import Counter
    import logging
    from pathlib import Path
    from sketchnu.helpers import parallel_add

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    input_dir = Path('/path/to/text/files')
    files = input_dir.iterdir()

    # User defined function
    # Likely faster to combine keys across records to add to the sketches all together
    def process_q_item(filepath:str, cms, hh, hll, lowercase:bool=False) -> int:
        with open(filepath) as f:
            n_records = 0
            counter = Counter()
            for line in f:
                if lowercase:
                    words = line.strip().lower().split()
                else:
                    words = line.strip().split()
                counter.update([w.encode("utf-8") for w in words])
                n_records += 1
            cms.update(counter)
            hh.update(counter)
            hll.update(counter)
        return n_records
    
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
        lowercase=True,
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

