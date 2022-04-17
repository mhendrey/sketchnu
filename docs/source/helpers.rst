Helper Functions
================

We have written a few helper functions to aid in parallelizing the creation of
HyperLogLogs and/or count-min sketches. The functions will spin up multiple
independent workers, each of which has its own sketch to add its portion of the
data to be processed. Once all the data has been processed, then the individual
sketches will be merged efficiently in successive rounds of merging to achieve
the final sketch.

There are separate functions to parallelize the creation of both HyperLogLogs and
count-min sketches, :code:`parallel_hll` and :code:`parallel_cms`, respectively.
Since reading large amounts of data from disk is an expensive operation, the
:code:`parallel_cms` can also be given arguments that will create both a
count-min sketch and a HyperLogLog at the same time.

The helper functions take a user-defined function, :code:`process_q_item`, which
turns an item from the queue into an iterable of records with each record being
an iterable of elements (bytes) that are to be added to the sketch.

Usage
-----

Let's assume that we have a bunch of text files in a directory. Each line in
a file represents a single record. Each line (record) will be split by white
space which will then be added to the sketches. Remember that only bytes can
be added to a sketch, hence the :code:`word.encode('utf-8')`.

The :code:`process_q_item()` method must yield records where each record is an
iterable of bytes.

::

    import logging
    from pathlib import Path
    from sketchnu.helpers import parallel_cms, parallel_hll

    input_dir = Path('/path/to/text/files')
    files = input_dir.iterdir()

    # Define the function that takes a queue item and yields records with each
    # record a list of elements to be added to the sketch
    def process_q_item(filepath:str):
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                record = [w.encode('utf-8') for w in line.split()]
                yield record
    
    cms = parallel_cms(files, process_q_item, 'linear', 2**20)

    # To see total number of elements added
    print(f'A total of {cms.n_added():,} elements have been added')
    # To see total number of records added
    print(f'{cms.n_records():,} records were processed')

    # files is an iterator that has now been exhausted. Let's make it a list
    files = list(input_dir.iterdir())
    hll = parallel_hll(files, process_q_item, log_console_level=logging.DEBUG)

    # You can make both at the same time
    cms, hll = parallel_cms(files, process_q_item, 'linear', 2**20, hll_p=16, n_workers=4)

