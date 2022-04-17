.. sketchnu documentation master file, created by
   sphinx-quickstart on Sat Dec  5 22:03:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: ../../README.rst

Modules
=======

HyperLogLog++
-------------

A sketch that estimates the number of unique elements, cardinality, that have
been added into the sketch.

.. toctree::
   :maxdepth: 2

   hyperloglog

Count-Min Sketch
----------------
A sketch that estimates the number of times a given element has been added into
the sketch. This is similar to the Python collections.Counter, but a count-min
sketch does not store the keys, which for large datasets, can greatly reduce the
memory required.

.. toctree::
   :maxdepth: 2

   countmin

Helpers
-------
Functions to aid in parallelizing the creation of HyperLogLogs and/or count-min
sketches.

.. toctree::
   :maxdepth: 2

   helpers

Hashes
------
The **non-cryptographic** hash functions FastHash (both 32 & 64-bit) and
Murmur3 (32-bit) have been implemented here.

.. toctree::
   :maxdepth: 2

   hashes


.. toctree::
   :maxdepth: 2
   :caption: API

   modules

.. toctree::
   :maxdepth: 2
   :caption: Admin

   license
   help


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
