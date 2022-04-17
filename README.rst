.. image:: images/logo.png
    :align: center

==================================

Numba implementations of some sketch algorithms. Current algorithms implemented
are the HyperLogLog++ (approximates cardinality), conservative updating count-min
sketch with either linear or log counters (approximates count of given keys),
and the top-k api, aka heavy-hitters, (approximates identification of most frequented
items in a stream). 


Documentation
=============
Documentation for the API and theoretical foundations of the algorithms can be
found at hxxp://some.webaddress

Installation
============
Sketchnu may be installed using conda::

    conda install -c conda-forge sketchnu
