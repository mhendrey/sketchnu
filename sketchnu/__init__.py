"""
Sketchnu has Numba implementations of sketch algorithms and other useful functions 
that utilize hash functions.

Copyright (C) 2022 Matthew Hendrey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
__version__ = "1.2.0"

from sketchnu.countmin import (
    CountMin,
    load,
    CountMinLinear,
    CountMinLog16,
    CountMinLog8,
)
from sketchnu.hashes import fasthash64, fasthash32, murmur3
from sketchnu.heavyhitters import HeavyHitters
from sketchnu.helpers import (
    parallel_add,
    parallel_merging,
    setup_logger,
    attach_shared_memory,
)
from sketchnu.hyperloglog import HyperLogLog
