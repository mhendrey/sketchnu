__version__ = "1.1.0"

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
