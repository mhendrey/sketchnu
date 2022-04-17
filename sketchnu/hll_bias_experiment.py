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


Code to replicate determining the bias corrections to low cardinality in the
hyperloglog algorithm.  The paper references doing 200 data points with
5000 experiments at each, for each of the different precision levels which for
sketchnu is [7,16].

We used 200 data points with 50,000 experiments for each precision level. Took
just under 2hrs on a small machine with 4-cores.

Bias is removed if cardinality is between [low, high] where high is 5*2^p

+------+--------+---------+
|   p  |   low  |   high  |
+------+--------+---------+
|   7  |     80 |     640 |
+------+--------+---------+
|   8  |    220 |   1,280 |
+------+--------+---------+
|   9  |    400 |   2,560 |
+------+--------+---------+
|  10  |    900 |   5,120 |
+------+--------+---------+
|  11  |  1,800 |  10,240 |
+------+--------+---------+
|  12  |  3,100 |  20,480 |
+------+--------+---------+
|  13  |  6,500 |  40,960 |
+------+--------+---------+
|  14  | 11,500 |  81, 920|
+------+--------+---------+
|  15  | 20,000 | 163,840 |
+------+--------+---------+
|  16  | 50,000 | 327,680 |
+------+--------+---------+

"""
import argparse
from datetime import datetime
from multiprocessing import Pool
from numba import njit, uint8, uint64, int64, float64, prange
from numba.typed import List
import numpy as np

from sketchnu.hyperloglog import HyperLogLog_nu

# Empirically determined threshold provided by the paper authors
# at http://goo.gl/iU8Ig).
sub_algorithm_threshold = np.array(
    [80, 220, 400, 900, 1800, 3100, 6500, 11500, 20000, 50000], dtype=np.int64
)


def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5000,
        help="Number of trials to run for each (p,v) pair. Default is 5000",
    )
    parser.add_argument(
        "--n_measurements",
        type=int,
        default=200,
        help="Number of measurements to do for each precision. Default is 200",
    )
    return parser.parse_args()


@njit(nogil=True)
def one_trial(keys, p, seed, lower, upper, n_measurements):
    """
    Numba function to run a single trial for the given precision level.
    Measurements for the estimated value and its bias are collected at
    `n_measurements` locations evenly spaced between [`lower`, `upper`] as the
    `keys` are added.

    Parameters
    ----------
    keys : numba.typed.List(bytes)
        Keys to be inserted into the HyperLogLog
    p : int
        Precision of the HyperLogLog
    seed : int
        Seed of the HyperLogLog
    lower : int
        Take first measurement when this many keys have been added
    upper : int
        Take last measurement when this many keys have been added
    n_measurements : int
        Number of points between [lower, upper] to record estimate & bias
    
    Returns
    -------
    raw_estimate : np.array, shape=(n_measurements,)
        Array of the raw estimate values at each of the n_measurements
    bias: np.ndarray, shape=(n_measurements,)
        Array of the measured bias values at each of the n_measurements

    """
    p = uint64(p)
    hll = HyperLogLog_nu(p, seed)
    lower = int64(lower)
    upper = int64(upper)

    measurement_points = np.linspace(lower, upper, n_measurements).astype(uint64)

    raw_estimate = np.zeros(n_measurements, float64)
    bias = np.zeros(n_measurements, float64)

    ptr = 0
    for n in range(upper):
        hll.add(keys[n])
        true_count = n + 1
        if true_count == measurement_points[ptr]:
            estimate = hll._estimation_function()
            raw_estimate[ptr] = estimate
            bias[ptr] = estimate - true_count
            ptr = ptr + 1

    return raw_estimate, bias


@njit(nogil=True, parallel=True)
def one_precision(keys, p, n_trials, lower, upper, n_measurements):
    """
    Runs `n_trials` runs for the given precision level. Measurements for the
    estimated value and its bias are collected at `n_measurements` locations
    evenly spaced between [lower, upper] as the keys are added. Note: This is
    run with `parallel=True`

    Parameters
    ----------
    keys : numba.typed.List(bytes)
        Keys to be inserted into the HyperLogLogs
    p : int
        Precision of the HyperLogLogs
    n_trials : int
        Number of independent trials to perform. Each trial will use the
        same keys, but different seeds for the HyperLogLogs
    lower : int
        Take first measurement when this many keys have been added
    upper : int
        Take last measurement when this many keys have been added
    n_measurements : int
        Number of points between [lower, upper] to record estimate & bias
    
    Returns
    -------
    raw_estimates : np.ndarray, shape=[n_trials, n_measurements]
        2-d array of the raw estimated values at each of the n_measurements
        and n_trials. Average over the n_trials to get final values
    biases : np.ndarray, shape=[n_trials, n_measurements]
        2-d arry of the bias values at each of the n_measurements and n_trials.
        Average over the n_trials to get the final values.

    """
    raw_estimates = np.zeros((n_trials, n_measurements), float64)
    biases = np.zeros((n_trials, n_measurements), float64)

    for i in prange(n_trials):
        raw_estimates[i], biases[i] = one_trial(
            keys, p, i, lower, upper, n_measurements
        )

    return raw_estimates, biases


def main():
    args = parse_cmd_line()
    n_trials = args.n_trials
    n_measurements = args.n_measurements

    # Store the mean values for each precision at each measurement
    raw_estimate = np.zeros((10, n_measurements))
    bias_data = np.zeros((10, n_measurements))

    start = datetime.now()
    for p in range(7, 17):
        start_p = datetime.now()
        print(f"{start_p}: Starting precision {p}")

        # These limits are from the HyperLogLog++ paper
        lower = sub_algorithm_threshold[p - 7]
        upper = 5 * 2 ** p

        # Make random 16-byte keys. Make sure no duplicates.
        keys = [bytes(r) for r in np.random.randint(0, 256, (upper, 16), np.uint8)]
        while len(keys) != len(set(keys)):
            keys = [bytes(r) for r in np.random.randint(0, 256, (2 ** p, 16), np.uint8)]
        keys = List(keys)

        # Run the experiments for this precision level
        # Each experiment uses a different seed value
        raw, bias = one_precision(keys, p, n_trials, lower, upper, n_measurements)
        raw_estimate[p - 7] = raw.mean(0)
        bias_data[p - 7] = bias.mean(0)
        end_p = datetime.now()
        print(f"{end_p}: Finished precision {p} in {end_p-start_p}")

    end = datetime.now()
    speed = n_trials / (end - start).total_seconds()
    print(f"{end}: Took {end-start} at {speed:.3f} trials / second")

    # Write results to a file that can easily be imported
    with open("hll_constants.py", "w") as f:
        f.write("import numpy as np\n")

        f.write("\nsub_algorithm_threshold = np.array([\n")
        f.write("    ")
        f.write(", ".join([str(i) for i in sub_algorithm_threshold]))
        f.write("\n], dtype=np.int64)\n")

        f.write("\nraw_estimate = np.array([\n")
        for i in range(7, 16):
            f.write(f"    # Precision {i}\n")
            f.write("    [")
            f.write(", ".join([f"{x:.5f}" for x in raw_estimate[i - 7, :]]))
            f.write("],\n")
        f.write(f"    # Precision 16\n")
        f.write("    [")
        f.write(", ".join([f"{x:.5f}" for x in raw_estimate[-1, :]]))
        f.write("]\n")
        f.write("])\n")

        f.write("\nbias_data = np.array([\n")
        for i in range(7, 16):
            f.write(f"    # Precision {i}\n")
            f.write("    [")
            f.write(", ".join([f"{x:.5f}" for x in bias_data[i - 7, :]]))
            f.write("],\n")
        f.write(f"    # Precision 16\n")
        f.write("    [")
        f.write(", ".join([f"{x:.5f}" for x in bias_data[-1, :]]))
        f.write("]\n")
        f.write("])\n")


if __name__ == "__main__":
    main()
