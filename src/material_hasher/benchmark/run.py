from time import time
from typing import Callable, Iterable, Optional

from material_hasher.benchmark.test_cases import get_test_cases, get_tests

def load_structures():

    structures = []

    return structures


def benchmark_hasher(
    hasher: Callable,
    test_cases: Optional[Iterable[str]] = None,
    ignore_test_cases: Optional[Iterable[str]] = None,
) -> float:
    """Measure the performance of a hasher function based on test cases listed
      in the :module:`material_hasher.benchmark.test_cases` module.

    Parameters
    ----------
    hasher : Callable
        A function that takes a single argument, a dictionary of test data, and returns a hash.
    test_cases : Optional[Iterable[str]], optional
        _description_, by default None
    ignore_test_cases : Optional[Iterable[str]], optional
        _description_, by default None

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    all_test_cases = get_test_cases()
    if test_cases is not None:
        all_test_cases = [tc for tc in all_test_cases if tc in test_cases]
    if ignore_test_cases is not None:
        all_test_cases = [tc for tc in all_test_cases if tc not in ignore_test_cases]

    if not all_test_cases:
        raise ValueError("No test cases to run.")

    times = {}
    total_time = 0.0
    for test_case in all_test_cases:
        start_time = time()
        test_data = get_tests(test_case)
        hasher(test_data)
        end_time = time()
        times[test_case] = end_time - start_time
        total_time += times[test_case]
