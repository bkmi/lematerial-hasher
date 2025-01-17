import datetime
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import tqdm
import yaml
from pymatgen.core import Structure

from material_hasher.benchmark.run_transformations import get_hugging_face_dataset
from material_hasher.benchmark.utils import get_structure_from_hf_row
from material_hasher.hasher import HASHERS
from material_hasher.similarity import SIMILARITY_MATCHERS
from material_hasher.types import StructureEquivalenceChecker

STRUCTURE_CHECKERS = {**HASHERS, **SIMILARITY_MATCHERS}


def get_benchmark_data(n_structures: int, seed: Optional[int] = 0) -> List[Structure]:
    """Get the benchmark data for the given number of structures.

    Parameters
    ----------
    n_structures : int
        Number of structures to use for benchmarking.

    Returns
    -------
    structures : List[Structure]
        List of structures for benchmarking.
    """

    np.random.seed(seed)
    hf_dataset = get_hugging_face_dataset()
    random_indices = np.random.choice(len(hf_dataset), n_structures, replace=False)
    hf_dataset = hf_dataset.select(random_indices).to_pandas()
    structures = [get_structure_from_hf_row(row) for _, row in hf_dataset.iterrows()]  # type: ignore
    return structures


def benchmark_time(
    structure_checker: StructureEquivalenceChecker,
    seeds: List[int] = [0, 1, 2, 3, 4],
) -> Tuple[pd.DataFrame, float]:
    """Benchmark the disordered structures using the given hasher or similarity matcher.

    Parameters
    ----------
    hasher : StructureEquivalenceChecker
        Hasher to use for benchmarking.

    Returns
    -------
    df_results : pd.DataFrame
        Dataframe containing the results of the benchmarking.
    total_time : float
        Total time taken for the benchmark
    """
    print("Downloading benchmark data...")
    structures = get_benchmark_data(1000)

    dissimilar_structures_unique_structures = [
        get_dissimilar_structures(structures, seed) for seed in seeds
    ]
    dissimilar_structures = [
        dissimilar_structures
        for dissimilar_structures, _ in dissimilar_structures_unique_structures
    ]
    unique_structures = [
        unique_structures
        for _, unique_structures in dissimilar_structures_unique_structures
    ]
    results = defaultdict(dict)

    start_time = time.time()
    print("\n\n-- Dissimilar Structures --")
    dissimilar_metrics = get_classification_results_dissimilar(
        structure_checker, dissimilar_structures, unique_structures
    )
    results["dissimilar_case"] = dissimilar_metrics
    print(
        f"Success rate: {np.mean(dissimilar_metrics['success_rate']) * 100:.2f}%"
        + r" $\pm$ "
        + f"{np.std(dissimilar_metrics['success_rate']) * 100:.2f}%"
    )

    print("Benchmarking disordered structures...")
    for group, structures in tqdm.tqdm(structures.items()):
        metrics = run_group_structures_benchmark(structure_checker, group, structures)
        results[group] = metrics
        print(
            f"Success rate: {(np.mean(metrics['success_rate']) * 100):.2f}%"
            + r" $\pm$ "
            + f"{(np.std(metrics['success_rate']) * 100):.2f}%"
        )
    total_time = time.time() - start_time
    results["total_time (s)"] = total_time  # type: ignore

    df_results = pd.DataFrame(results).T
    print(df_results)

    return df_results, total_time


def main():
    """
    Run the benchmark for the disordered structures.

    This function provides a command-line interface to benchmark hashers and similarity matchers.

    Get help with:

    .. code-block:: bash

        $ python -m material_hasher.benchmark.run_disordered --help
    """
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Benchmark hashers and similarity matchers for disordered structures."
    )
    parser.add_argument(
        "--algorithm",
        choices=list(STRUCTURE_CHECKERS.keys()) + ["all"],
        help=f"The name of the structure checker to benchmark. One of: {list(STRUCTURE_CHECKERS.keys()) + ['all']}",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for the results. Default: 'results/'",
        default="results/",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Name of the .yaml configuration file to use for the hyperparameters of the hasher. Defaults to default.yaml",
        default="default.yaml",
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(Path("configs") / args.config, "r"))
    output_path = Path(args.output_path) / datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    os.makedirs(output_path, exist_ok=True)
    yaml.dump(config, open(output_path / "config.yaml", "w"))

    if args.algorithm not in STRUCTURE_CHECKERS and args.algorithm != "all":
        raise ValueError(
            f"Invalid algorithm: {args.algorithm}. Must be one of: {list(STRUCTURE_CHECKERS.keys()) + ['all']}"
        )

    all_results = {}
    for structure_checker_name, structure_checker_class in STRUCTURE_CHECKERS.items():
        if args.algorithm != "all" and structure_checker_name != args.algorithm:
            continue

        structure_checker = structure_checker_class(
            **config.get(structure_checker_name, {})
        )
        df_results, structure_checker_time = benchmark_disordered_structures(
            structure_checker
        )
        df_results.to_csv(
            output_path / f"{structure_checker_name}_results_disordered.csv"
        )
        all_results[structure_checker_name] = df_results
        print(f"{structure_checker_name}: {structure_checker_time:.3f} s")

    if args.algorithm == "all":
        all_results = pd.concat(all_results, names=["algorithm"])
        all_results.to_csv(output_path / "all_results_disordered.csv")


if __name__ == "__main__":
    main()
