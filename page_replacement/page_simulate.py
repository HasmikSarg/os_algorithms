"""
Page replacement algorithm simulator
    registers, runs and compares page replacement algorithms
    returns performance statistics of algorithms based on frame count and dataset
"""

import pandas as pd
import time
import os
from typing import List
from dataclasses import dataclass
from page_generate import PageGenerator, Pages


@dataclass
class SimulationResult:
    """Result of a single simulation."""

    algorithm: str
    frame_count: int
    sequence_type: str
    sequence_length: int
    page_faults: int
    page_hits: int
    hit_percentage: float
    execution_time: float


class PageReplacementSimulator(PageGenerator, Pages):
    """
    Simulate page replacement algorithms
    """
    
    def __init__(self):
        self.algorithms: dict = {}
        self.frame_counts = [2, 4, 8, 16, 32]
        self.results_history = []

        self.input_datasets = []
        self.input_directory = os.path.join(os.path.dirname(__file__), 'input_data')
        self.output_directory = os.path.join(os.path.dirname(__file__), 'output_data')

        os.makedirs(self.input_directory, exist_ok=True)
        os.makedirs(self.output_directory, exist_ok=True)

    def register_algorithm(self, name: str, algorithm_class):
        """
        Register a new page replacement algorithm.
        """

        self.algorithms[name] = algorithm_class

    def _load_algorithms(self) -> List[str]:
        """
        Return a list of available page replacement algorithms.
        """

        loaded_algorithms = []

        try:
            from LFU import LFU
            self.register_algorithm('LFU', LFU)
            loaded_algorithms.append('LFU')
        except ImportError:
            print("LFU not available")
        try:
            from LRU import LRU
            self.register_algorithm('LRU', LRU)
            loaded_algorithms.append('LRU')
        except ImportError:
            print("LRU not available")
        return list(self.algorithms.keys())

    def show_datasets(self) -> List[str]:
        """
        Return a list of available page sequence datasets (folders with CSV files).
        """

        if not os.path.exists(self.input_directory):
            return []
        dataset_folders = []
        for item in os.listdir(self.input_directory):
            item_path = os.path.join(self.input_directory, item)
            if os.path.isdir(item_path):
                csv_file = os.path.join(item_path, f"{item}.csv")
                if os.path.exists(csv_file):
                    dataset_folders.append(item)
        return sorted(dataset_folders)

    def load_dataset(self, dataset_name: str) -> List[int]:
        """
        Load page sequence from selected dataset (CSV file).
        """

        dataset_folder = os.path.join(self.input_directory, dataset_name)
        csv_path = os.path.join(dataset_folder, f"{dataset_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File does not exist: {csv_path}")
        df = pd.read_csv(csv_path)
        return df['page_number'].tolist()

    def run_algorithm_test(self, algorithm_name: str, frame_count: int, 
                       sequence: List[int], sequence_type: str = "unknown") -> SimulationResult:
        """
        Run a single algorithm test.
        """

        if algorithm_name not in self.algorithms:
            raise ValueError(f"unknown algorithm: {algorithm_name}")
        
        algorithm = self.algorithms[algorithm_name](frame_count)

        start_time = time.time()
        page_faults, page_hits = algorithm.simulate(sequence)
        execution_time = time.time() - start_time

        total_accesses = len(sequence)
        hit_percentage = (page_hits / total_accesses * 100) if total_accesses > 0 else 0
        return SimulationResult(
            algorithm=algorithm_name,
            frame_count=frame_count,
            sequence_type=sequence_type,
            sequence_length=total_accesses,
            page_faults=page_faults,
            page_hits=page_hits,
            hit_percentage=hit_percentage,
            execution_time=execution_time
        )
    
    def run_dataset_test(self, algorithms: List[str], frame_counts: List[int],
                           sequence: List[int], sequence_type: str = "unknown") -> List[SimulationResult]:
        """
        Compare multiple algorithms on the same sequence.
        """

        results = []
        for algorithm in algorithms:
            for frame_count in frame_counts:
                result = self.run_algorithm_test(algorithm, frame_count, sequence, sequence_type)
                results.append(result)
        return results
    
    def run_tests(self) -> pd.DataFrame:
        """
        Run tests for all available datasets and algorithms.
        """

        print("\n>>>>>> running simulations <<<<<")
        input_datasets = self.show_datasets()
        if not input_datasets:
            print("No datasets found in input directory")
            return pd.DataFrame()

        all_results = []

        for dataset_name in input_datasets:
            sequence = self.load_dataset(dataset_name)
            sequence_type = dataset_name.split('_')[0] if '_' in dataset_name else 'unknown'
            algorithms = self._load_algorithms()
            results = self.run_dataset_test(
                algorithms=algorithms,
                frame_counts=self.frame_counts,
                sequence=sequence,
                sequence_type=sequence_type
            )
            all_results.extend(results)
            self.generate_report(results, dataset_name)

        return self.results_to_dataframe(all_results)
    
    def results_to_dataframe(self, results: List[SimulationResult]) -> pd.DataFrame:
        """Convert results to DataFrame."""

        data = []
        for result in results:
            data.append({
                'algorithm': result.algorithm,
                'frame_count': result.frame_count,
                'sequence_type': result.sequence_type,
                'sequence_length': result.sequence_length,
                'page_faults': result.page_faults,
                'page_hits': result.page_hits,
                'hit_percentage': result.hit_percentage,
                'execution_time': result.execution_time
            })
        return pd.DataFrame(data)
    
    def generate_report(self, results: List[SimulationResult], dataset_name: str = ""):
        """
        Generate a report of page replacement algorithm simulations.
        """

        if not results:
            return "No simulation results available."
        
        df = self.results_to_dataframe(results)
        print(f"\nComparison results ({dataset_name}):")

        df_table = df[['algorithm', 'sequence_length', 'frame_count', 'sequence_type', 'page_faults', 'page_hits', 'hit_percentage', 'execution_time']].round(8).to_string(index=False)
        print('\n'.join(['\t' + line for line in df_table.split('\n')]))


if __name__ == "__main__":
    simulator = PageReplacementSimulator()

    title = "Simulating page replacement algorithms"
    print(f'\n{"=" * len(title)}')
    print(title)
    print(f'{"=" * len(title)}\n')

    generator = PageGenerator(seed=42)

    print(f"\t{', '.join(simulator._load_algorithms())}")
    print(f"\t{', '.join(simulator.show_datasets())}")

    results = simulator.run_tests()