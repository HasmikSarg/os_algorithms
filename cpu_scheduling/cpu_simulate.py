"""
CPU Scheduling Algorithms Simulator.
    registers, runs and compares process scheduling algorithms
    returns performance statistics based on time quantum and dataset
"""

import pandas as pd
import time
import os
from typing import List, Optional
from dataclasses import dataclass
from cpu_generate import Process, ProcessGenerator


@dataclass
class SimulationResult:
    """Result of a single simulation"""

    algorithm: str
    dataset_type: str
    process_count: int
    avg_waiting_time: float
    avg_turnaround_time: float
    avg_response_time: float
    avg_completion_time: float
    execution_time: float
    quantum: Optional[int] = None


class SchedulingSimulator(ProcessGenerator, Process):
    """
    Simulate CPU process scheduling algorithms
    """

    def __init__(self):
        super().__init__()

        self.algorithms: dict = {}
        self.quantum_values = [2, 4, 8, 16, 32]
        self.results_history = []

        self.input_datasets = []
        self.input_directory = os.path.join(os.path.dirname(__file__), 'input_data')
        self.output_directory = os.path.join(os.path.dirname(__file__), 'output_data')

        os.makedirs(self.input_directory, exist_ok=True)
        os.makedirs(self.output_directory, exist_ok=True)
    
    def register_algorithm(self, name: str, algorithm_class, **default_args):
        """
        Register a scheduling algorithm
        """

        self.algorithms[name] = {
            'class': algorithm_class,
            'default_args': default_args
        }

    def _load_algorithms(self):
        """Load available scheduling algorithms"""

        loaded_algorithms = []
        try:
            from RR import RoundRobin
            self.register_algorithm('RR', RoundRobin)
            loaded_algorithms.append('Round-Robin')
        except ImportError:
            print("Round-Robin not available")
        try:
            from SJF import SJFNonPreemptive
            self.register_algorithm('SJF', SJFNonPreemptive)
            loaded_algorithms.append('SJF Non-Preemptive')
        except ImportError:
            print("SJF Non-Preemptive not available")
        try:
            from SRTF import SJFPreemptive
            self.register_algorithm('SRTF', SJFPreemptive)
            loaded_algorithms.append('SJF Preemptive (SRTF)')
        except ImportError:
            print("SJF Preemptive (SRTF) not available")
        return loaded_algorithms
    
    def list_datasets(self) -> List[str]:
        """List available datasets"""

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
    
    def load_dataset(self, dataset_name: str) -> List[Process]:
        """Load dataset from CSV file"""

        dataset_folder = os.path.join(self.input_directory, dataset_name)
        csv_path = os.path.join(dataset_folder, f"{dataset_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File doesn't exist: {csv_path}")
        df = pd.read_csv(csv_path)
        processes = []
        for _, row in df.iterrows():
            process = Process(
                pid=int(row['PID']),
                arrival_time=int(row['Arrival_Time']),
                burst_time=int(row['Burst_Time']),
                priority=int(row['Priority'])
            )
            processes.append(process)
        
        return processes
        
    def run_algorithm_test(self, algorithm_name: str, processes: List[Process], 
                        dataset_type: str = "unknown", **kwargs) -> SimulationResult:
        """
        Run a single algorithm test
        """

        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        algorithm_info = self.algorithms[algorithm_name]
        algorithm_args = algorithm_info['default_args'].copy()
        algorithm_args.update(kwargs)
        
        algorithm = algorithm_info['class'](**algorithm_args)
        cpu_processes = [self._copy_process(p) for p in processes]
        start_time = time.time()
        try:
            completed_processes, stats = algorithm.simulate(cpu_processes)
            execution_time = time.time() - start_time
            result = SimulationResult(
                algorithm=algorithm_name,
                dataset_type=dataset_type,
                process_count=len(processes),
                avg_waiting_time=stats.get_value('avg_waiting_time'),
                avg_turnaround_time=stats.get_value('avg_turnaround_time'),
                avg_response_time=stats.get_value('avg_response_time'),
                avg_completion_time=stats.get_value('avg_completion_time'),
                execution_time=execution_time,
                quantum=kwargs.get('time_quantum')
            )

            return result 
        except Exception as e:
            print(f"Error executing {algorithm_name}: {str(e)}")

            return SimulationResult(
                algorithm=algorithm_name,
                dataset_type=dataset_type,
                process_count=len(processes),
                avg_waiting_time=float('inf'),
                avg_turnaround_time=float('inf'),
                avg_response_time=float('inf'),
                avg_completion_time=float('inf'),
                execution_time=time.time() - start_time,
                quantum=kwargs.get('time_quantum')
            )
    
    def run_dataset_test(self, processes: List['Process'],
                               dataset_type: str = "unknown", algorithms: Optional[List[str]] = None, **kwargs) -> List[SimulationResult]:
        """
        Run a single dataset test
        """

        results = []
        algorithms_to_test = algorithms if algorithms is not None else list(self.algorithms.keys())

        for algorithm in algorithms_to_test:
            if algorithm in self.algorithms:
                kwargs_clean = {k: v for k, v in kwargs.items() if k != 'algorithms'}
                result = self.run_algorithm_test(algorithm, processes, dataset_type, **kwargs_clean)
                results.append(result)
                self.results_history.append(result)
            else:
                print(f"Algorithm {algorithm} not available")
        return results

    def run_tests(self) -> pd.DataFrame:
        """
        Run tests for all available datasets and algorithms.
        """

        print("\n>>>>>> running simulations <<<<<")
        self.input_datasets = self.list_datasets()
        if not self.input_datasets:
            print("No datasets in input directory")
            ProcessGenerator().generate_datasets()
            self.input_datasets = self.list_datasets()

        all_results = []

        for dataset_name in self.input_datasets:
            processes = self.load_dataset(dataset_name)
            dataset_type = dataset_name.split('_')[0] if '_' in dataset_name else 'unknown'
            dataset_results = []
            for algorithm in self.algorithms:
                if algorithm == 'RR':
                    for quantum in self.quantum_values:
                        results = self.run_dataset_test(
                            processes=processes,
                            dataset_type=dataset_type,
                            algorithms=[algorithm],
                            time_quantum=quantum
                        )
                        dataset_results.extend(results)
                        all_results.extend(results)
                else:
                    results = self.run_dataset_test(
                        processes=processes,
                        dataset_type=dataset_type,
                        algorithms=[algorithm]
                    )
                    dataset_results.extend(results)
                    all_results.extend(results)
            self.generate_report(processes, dataset_results, dataset_name)

        return self.results_to_dataframe(all_results)

    
    def results_to_dataframe(self, results: List[SimulationResult]) -> pd.DataFrame:
        """Convert results to DataFrame"""

        data = []

        for result in results:
            data.append({
                'algorithm': result.algorithm,
                'dataset_type': result.dataset_type,
                'process_count': result.process_count,
                'avg_waiting_time': result.avg_waiting_time,
                'avg_turnaround_time': result.avg_turnaround_time,
                'avg_response_time': result.avg_response_time,
                'avg_completion_time': result.avg_completion_time,
                'execution_time': result.execution_time,
                'quantum': result.quantum
            })
        return pd.DataFrame(data)
    
    def _copy_process(self, process: 'Process') -> 'Process':
        """Copy a process"""

        return Process(
            pid=process.pid,
            arrival_time=process.arrival_time,
            burst_time=process.burst_time,
            priority=process.priority
        )

    def generate_report(self, cpu_processes, results: List[SimulationResult], dataset_name: str = ""):
        """
        Generate simulation report.
        """

        if not results:
            return "No simulation results."
        
        df = simulator.results_to_dataframe(results)
        print(f"\nComparison results ({dataset_name}):")

        df_table = df[['algorithm', 'quantum', 'avg_waiting_time', 'avg_turnaround_time', 'avg_response_time', 'avg_completion_time']].round(8).to_string(index=False)
        print('\n'.join(['\t' + line for line in df_table.split('\n')]))


if __name__ == "__main__":
    simulator = SchedulingSimulator()
    title = "CPU Scheduling Algorithms Simulation"
    print(f'\n{"=" * len(title)}')
    print(title)
    print(f'{"=" * len(title)}\n')

    generator = ProcessGenerator(seed=42)

    print(f"\t{', '.join(simulator._load_algorithms())}")
    print(f"\t{', '.join(simulator.list_datasets())}")

    results = simulator.run_tests()