"""
CPU Process Sets Generator
    generates process sets for testing process scheduling algorithms
    saves results in input_data/ in separate directories as .csv and .json
"""

import numpy as np
import pandas as pd
import os
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Process:
    """Basic attributes of CPU process scheduling"""
    pid: int = 0
    arrival_time: int = 0
    burst_time: int = 0
    priority: int = 1
    completion_time: int = 0
    turnaround_time: int = 0
    waiting_time: int = 0
    response_time: int = -1
    remaining_time: int = 0
    start_time: int = -1
    has_started: bool = False
    
    def __post_init__(self):
        self.remaining_time = self.burst_time
    
    def calculate_times(self):
        """Calculate time metrics after process completion"""

        self.turnaround_time = self.completion_time - self.arrival_time
        self.waiting_time = self.turnaround_time - self.burst_time
        if self.start_time != -1:
            self.response_time = self.start_time - self.arrival_time
    
    def set_start_time(self, time: int):
        """Set time of first CPU allocation"""

        if not self.has_started:
            self.start_time = time
            self.has_started = True
    
    def __repr__(self):
        return f"P{self.pid} (AT:{self.arrival_time}, BT:{self.burst_time})"


@dataclass
class GanttEntry:
    """CPU process entry in Gantt chart"""

    process_id: int
    start_time: int
    duration: int
    
    @property
    def end_time(self):
        return self.start_time + self.duration


class SchedulingStatistics:
    """Performance statistics of CPU process scheduling algorithms"""
    
    def __init__(self, processes: List[Process], gantt_chart: List[GanttEntry]):
        self.processes = processes
        self.gantt_chart = gantt_chart
        self.statistics = self._calculate()

    def get_value(self, name: str) -> float:
        return self.statistics.get(name, 0.0)
    
    def _calculate(self) -> Dict[str, float]:
        """Calculate scheduling algorithm performance statistics"""

        n = len(self.processes)
        if n == 0:
            return {}

        avg_turnaround_time = sum(p.turnaround_time for p in self.processes) / n
        avg_waiting_time = sum(p.waiting_time for p in self.processes) / n
        avg_response_time = sum(p.response_time for p in self.processes if p.response_time >= 0) / n
        avg_completion_time = sum(p.completion_time for p in self.processes) / n

        return {
            'avg_turnaround_time': avg_turnaround_time,
            'avg_waiting_time': avg_waiting_time,
            'avg_response_time': avg_response_time,
            'avg_completion_time': avg_completion_time,
        }
    
    def show_summary(self, algorithm: str = ""):
        """
        Display statistics summary for given algorithm.
        """
        if algorithm:
            print(f"{algorithm} Summary:")
        print(f"\tprocess count: {len(self.processes)}")
        print(f"\taverage completion time (ACT): {self.statistics['avg_completion_time']:.2f}")
        print(f"\taverage turnaround time (ATAT): {self.statistics['avg_turnaround_time']:.2f}")
        print(f"\taverage waiting time (AWT): {self.statistics['avg_waiting_time']:.2f}")
        print(f"\taverage response time (ART): {self.statistics['avg_response_time']:.2f}")


class ProcessGenerator():
    """Generator of process sequence sets depending on parameters"""

    def __init__(self, seed: Optional[int] = None):
        super().__init__()

        self.input_directory = "cpu_scheduling/input_data"
        os.makedirs(self.input_directory, exist_ok=True)
        
        if seed is not None:
            np.random.seed(seed)
    
    def random_workload(self, n: int, 
                       arrival_range: Tuple[int, int], 
                       burst_range: Tuple[int, int], 
                       priority_range: Tuple[int, int]) -> List[Process]:
        """Generate random processes with typical arrival and burst times"""

        processes = []

        for i in range(n):
            process = Process(
                pid=i + 1,
                arrival_time=np.random.randint(arrival_range[0], arrival_range[1] + 1),
                burst_time=np.random.randint(burst_range[0], burst_range[1] + 1),
                priority=np.random.randint(priority_range[0], priority_range[1] + 1)
            )
            processes.append(process)
        filename = f"random_{n}_ap{arrival_range[0]}-{arrival_range[1]}_bt{burst_range[0]}-{burst_range[1]}"

        self._save_set(processes, filename, {
            'type': 'random',
            'process_count': n,
            'description': 'Random processes with typical arrival and burst times'
        })
        return processes
    
    def sequential_workload(self, n: int, 
                           interval: int,
                           burst_range: Tuple[int, int],
                           burst_pattern: str) -> List[Process]:
        """Generate processes with sequential arrival times"""

        processes = []
        for i in range(n):
            if burst_pattern == 'increasing':
                burst_time = burst_range[0] + (i * (burst_range[1] - burst_range[0]) // n)
            elif burst_pattern == 'decreasing':
                burst_time = burst_range[1] - (i * (burst_range[1] - burst_range[0]) // n)
            else:
                burst_time = np.random.randint(burst_range[0], burst_range[1] + 1)
            process = Process(
                pid=i + 1,
                arrival_time=i * interval,
                burst_time=burst_time,
                priority=np.random.randint(1, 6)
            )
            processes.append(process)
        if burst_pattern == 'increasing':
            burst_pattern = '->'
        elif burst_pattern == 'decreasing':
            burst_pattern = '<-'
        else:
            burst_pattern = '--'
        filename = f"sequential_{n}_i{interval}_bt{burst_range[0]}-{burst_range[1]}_{burst_pattern}"

        self._save_set(processes, filename, {
            'type': 'sequential',
            'process_count': n,
            'description': 'Processes with sequential arrival times'
        })
        return processes

    def mixed_workload(self, n: int,
                        short_ratio: float,
                        short_burst_range: Tuple[int, int],
                        long_burst_range: Tuple[int, int]) -> List[Process]:
        """Generate mixed processes with short and long tasks"""

        processes = []
        n_short = int(n * short_ratio)

        # generate n short tasks
        for i in range(n_short):
            process = Process(
                pid=i + 1,
                arrival_time=np.random.randint(0, max(10, n // 4)),
                burst_time=np.random.randint(short_burst_range[0], short_burst_range[1] + 1),
                priority=np.random.randint(1, 3)
            )
            processes.append(process)

        for i in range(n - n_short):
            process = Process(
                pid=n_short + i + 1,
                arrival_time=np.random.randint(0, max(15, n // 3)),
                burst_time=np.random.randint(long_burst_range[0], long_burst_range[1] + 1),
                priority=np.random.randint(3, 6)
            )
            processes.append(process)

        processes.sort(key=lambda p: (p.arrival_time, p.pid))
        
        filename = f"mixed_{n}_short{int(short_ratio*100)}"

        self._save_set(processes, filename, {
            'type': 'mixed',
            'description': 'Mixed processes with short and long tasks'
        })
        return processes

    def cpu_intensive_workload(self, n: int, 
                       min_burst: int,
                       max_burst: int) -> List[Process]:
        """Generate CPU-intensive workload with long burst times"""

        processes = []
        for i in range(n):
            process = Process(
                pid=i + 1,
                arrival_time=np.random.randint(0, n // 2),
                burst_time=np.random.randint(min_burst, max_burst + 1),
                priority=np.random.randint(1, 5)
            )
            processes.append(process)

        processes.sort(key=lambda p: p.arrival_time)
        filename = f"cpu_intensive_{n}_bt{min_burst}-{max_burst}"

        self._save_set(processes, filename, {
            'type': 'cpu_intensive',
            'description': 'CPU-intensive workload with long burst times'
        })
        return processes
    
    def io_intensive_workload(self, n: int, 
                      max_burst: int,
                      priority_range: Tuple[int, int]) -> List[Process]:
        """Generate I/O-intensive workload with short burst times"""

        processes = []
        for i in range(n):
            process = Process(
                pid=i + 1,
                arrival_time=np.random.randint(0, n),
                burst_time=np.random.randint(1, max_burst + 1),
                priority=np.random.randint(priority_range[0], priority_range[1] + 1)
            )
            processes.append(process)

        processes.sort(key=lambda p: p.arrival_time)
        filename = f"io_intensive_{n}_bt1-{max_burst}"

        self._save_set(processes, filename, {
            'type': 'io_intensive',
            'description': 'I/O-intensive workload with short burst times'
        })
        return processes
    

    def _save_set(self, processes: List[Process], filename: str, metadata: dict):
        """Save dataset with metadata"""

        dataset_dir = os.path.join(self.input_directory, filename)
        os.makedirs(dataset_dir, exist_ok=True)
        
        data = [{
            'PID': p.pid,
            'Arrival_Time': p.arrival_time,
            'Burst_Time': p.burst_time,
            'Priority': p.priority
        } for p in processes]
        df = pd.DataFrame(data)

        csv_path = os.path.join(dataset_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)

        metadata = {
            **metadata,
            'dataset_name': filename,
            'process_count': len(processes),
            'arrival_time_range': f"{min(p.arrival_time for p in processes)}-{max(p.arrival_time for p in processes)}",
            'burst_time_range': f"{min(p.burst_time for p in processes)}-{max(p.burst_time for p in processes)}",
            'priority_range': f"{min(p.priority for p in processes)}-{max(p.priority for p in processes)}",
            'avg_burst_time': sum(p.burst_time for p in processes) / len(processes),
            'avg_arrival_time': sum(p.arrival_time for p in processes) / len(processes),
            'avg_priority': sum(p.priority for p in processes) / len(processes),
            'timestamp': pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S"),
        }

        metadata_path = os.path.join(dataset_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"\t{filename}")


    def generate_datasets(self):
        """
        Generate test datasets
        """

        print(">>>>> generating datasets <<<<<\n")

        # basic set - standard comparison of algorithm performance on random processes
        self.random_workload(
            n=100,
            arrival_range=(0, 50),
            burst_range=(10, 50),
            priority_range=(1, 5)
        )
        # sequential set - analysis of behavior with sequential process arrival
        self.sequential_workload(
            n=100,
            interval=2,
            burst_range=(1, 50),
            burst_pattern='increasing'  # 'decreasing' or 'random'
        )
        # realistic set - simulation of real system workload patterns
        self.mixed_workload(
            n=100,
            short_ratio=0.7,
            short_burst_range=(1, 5),
            long_burst_range=(10, 20)
        )
        # CPU-intensive set - heavy tasks
        self.cpu_intensive_workload(
            n=100,
            min_burst=25,
            max_burst=100
        )
        # IO-intensive set - frequent I/O operations
        self.io_intensive_workload(
            n=100,
            max_burst=10,
            priority_range=(1, 5)
        )

        self._generate_summary()

    def _generate_summary(self):
        """Generate dataset summary"""

        summary_data = []
        
        for item in os.listdir(self.input_directory):
            item_path = os.path.join(self.input_directory, item)
            if os.path.isdir(item_path):
                metadata_file = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    summary_data.append(metadata)

        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            summary_file = os.path.join(self.input_directory, 'datasets_summary.csv')
            df_summary.to_csv(summary_file, index=False)
            print(f"{summary_file}")


if __name__ == "__main__":
    generator = ProcessGenerator(seed=42)
    title = "CPU Process Sets Generation"
    print(f'\n{"=" * len(title)}')
    print(title)
    print(f'{"=" * len(title)}\n')
    
    generator.generate_datasets()
    print(f"\nFiles saved in directory: {generator.input_directory}")