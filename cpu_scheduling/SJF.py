"""
CPU Scheduling Algorithm SJF (Shortest Job First) non-preemptive = SJF
"""

from typing import List, Tuple
from cpu_generate import Process, GanttEntry, SchedulingStatistics, ProcessGenerator


class SJFNonPreemptive(Process):
    """Implementation of non-preemptive SJF"""
    
    def __init__(self):
        super().__init__()
        self.gantt_chart = []
        self.completed_processes = []
        self.current_time = 0
    
    def simulate(self, processes: List[Process]) -> Tuple[List[Process], SchedulingStatistics]:
        """Execute SJF scheduling"""

        self._reset()
        
        working_processes = [self._copy_process(p) for p in processes]
        working_processes.sort(key=lambda x: (x.arrival_time, x.pid))
        
        ready_queue = []
        process_index = 0
        
        while process_index < len(working_processes) or ready_queue:
            while (process_index < len(working_processes) and
                   working_processes[process_index].arrival_time <= self.current_time):
                ready_queue.append(working_processes[process_index])
                process_index += 1
            
            if not ready_queue:
                if process_index < len(working_processes):
                    self.current_time = working_processes[process_index].arrival_time
            else:
                selected = min(ready_queue, key=lambda x: (x.burst_time, x.arrival_time, x.pid))
                ready_queue.remove(selected)
                self._execute_process(selected)
        
        stats = SchedulingStatistics(self.completed_processes, self.gantt_chart)
        return self.completed_processes.copy(), stats
    
    def _reset(self):
        """Reset algorithm state"""

        self.gantt_chart.clear()
        self.completed_processes.clear()
        self.current_time = 0
    
    def _copy_process(self, process: Process) -> Process:
        """Copy process to local instance"""

        return Process(
            pid=process.pid,
            arrival_time=process.arrival_time,
            burst_time=process.burst_time,
            priority=process.priority
        )
    
    def _execute_process(self, process: Process):
        """Execute process without preemption"""

        process.set_start_time(self.current_time)
        
        self.gantt_chart.append(GanttEntry(
            process_id=process.pid,
            start_time=self.current_time,
            duration=process.burst_time
        ))

        self.current_time += process.burst_time
        process.completion_time = self.current_time
        process.calculate_times()
        self.completed_processes.append(process)


if __name__ == "__main__":
    title = "Testing Non-Preemptive SJF Algorithm"
    print(f'\n{"=" * len(title)}')
    print(title)
    print(f'{"=" * len(title)}\n')

    generator = ProcessGenerator(seed=42)
    cpu_processes = generator.random_workload(8, (0, 16), (2, 24), (1, 5))

    print("\nTest processes:")
    for p in cpu_processes:
        print(f"\tP{p.pid}: AT={p.arrival_time}, BT={p.burst_time}")
    
    algorithm = SJFNonPreemptive()
    completed, statistics = algorithm.simulate(cpu_processes)
    
    print("\nTime results:")
    for p in sorted(completed, key=lambda x: x.pid):
        print(f"\tP{p.pid}: CT={p.completion_time}, TAT={p.turnaround_time}, WT={p.waiting_time}, RT={p.response_time}")
    
    print("\nGantt Chart:")
    for entry in algorithm.gantt_chart:
        print(f"\tP{entry.process_id}: [{entry.start_time}-{entry.end_time}]")
    print()
    statistics.show_summary("SJF")