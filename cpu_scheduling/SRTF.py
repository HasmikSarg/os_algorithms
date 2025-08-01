"""
CPU Scheduling Algorithm SJF (Shortest Job First) preemptive = SRTF (Shortest Remaining Time First)
"""

from typing import List, Tuple
from cpu_generate import Process, GanttEntry, SchedulingStatistics, ProcessGenerator


class SJFPreemptive(Process):
    """Implementation of preemptive SJF = SRTF (Shortest Remaining Time First)"""
    
    def __init__(self):
        super().__init__()
        self.gantt_chart = []
        self.completed_processes = []
        self.current_time = 0
        self.preemption_count = 0
    
    def simulate(self, processes: List[Process]) -> Tuple[List[Process], SchedulingStatistics]:
        """Execute SRTF scheduling"""

        self._reset()
        
        working_processes = [self._copy_process(p) for p in processes]
        working_processes.sort(key=lambda x: (x.arrival_time, x.pid))
        
        ready_queue = []
        process_index = 0
        current_process = None
        execution_start_time = 0
        
        while process_index < len(working_processes) or ready_queue or current_process:
            while (process_index < len(working_processes) and 
                   working_processes[process_index].arrival_time <= self.current_time):
                ready_queue.append(working_processes[process_index])
                process_index += 1
            
            if current_process and ready_queue:
                shortest = min(ready_queue, key=lambda x: (x.remaining_time, x.arrival_time, x.pid))
                if shortest.remaining_time < current_process.remaining_time:
                    self._record_execution(current_process, execution_start_time)
                    ready_queue.append(current_process)
                    current_process = None
                    self.preemption_count += 1

            if not current_process:
                if ready_queue:
                    current_process = min(ready_queue, key=lambda x: (x.remaining_time, x.arrival_time, x.pid))
                    ready_queue.remove(current_process)
                    current_process.set_start_time(self.current_time)
                    execution_start_time = self.current_time
                elif process_index < len(working_processes):
                    self.current_time = working_processes[process_index].arrival_time
                    continue
                else:
                    break
            
            if current_process:
                self.current_time += 1
                current_process.remaining_time -= 1
                if current_process.remaining_time == 0:
                    self._record_execution(current_process, execution_start_time)
                    current_process.completion_time = self.current_time
                    current_process.calculate_times()
                    self.completed_processes.append(current_process)
                    current_process = None
        
        stats = SchedulingStatistics(self.completed_processes, self.gantt_chart)
        return self.completed_processes.copy(), stats
    
    def _reset(self):
        """Reset algorithm state"""

        self.gantt_chart.clear()
        self.completed_processes.clear()
        self.current_time = 0
        self.preemption_count = 0
    
    def _copy_process(self, process: Process) -> Process:
        """Copy process to local instance"""

        return Process(
            pid=process.pid,
            arrival_time=process.arrival_time,
            burst_time=process.burst_time,
            priority=process.priority
        )
    
    def _record_execution(self, process: Process, start_time: int):
        """Record process execution"""

        duration = self.current_time - start_time

        if duration > 0:
            self.gantt_chart.append(GanttEntry(
                process_id=process.pid,
                start_time=start_time,
                duration=duration
            ))


if __name__ == "__main__":
    from cpu_generate import ProcessGenerator

    title = "Testing SRTF Algorithm (Preemptive SJF)"
    print(f'\n{"=" * len(title)}')
    print(title)
    print(f'{"=" * len(title)}\n')

    generator = ProcessGenerator(seed=42)
    cpu_processes = generator.random_workload(8, (0, 16), (2, 24), (1, 5))

    print("\nTest processes:")
    for p in cpu_processes:
        print(f"\tP{p.pid}: AT={p.arrival_time}, BT={p.burst_time}")

    algorithm = SJFPreemptive()
    completed, statistics = algorithm.simulate(cpu_processes)

    print(f"\nTime results:")
    for p in sorted(completed, key=lambda x: x.pid):
        print(f"\tP{p.pid}: CT={p.completion_time}, TAT={p.turnaround_time}, WT={p.waiting_time}, RT={p.response_time}")
    print(f"\tpreemptions: {algorithm.preemption_count}")
    
    print("\nGantt Chart:")
    for entry in algorithm.gantt_chart:
        print(f"\tP{entry.process_id}: [{entry.start_time}-{entry.end_time}]")
    print()
    statistics.show_summary("SRTF")