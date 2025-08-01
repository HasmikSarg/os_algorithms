"""
CPU Scheduling Algorithm RR (Round-Robin)
"""

from typing import List, Tuple
from collections import deque
from cpu_generate import Process, GanttEntry, SchedulingStatistics, ProcessGenerator


class RoundRobin(Process):
    """Implementation of Round Robin algorithm"""
    
    def __init__(self, time_quantum: int = 8):
        super().__init__()
        self.time_quantum = time_quantum
        self.ready_queue = deque()
        self.gantt_chart = []
        self.completed_processes = []
        self.current_time = 0
    
    def simulate(self, processes: List[Process]) -> Tuple[List[Process], SchedulingStatistics]:
        """Execute RR scheduling"""

        self._reset()
        
        working_processes = [self._copy_process(p) for p in processes]
        working_processes.sort(key=lambda x: (x.arrival_time, x.pid))
        
        process_index = 0
        
        while process_index < len(working_processes) or self.ready_queue:
            while (process_index < len(working_processes) and 
                   working_processes[process_index].arrival_time <= self.current_time):
                self.ready_queue.append(working_processes[process_index])
                process_index += 1
            
            if not self.ready_queue:
                if process_index < len(working_processes):
                    self.current_time = working_processes[process_index].arrival_time
                else:
                    break
            else:
                current_process = self.ready_queue.popleft()
                self._execute_process(current_process, working_processes, process_index)

        stats = SchedulingStatistics(self.completed_processes, self.gantt_chart)
        return self.completed_processes.copy(), stats
    
    def _reset(self):
        """Reset algorithm state"""

        self.ready_queue.clear()
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
    
    def _execute_process(self, process: Process, all_processes: List[Process], process_index: int):
        """Execute process for quantum time or until completion"""

        process.set_start_time(self.current_time)
        execution_time = min(self.time_quantum, process.remaining_time)
        
        self.gantt_chart.append(GanttEntry(
            process_id=process.pid,
            start_time=self.current_time,
            duration=execution_time
        ))
        
        process.remaining_time -= execution_time
        self.current_time += execution_time
        
        new_index = process_index
        while (process_index < len(all_processes) and 
               all_processes[process_index].arrival_time <= self.current_time):
            process_index += 1

        for i in range(new_index, process_index):
            self.ready_queue.append(all_processes[i])
        
        if process.remaining_time == 0:
            process.completion_time = self.current_time
            process.calculate_times()
            self.completed_processes.append(process)
        else:
            self.ready_queue.append(process)


if __name__ == "__main__":
    title = "Testing RR Algorithm"
    print(f'\n{"=" * len(title)}')
    print(title)
    print(f'{"=" * len(title)}\n')
    
    generator = ProcessGenerator(seed=42)
    cpu_processes = generator.random_workload(8, (0, 16), (2, 24), (1, 5))

    print("\nTest processes:")
    for p in cpu_processes:
        print(f"\tP{p.pid}: AT={p.arrival_time}, BT={p.burst_time}")
    
    for q in [2, 4]:
        algorithm = RoundRobin(time_quantum=q)
        completed, statistics = algorithm.simulate(cpu_processes)

        print(f"\nTime results (q={q}):")
        for p in sorted(completed, key=lambda x: x.pid):
            print(f"\tP{p.pid}: CT={p.completion_time}, TAT={p.turnaround_time}, WT={p.waiting_time}, RT={p.response_time}")
        
        statistics.show_summary(f"RR (q={q})")