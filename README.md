# CPU & Page Algorithms

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9.2-red.svg)](https://pypi.org/project/matplotlib/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-yellow.svg)](https://pypi.org/project/pandas/)
[![Numpy](https://img.shields.io/badge/Numpy-1.26.4-green.svg)](https://pypi.org/project/numpy/)
[![Status](https://img.shields.io/badge/Status-Beta-orange.svg)]()

```
 ___  ___  _ _   _    ___                  ___  _                 _    _    _              
|  _>| . \| | | < >  | . \ ___  ___  ___  | . || | ___  ___  _ _ <_> _| |_ | |_ ._ _ _  ___
| <__|  _/| ' | /.\/ |  _/<_> |/ . |/ ._> |   || |/ . |/ . \| '_>| |  | |  | . || ' ' |<_-<
`___/|_|  `___' \_/\ |_|  <___|\_. |\___. |_|_||_|\_. |\___/|_|  |_|  |_|  |_|_||_|_|_|/__/
                               <___'               <___'                                   
```

A Python tool for analyzing CPU scheduling and page replacement algorithms. It generates test data automatically and creates nice visual reports.

## What's Inside

**CPU Scheduling Algorithms:**
- SJF (Shortest Job First) - both preemptive and non-preemptive versions
- Round Robin with customizable time quantum
- Priority-based scheduling
- Auto-generated workloads for testing

**Page Replacement Algorithms:**
- LRU (Least Recently Used)
- LFU (Least Frequently Used)
- Configurable number of frames

**Data Generation:**
- Creates realistic process and page reference patterns
- Exports to CSV and JSON
- Can generate multiple test sets at once

**Performance Analysis:**
- Tracks turnaround time, waiting time, completion time, and response time
- Calculates page hit/miss rates
- Compares different algorithms side by side

**Visualization:**
- Gantt charts for CPU scheduling
- Performance graphs and histograms
- Saves everything as PNG, CSV, and TXT files

## Requirements

- Python 3.13
- A few packages:
  ```bash
  matplotlib>=3.9.2
  pandas>=2.2.2
  numpy>=1.26.4
  ```

## Getting Started

1. **Install the packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simulations:**
   
   For CPU scheduling:
   ```bash
   python cpu_scheduling/cpu_RUN.py
   ```
   
   For page replacement:
   ```bash
   python page_replacement/page_RUN.py
   ```

## Project Structure

```
README.md
cpu_scheduling/
    cpu_generate.py
    cpu_RUN.py
    cpu_simulate.py
    SJF.py
    SRTF.py
    RR.py
    input_data/
    output_data/
page_replacement/
    page_generate.py
    page_RUN.py
    page_simulate.py
    LRU.py
    LFU.py
    input_data/
    output_data/
```

## License

MIT License - feel free to use it however you want.
