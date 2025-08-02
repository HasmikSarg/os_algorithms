# AlgoOS

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9.2-red.svg)](https://pypi.org/project/matplotlib/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-yellow.svg)](https://pypi.org/project/pandas/)
[![Pandas](https://img.shields.io/badge/Numpy-1.26.4-green.svg)](https://pypi.org/project/pandas/)
[![Status](https://img.shields.io/badge/Status-Beta-orange.svg)]()

```

 ___  ___  _ _   _    ___                  ___  _                 _    _    _              
|  _>| . \| | | < >  | . \ ___  ___  ___  | . || | ___  ___  _ _ <_> _| |_ | |_ ._ _ _  ___
| <__|  _/| ' | /.\/ |  _/<_> |/ . |/ ._> |   || |/ . |/ . \| '_>| |  | |  | . || ' ' |<_-<
`___/|_|  `___' \_/\ |_|  <___|\_. |\___. |_|_||_|\_. |\___/|_|  |_|  |_|  |_|_||_|_|_|/__/
                               <___'               <___'                                   

```

A simple program for analyzing and visualizing CPU scheduling and page replacement algorithms, with automated data generation and reporting.

## Features

- **CPU Scheduling Algorithms**:
  - SJF (Shortest Job First, preemptive & non-preemptive)
  - Round Robin (configurable quantum)
  - Priority-based scheduling
  - Automated workload generation

- **Page Replacement Algorithms** (configurable frames):
  - LRU (Least Recently Used)
  - LFU (Least Frequently Used)
  - Automated sequence generation

- **Data Generation**:
  - Custom distributions for processes and page references
  - CSV and JSON export for input datasets
  - Batch generation for multiple test sets

- **Performance Analysis**:
  - Average turnaround, waiting, completion and response times
  - Page hit/miss rates and execution time
  - Comparative analysis across algorithms and datasets

- **Visualization**:
  - Gantt charts for CPU scheduling
  - Performance graphs and histograms
  - Automated report and chart export (PNG, CSV, TXT)

## Prerequisites

- Python 3.13
- Required packages:
  ```bash
  matplotlib>=3.9.2
  pandas>=2.2.2
  numpy>=1.26.4
  ```

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run simulations:**
   - **CPU Scheduling:**
     ```bash
     python cpu_scheduling/cpu_RUN.py
     ```
   - **Page Replacement:**
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

This project is released under the MIT License.
