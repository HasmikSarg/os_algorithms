# SimulationPrograms

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9.2-red.svg)](https://pypi.org/project/matplotlib/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-yellow.svg)](https://pypi.org/project/pandas/)
[![Status](https://img.shields.io/badge/Status-Beta-orange.svg)]()

```

 ___  ___    ___       _           _       _  _             _     ___            _                                   _      ___  _                 _    _    _              
| . |/ __>  / __> ___ | |_  ___  _| | _ _ | |<_>._ _  ___  < >   | . \ ___  ___ | | ___  ___  ___ ._ _ _  ___ ._ _ _| |_   | . || | ___  ___  _ _ <_> _| |_ | |_ ._ _ _  ___
| | |\__ \  \__ \/ | '| . |/ ._>/ . || | || || || ' |/ . | /.\/  |   // ._>| . \| |<_> |/ | '/ ._>| ' ' |/ ._>| ' | | |    |   || |/ . |/ . \| '_>| |  | |  | . || ' ' |<_-<
`___'<___/  <___/\_|_.|_|_|\___.\___|`___||_||_||_|_|\_. | \_/\  |_\_\\___.|  _/|_|<___|\_|_.\___.|_|_|_|\___.|_|_| |_|    |_|_||_|\_. |\___/|_|  |_|  |_|  |_|_||_|_|_|/__/
                                                     <___'                 |_|                                                     <___'                                    

```

A comprehensive suite for analyzing and visualizing CPU scheduling and page replacement algorithms, with automated data generation and reporting.

## Features

- **CPU Scheduling Algorithms**:
  - SJF (Shortest Job First, preemptive & non-preemptive)
  - Round Robin (configurable quantum)
  - Priority-based scheduling
  - Custom workload generation

- **Page Replacement Algorithms**:
  - LRU (Least Recently Used)
  - LFU (Least Frequently Used)
  - Automated sequence generation

- **Data Generation**:
  - Random, sequential, mixed, and custom distributions for processes and page references
  - CSV and JSON export for input datasets
  - Batch generation for multiple test sets

- **Performance Analysis**:
  - Average waiting, turnaround, response, and completion times
  - Page hit/miss rates, error statistics
  - Comparative analysis across algorithms and datasets

- **Visualization**:
  - Gantt charts for CPU scheduling
  - Performance graphs and histograms
  - Page replacement efficiency plots
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
     python szeregowanie_cpu/cpu_PROGRAM.py
     ```
   - **Page Replacement:**
     ```bash
     python zastepowanie_stron/strony_PROGRAM.py
     ```

4. **View results:**
   - Output files are saved in `szeregowanie_cpu/dane_wyjsciowe/` and `zastepowanie_stron/dane_wyjsciowe/`
   - Reports, CSV summaries, and PNG charts are auto-generated


## Screenshots

![CPU Scheduling Gantt Chart](https://github.com/user-attachments/assets/b6293cc8-86d5-44fb-b69c-bee71dd9b673)
![Performance Analysis](https://github.com/user-attachments/assets/ae1f0b9d-70df-4373-b79a-29b827c7ea34)
![Data Sets](https://github.com/user-attachments/assets/ac8ed108-3784-45c4-aaff-3e7f0e430402)
![Distribution View](https://github.com/user-attachments/assets/7104f9d1-6ebb-4bde-9cb4-c1438117932b)

## Project Structure

```
README.md
szeregowanie_cpu/
    cpu_generator.py
    cpu_PROGRAM.py
    cpu_zasady.py
    SJF_niewywl.py
    SJF_wywl.py
    Round_Robin.py
    dane_wejsciowe/
    dane_wyjsciowe/
zastepowanie_stron/
    strony_generator.py
    strony_PROGRAM.py
    strony_zasady.py
    LRU.py
    LFU.py
    dane_wejsciowe/
    dane_wyjsciowe/
```

## License

This project is released under the MIT License.
