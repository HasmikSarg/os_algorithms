"""
Page reference sequence generator
    generates page sets for testing page replacement algorithms
    saves results in input_data/ in separate directories as .csv and .json
"""

import numpy as np
import pandas as pd
import os
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Pages:
    """Configure page sequence generation"""

    length: int
    page_range: Tuple[int, int]
    pattern_type: str = "random"
    seed: Optional[int] = None
    locality_factor: float = 0.8
    locality_size: int = 5
    working_set_size: int = 10


class PageGenerator:
    """Generate page reference sequences with different characteristics"""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        self.input_directory = "page_replacement/input_data"
        os.makedirs(self.input_directory, exist_ok=True)

    def _save_csv(self, sequence: List[int], filename: str):
        """Save sequence to CSV in a separate folder for each dataset"""

        dataset_name = filename.replace('.csv', '')

        dataset_directory = os.path.join(self.input_directory, dataset_name)
        os.makedirs(dataset_directory, exist_ok=True)
        df = pd.DataFrame({
            'step': range(1, len(sequence) + 1),
            'page_number': sequence
        })

        csv_path = os.path.join(dataset_directory, f"{dataset_name}.csv")
        df.to_csv(csv_path, index=False)

        metadata = {
            'pattern_type': dataset_name.split('_')[0],
            'dataset_name': dataset_name,
            'sequence_length': len(sequence),
            'page_range': f"{min(sequence)}-{max(sequence)}",
            'unique_pages': len(set(sequence)),
            'generation_timestamp': pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S"),
        }
        metadata_path = os.path.join(dataset_directory, 'metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\t{dataset_name}")

    def random_load(self, length: int, page_range: Tuple[int, int]) -> List[int]:
        """Generate random page reference sequence"""

        config = Pages(
            length=length,
            page_range=page_range,
            pattern_type="random",
            seed=self.seed,
            locality_factor=0.5,
            locality_size=5,
            working_set_size=8
        )
        min_page, max_page = config.page_range
        sequence = np.random.randint(min_page, max_page + 1, size=config.length).tolist()
        filename = f"random_{config.length}_cp{max_page}-{min_page}_cw{config.locality_size}-{config.working_set_size}"

        self._save_csv(sequence, filename)
        return sequence

    def sequential_load(self, length: int, page_range: Tuple[int, int]) -> List[int]:
        """Generate sequential page reference sequence"""

        config = Pages(
            length=length,
            page_range=page_range,
            pattern_type="sequential",
            seed=self.seed,
            locality_factor=0.6,
            locality_size=4,
            working_set_size=6
        )
        min_page, max_page = config.page_range
        sequence = []
        current_page = min_page

        for _ in range(config.length):
            sequence.append(current_page)
            current_page += 1
            if current_page > max_page:
                current_page = min_page
        filename = f"sequential_{config.length}_cp{max_page}-{min_page}_cw{config.locality_size}-{config.working_set_size}"

        self._save_csv(sequence, filename)
        return sequence

    def locality_load(self, length: int, page_range: Tuple[int, int], locality_factor: float, locality_size: int, working_set_size: int) -> List[int]:
        """Generate sequence with temporal and spatial locality"""
        config = Pages(
            length=length,
            page_range=page_range,
            pattern_type="locality",
            seed=self.seed,
            locality_factor=locality_factor,
            locality_size=locality_size,
            working_set_size=working_set_size
        )
        min_page, max_page = config.page_range
        sequence = []

        current_locality = set(np.random.choice(
            range(min_page, max_page + 1),
            size=min(config.locality_size, max_page - min_page + 1),
            replace=False
        ))
        for _ in range(config.length):
            if np.random.random() < config.locality_factor and current_locality:
                page = np.random.choice(list(current_locality))
            else:
                page = np.random.randint(min_page, max_page + 1)
                if np.random.random() < 0.3:
                    if len(current_locality) >= config.locality_size:
                        current_locality.remove(np.random.choice(list(current_locality)))
                    current_locality.add(page)
            sequence.append(page)
        filename = f"locality_{config.length}_cp{max_page}-{min_page}_cw{config.locality_size}-{config.working_set_size}"

        self._save_csv(sequence, filename)
        return sequence

    def working_set_load(self, length: int, page_range: Tuple[int, int], locality_factor: float, locality_size: int, working_set_size: int) -> List[int]:
        """Generate sequence following the working set principle"""
        config = Pages(
            length=length,
            page_range=page_range,
            pattern_type="working_set",
            seed=self.seed,
            locality_factor=locality_factor,
            locality_size=locality_size,
            working_set_size=working_set_size
        )
        min_page, max_page = config.page_range
        sequence = []

        phase_length = config.length // 4
        working_sets = []

        for _ in range(4):
            working_set = set(np.random.choice(
                range(min_page, max_page + 1),
                size=min(config.working_set_size, max_page - min_page + 1),
                replace=False
            ))
            working_sets.append(working_set)
        current_phase = 0
        for i in range(config.length):
            if i > 0 and i % phase_length == 0:
                current_phase = min(current_phase + 1, len(working_sets) - 1)
            if np.random.random() < 0.9 and working_sets[current_phase]:
                page = np.random.choice(list(working_sets[current_phase]))
            else:
                page = np.random.randint(min_page, max_page + 1)
            sequence.append(page)
        filename = f"working_set_{config.length}_cp{max_page}-{min_page}_cw{config.locality_size}-{config.working_set_size}"

        self._save_csv(sequence, filename)
        return sequence

    def generate_datasets(self):
        """
        Generate test datasets
        """

        print(">>>>> generating datasets <<<<<\n")

        # basic dataset - standard comparison of algorithm performance on random sequences
        self.random_load(length=500, page_range=(0, 20))
        # locality dataset - analysis of adaptation to temporal and spatial locality changes
        self.locality_load(length=750, page_range=(0, 20), locality_factor=0.8, locality_size=6, working_set_size=8)
        # working set dataset - simulation of real patterns testing adaptation to local changes and resistance to thrashing
        self.working_set_load(length=1000, page_range=(0, 20), locality_factor=0.7, locality_size=5, working_set_size=10)
        # sequential dataset - stack behavior analysis testing inclusion property and resistance to Belady's anomaly
        self.sequential_load(length=500, page_range=(0, 20))

        self._generate_summary()

    def load_csv(self, filename: str) -> List[int]:
        """Load sequence from CSV"""
        path = os.path.join(self.input_directory, filename)
        df = pd.read_csv(path)
        return df['page_number'].tolist()
    
    def _generate_summary(self):
        """Generate dataset summary based on metadata"""

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
            summary_file = os.path.join(self.input_directory, 'dataset_summary.csv')
            df_summary.to_csv(summary_file, index=False)
            print(f"{summary_file}")

if __name__ == "__main__":
    generator = PageGenerator(seed=42)
    title = "Generating page reference sequences"
    print(f'\n{"=" * len(title)}')
    print(title)
    print(f'{"=" * len(title)}\n')

    generator.generate_datasets()
    print(f"\nFiles saved in directory: {generator.input_directory}")