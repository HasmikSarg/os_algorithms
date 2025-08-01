"""
LFU (Least Frequently Used) page replacement algorithm
"""

from collections import defaultdict
from typing import List, Tuple
from page_generate import PageGenerator, Pages


class LFU(Pages):
    """
    LFU algorithm implementation.
    """
    
    def __init__(self, frame_count: int):
        self.frame_count = frame_count
        self.memory = set()
        self.frequencies = defaultdict(int)
        self.page_faults = 0
        self.page_hits = 0
        
    def access_page(self, page_number: int) -> bool:
        """
        Access a page.
        """

        if page_number in self.memory:
            self.page_hits += 1
            self.frequencies[page_number] += 1
            return False
        else:
            self.page_faults += 1
            if len(self.memory) >= self.frame_count:
                lfu_page = min(self.memory, key=lambda p: self.frequencies[p])
                self.memory.remove(lfu_page)
            self.memory.add(page_number)
            self.frequencies[page_number] += 1
            return True
    
    def simulate(self, page_sequence: List[int]) -> Tuple[int, int]:
        """
        Simulate the LFU algorithm.
        """

        for page in page_sequence:
            self.access_page(page)
        return self.page_faults, self.page_hits
    
    def get_stats(self) -> dict:
        """Return basic statistics."""

        total = self.page_hits + self.page_faults
        return {
            'algorithm': 'LFU',
            'frames': self.frame_count,
            'accesses': total,
            'hits': self.page_hits,
            'faults': self.page_faults,
            'hit_percentage': (self.page_hits / total * 100) if total > 0 else 0
        }
    
    def reset(self):
        """Reset algorithm state."""

        self.memory.clear()
        self.frequencies.clear()
        self.page_faults = 0
        self.page_hits = 0

if __name__ == "__main__":
    title = "Testing LFU algorithm"
    print(f'\n{"=" * len(title)}')
    print(title)
    print(f'{"=" * len(title)}\n')

    generator = PageGenerator(seed=42)
    page_sequence = generator.random_load(length=50, page_range=(0, 20))
    frame_count = 16

    print(f"\nPage sequence:\n\t{' '.join(map(str, page_sequence))}")
    print(f"\nFrame count:\n\t{frame_count}\n")

    lfu = LFU(frame_count)
    faults, hits = lfu.simulate(page_sequence)

    print("LFU results summary:")
    print(f"\tpage faults: {faults}")
    print(f"\tpage hits: {hits}")
    print(f"\thit percentage: {lfu.get_stats()['hit_percentage']:.2f}%")