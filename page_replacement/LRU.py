"""
LRU (Least Recently Used) page replacement algorithm
"""

from collections import OrderedDict
from typing import List, Tuple
from page_generate import PageGenerator, Pages


class LRU(Pages):
    """
    LRU algorithm implementation.
    """
    
    def __init__(self, frame_count: int):
        self.frame_count = frame_count
        self.memory = OrderedDict()
        self.page_faults = 0
        self.page_hits = 0
        
    def access_page(self, page_number: int) -> bool:
        """
        Access a page - updates LRU ordering
        """
        if page_number in self.memory:
            self.memory.move_to_end(page_number)
            self.page_hits += 1
            return False
        else:
            self.page_faults += 1
            if len(self.memory) >= self.frame_count:
                self.memory.popitem(last=False)
            self.memory[page_number] = True
            return True
    
    def simulate(self, page_sequence: List[int]) -> Tuple[int, int]:
        """
        Simulate the LRU algorithm on a page reference sequence
        """
        for page in page_sequence:
            self.access_page(page)
        return self.page_faults, self.page_hits
    
    def get_stats(self) -> dict:
        """Return performance statistics"""
        total_accesses = self.page_hits + self.page_faults
        return {
            'algorithm': 'LRU',
            'frames': self.frame_count,
            'accesses': total_accesses,
            'hits': self.page_hits,
            'faults': self.page_faults,
            'hit_percentage': (self.page_hits / total_accesses * 100) if total_accesses > 0 else 0
        }
    
    def reset(self):
        """Reset the algorithm's state"""
        self.memory.clear()
        self.page_faults = 0
        self.page_hits = 0
        

if __name__ == "__main__":
    title = "Testing LRU Algorithm"
    print(f'\n{"=" * len(title)}')
    print(title)
    print(f'{"=" * len(title)}\n')

    generator = PageGenerator(seed=42)
    page_sequence = generator.random_load(length=50, page_range=(0, 20))
    frame_count = 16

    print(f"\nPage reference sequence:\n\t{' '.join(map(str, page_sequence))}")
    print(f"\nFrame count:\n\t{frame_count}\n")

    lru = LRU(frame_count)
    faults, hits = lru.simulate(page_sequence)

    print("LRU Performance Results:")
    print(f"\tPage faults: {faults}")
    print(f"\tPage hits: {hits}")
    print(f"\tHit percentage: {lru.get_stats()['hit_percentage']:.2f}%")