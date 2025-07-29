"""
Algorytm wymiany stron LRU (Least Recently Used)
"""

from collections import OrderedDict
from typing import List, Tuple
from strony_generator import GeneratorStron, Strony


class LRU(Strony):
    """
    Implementacja LRU.
    """
    
    def __init__(self, liczba_ramek: int):
        self.liczba_ramek = liczba_ramek
        self.pamiec = OrderedDict()  # strony w kolejności użycia
        self.bledy_stron = 0
        self.trafienia_stron = 0
        
    def dostep_do_strony(self, numer_strony: int) -> bool:
        """
        Dostęp do strony
        """

        if numer_strony in self.pamiec:
            # jeśli trafi, to przenosi na koniec (najnowsze)
            self.pamiec.move_to_end(numer_strony)
            self.trafienia_stron += 1
            return False
        else:
            self.bledy_stron += 1
            # jeśli pamięć pełna, usuwa najstarszą stronę (LRU)
            if len(self.pamiec) >= self.liczba_ramek:
                self.pamiec.popitem(last=False)
            # dodaje nową stronę na koniec
            self.pamiec[numer_strony] = True
            return True
    
    def symuluj(self, sekwencja_stron: List[int]) -> Tuple[int, int]:
        """
        Symuluj algorytm LRU.
        """

        for strona in sekwencja_stron:
            self.dostep_do_strony(strona)
        return self.bledy_stron, self.trafienia_stron
    
    def pobierz_statystyki(self) -> dict:
        """Zwróć podstawowe statystyki."""

        laczne = self.trafienia_stron + self.bledy_stron
        return {
            'algorytm': 'LRU',
            'ramki': self.liczba_ramek,
            'dostępy': laczne,
            'trafienia': self.trafienia_stron,
            'błędy': self.bledy_stron,
            'procent_trafień': (self.trafienia_stron / laczne * 100) if laczne > 0 else 0
        }
    
    def resetuj(self):
        """Resetuj stan algorytmu."""

        self.pamiec.clear()
        self.bledy_stron = 0
        self.trafienia_stron = 0
        

if __name__ == "__main__":
    tytul = "Testowanie algorytmu LRU"
    print(f'\n{"=" * len(tytul)}')
    print(tytul)
    print(f'{"=" * len(tytul)}\n')

    # tworzy przykładowe dane testowe
    generator = GeneratorStron(ziarno=42)
    sekwencja_stron = generator.obciazenie_losowe(dlugosc=50, zakres_stron=(0, 20))
    liczba_ramek = 16

    print(f"\nSekwencja stron:\n\t{' '.join(map(str, sekwencja_stron))}")
    print(f"\nLiczba ramek:\n\t{liczba_ramek}\n")

    # uruchamia algorytm LRU
    lru = LRU(liczba_ramek)
    bledy, trafienia = lru.symuluj(sekwencja_stron)

    print("Podsumowanie wyników LRU:")
    print(f"\tbłędy stron: {bledy}")
    print(f"\ttrafienia stron: {trafienia}")
    print(f"\tprocent trafień: {lru.pobierz_statystyki()['procent_trafień']:.2f}%")
