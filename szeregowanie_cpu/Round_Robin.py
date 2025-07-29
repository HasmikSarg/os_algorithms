"""
Algorytm planowania czasu procesora RR (Round-Robin)
"""

from typing import List, Tuple
from collections import deque
from cpu_generator import Proces, WpisGantta, StatystykiSzeregowania, GeneratorProcesow


class RoundRobin(Proces):
    """Implementacja algorytmu Round Robin"""
    
    def __init__(self, kwant_czasu: int = 8):
        super().__init__()
        self.kwant_czasu = kwant_czasu
        self.kolejka_gotowych = deque()
        self.wykres_gantta = []
        self.procesy_ukonczone = []
        self.aktualny_czas = 0
    
    def symuluj(self, procesy: List[Proces]) -> Tuple[List[Proces], StatystykiSzeregowania]:
        """Wykonaj szeregowanie RR"""

        self._resetuj()
        
        # kopiuje procesy do pracy
        procesy_robocze = [self._kopiuj_proces(p) for p in procesy]
        procesy_robocze.sort(key=lambda x: (x.czas_przybycia, x.pid))
        
        indeks_procesu = 0  # Indeks do śledzenia, które procesy już przybyły
        
        while indeks_procesu < len(procesy_robocze) or self.kolejka_gotowych:
            # Dodaje do kolejki wszystkie procesy, które już przybyły
            while (indeks_procesu < len(procesy_robocze) and 
                   procesy_robocze[indeks_procesu].czas_przybycia <= self.aktualny_czas):
                self.kolejka_gotowych.append(procesy_robocze[indeks_procesu])
                indeks_procesu += 1
            
            if not self.kolejka_gotowych:
                 # przechodzi do następnego procesu gdy CPU bezczynny
                if indeks_procesu < len(procesy_robocze):
                    self.aktualny_czas = procesy_robocze[indeks_procesu].czas_przybycia
                else:
                    # Wszystkie procesy zakończone
                    break
            else:
                # Pobiera pierwszy proces z kolejki i wykonuje go przez kwant czasu
                aktualny_proces = self.kolejka_gotowych.popleft()
                self._wykonaj_proces(aktualny_proces, procesy_robocze, indeks_procesu)
        
        # Tworzy statystyki na podstawie zakończonych procesów i wykresu Gantta
        statystyki = StatystykiSzeregowania(self.procesy_ukonczone, self.wykres_gantta)
        return self.procesy_ukonczone.copy(), statystyki
    
    def _resetuj(self):
        """Resetuj stan algorytmu"""

        self.kolejka_gotowych.clear()
        self.wykres_gantta.clear()
        self.procesy_ukonczone.clear()
        self.aktualny_czas = 0
    
    def _kopiuj_proces(self, proces: Proces) -> Proces:
        """Skopiuj proces do lokalnej instancji"""

        return Proces(
            pid=proces.pid,
            czas_przybycia=proces.czas_przybycia,
            czas_wykonania=proces.czas_wykonania,
            priorytet=proces.priorytet
        )
    
    def _wykonaj_proces(self, proces: Proces, wszystkie_procesy: List[Proces], indeks_procesu: int):
        """Wykonaj proces przez kwant czasu lub do końca"""

        proces.ustaw_czas_startu(self.aktualny_czas)
        czas_wykonania = min(self.kwant_czasu, proces.pozostaly_czas)
        
        # Dodaje wpis do wykresu Gantta
        self.wykres_gantta.append(WpisGantta(
            id_procesu=proces.pid,
            czas_startu=self.aktualny_czas,
            dlugosc=czas_wykonania
        ))
        
        proces.pozostaly_czas -= czas_wykonania  # Odejmuje wykonany czas
        self.aktualny_czas += czas_wykonania  # Przesuwa czas symulacji
        
        # Dodaje do kolejki procesy, które przybyły w trakcie wykonywania
        indeks_nowych = indeks_procesu
        while (indeks_procesu < len(wszystkie_procesy) and 
               wszystkie_procesy[indeks_procesu].czas_przybycia <= self.aktualny_czas):
            indeks_procesu += 1

        for i in range(indeks_nowych, indeks_procesu):
            self.kolejka_gotowych.append(wszystkie_procesy[i])
        
        if proces.pozostaly_czas == 0:
            # Proces zakończony, zapisuje czasy i dodaje do zakończonych
            proces.czas_zakonczenia = self.aktualny_czas
            proces.oblicz_czasy()
            self.procesy_ukonczone.append(proces)
        else:
            # Proces nie zakończony, wraca na koniec kolejki
            self.kolejka_gotowych.append(proces)


if __name__ == "__main__":
    tytul = "Testowanie algorytmu RR"
    print(f'\n{"=" * len(tytul)}')
    print(tytul)
    print(f'{"=" * len(tytul)}\n')
    
    # tworzy przykładowe dane testowe
    generator = GeneratorProcesow(seed=42)
    procesy_procesora = generator.obciazenie_losowe(8, (0, 16), (2, 24), (1, 5))

    print("\nProcesy testowe:")
    for p in procesy_procesora:
        print(f"\tP{p.pid}: AT={p.czas_przybycia}, BT={p.czas_wykonania}")
    
    # test różnych kwantów
    for q in [2, 4]:
        algorytm = RoundRobin(kwant_czasu=q)
        ukonczone, statystyki = algorytm.symuluj(procesy_procesora)

        print(f"\nWyniki czasowe (q={q}):")
        for p in sorted(ukonczone, key=lambda x: x.pid):
            print(f"\tP{p.pid}: CT={p.czas_zakonczenia}, TAT={p.czas_obrotu}, WT={p.czas_oczekiwania}, RT={p.czas_odpowiedzi}")
        
        statystyki.wyswietl_podsumowanie(f"RR (q={q})")
