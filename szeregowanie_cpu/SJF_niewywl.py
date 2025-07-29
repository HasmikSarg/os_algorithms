"""
Algorytm planowania czasu procesora SJF (Shortest Job First) niewywłaszczający = SJF
"""

from typing import List, Tuple
from cpu_generator import Proces, WpisGantta, StatystykiSzeregowania, GeneratorProcesow


class SJFNiewywlaszczajacy(Proces):
    """Implementacja SJF bez wywłaszczenia (non-preemptive SJF)"""
    
    def __init__(self):
        super().__init__()
        self.wykres_gantta = []
        self.procesy_ukonczone = []
        self.aktualny_czas = 0
    
    def symuluj(self, procesy: List[Proces]) -> Tuple[List[Proces], StatystykiSzeregowania]:
        """Wykonaj szeregowanie SJF"""

        self._resetuj()
        
        # kopiuje procesy do pracy
        procesy_robocze = [self._kopiuj_proces(p) for p in procesy]
        procesy_robocze.sort(key=lambda x: (x.czas_przybycia, x.pid))
        
        kolejka_gotowych = []
        indeks_procesu = 0
        
        while indeks_procesu < len(procesy_robocze) or kolejka_gotowych:
            # dodaje przybyłe procesy do kolejki
            while (indeks_procesu < len(procesy_robocze) and
                   procesy_robocze[indeks_procesu].czas_przybycia <= self.aktualny_czas):
                kolejka_gotowych.append(procesy_robocze[indeks_procesu])
                indeks_procesu += 1
            
            if not kolejka_gotowych:
                # przechodzi do następnego procesu gdy CPU bezczynny
                if indeks_procesu < len(procesy_robocze):
                    self.aktualny_czas = procesy_robocze[indeks_procesu].czas_przybycia
            else:
                # wybiera najkrótszy proces do wykonania
                wybrany = min(kolejka_gotowych, key=lambda x: (x.czas_wykonania, x.czas_przybycia, x.pid))
                kolejka_gotowych.remove(wybrany)
                self._wykonaj_proces(wybrany)
        
        statystyki = StatystykiSzeregowania(self.procesy_ukonczone, self.wykres_gantta)
        return self.procesy_ukonczone.copy(), statystyki
    
    def _resetuj(self):
        """Resetuj stan algorytmu"""

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
    
    def _wykonaj_proces(self, proces: Proces):
        """Wykonaj proces bez wywłaszczenia"""

        proces.ustaw_czas_startu(self.aktualny_czas)
        
        self.wykres_gantta.append(WpisGantta(
            id_procesu=proces.pid,
            czas_startu=self.aktualny_czas,
            dlugosc=proces.czas_wykonania
        ))

        self.aktualny_czas += proces.czas_wykonania
        proces.czas_zakonczenia = self.aktualny_czas
        proces.oblicz_czasy()
        self.procesy_ukonczone.append(proces)


if __name__ == "__main__":
    tytul = "Testowanie algorytmu SJF bez wywłaszczenia"
    print(f'\n{"=" * len(tytul)}')
    print(tytul)
    print(f'{"=" * len(tytul)}\n')

    # tworzy przykładowe dane testowe
    generator = GeneratorProcesow(seed=42)
    procesy_procesora = generator.obciazenie_losowe(8, (0, 16), (2, 24), (1, 5))

    print("\nProcesy testowe:")
    for p in procesy_procesora:
        print(f"\tP{p.pid}: AT={p.czas_przybycia}, BT={p.czas_wykonania}")
    
    # uruchamia algorytm SJF
    algorytm = SJFNiewywlaszczajacy()
    ukonczone, statystyki = algorytm.symuluj(procesy_procesora)
    
    print("\nWyniki czasowe:")
    for p in sorted(ukonczone, key=lambda x: x.pid):
        print(f"\tP{p.pid}: CT={p.czas_zakonczenia}, TAT={p.czas_obrotu}, WT={p.czas_oczekiwania}, RT={p.czas_odpowiedzi}")
    
    # tworzy diagram Gantta
    print("\nDiagram Gantta:")
    for wpis in algorytm.wykres_gantta:
        print(f"\tP{wpis.id_procesu}: [{wpis.czas_startu}-{wpis.czas_konca}]")
    print()
    statystyki.wyswietl_podsumowanie("SJF")
