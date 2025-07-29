"""
Algorytm planowania czasu procesora SJF (Shortest Job First) wywłaszczający = SRTF (Shortest Remaining Time First)
"""

from typing import List, Tuple
from cpu_generator import Proces, WpisGantta, StatystykiSzeregowania, GeneratorProcesow


class SJFWywlaszczajacy(Proces):
    """Implementacja SJF z wywłaszczeniem (preemptive SJF) = SRTF (Shortest Remaining Time First)"""
    
    def __init__(self):
        super().__init__()
        self.wykres_gantta = []
        self.procesy_ukonczone = []
        self.aktualny_czas = 0
        self.liczba_wywlaszczen = 0
    
    def symuluj(self, procesy: List[Proces]) -> Tuple[List[Proces], StatystykiSzeregowania]:
        """Wykonaj szeregowanie SRTF"""

        self._resetuj()
        
        # kopiuje procesy do pracy
        procesy_robocze = [self._kopiuj_proces(p) for p in procesy]
        procesy_robocze.sort(key=lambda x: (x.czas_przybycia, x.pid))
        
        kolejka_gotowych = []
        indeks_procesu = 0
        aktualny_proces = None
        czas_startu_wykonania = 0
        
        while indeks_procesu < len(procesy_robocze) or kolejka_gotowych or aktualny_proces:
            # dodaje przybyłe procesy do kolejki
            while (indeks_procesu < len(procesy_robocze) and 
                   procesy_robocze[indeks_procesu].czas_przybycia <= self.aktualny_czas):
                kolejka_gotowych.append(procesy_robocze[indeks_procesu])
                indeks_procesu += 1
            
            # sprawdza czy trzeba wywłaszczyć aktualny proces
            if aktualny_proces and kolejka_gotowych:
                najkrotszy = min(kolejka_gotowych, key=lambda x: (x.pozostaly_czas, x.czas_przybycia, x.pid))
                if najkrotszy.pozostaly_czas < aktualny_proces.pozostaly_czas:
                    # wywłaszczenie i zapisanie wykonania aktualnego procesu
                    self._zapisz_wykonanie(aktualny_proces, czas_startu_wykonania)
                    kolejka_gotowych.append(aktualny_proces)
                    aktualny_proces = None
                    self.liczba_wywlaszczen += 1

            # wybiera proces do wykonania
            if not aktualny_proces:
                if kolejka_gotowych:
                    aktualny_proces = min(kolejka_gotowych, key=lambda x: (x.pozostaly_czas, x.czas_przybycia, x.pid))
                    kolejka_gotowych.remove(aktualny_proces)
                    aktualny_proces.ustaw_czas_startu(self.aktualny_czas)
                    czas_startu_wykonania = self.aktualny_czas
                elif indeks_procesu < len(procesy_robocze):
                    # przechodzi do następnego procesu gdy CPU bezczynny
                    self.aktualny_czas = procesy_robocze[indeks_procesu].czas_przybycia
                    continue
                else:
                    break
            
            # wykonuje przez jednostkę czasu
            if aktualny_proces:
                self.aktualny_czas += 1
                aktualny_proces.pozostaly_czas -= 1
                # sprawdza czy proces się zakończył
                if aktualny_proces.pozostaly_czas == 0:
                    self._zapisz_wykonanie(aktualny_proces, czas_startu_wykonania)
                    aktualny_proces.czas_zakonczenia = self.aktualny_czas
                    aktualny_proces.oblicz_czasy()
                    self.procesy_ukonczone.append(aktualny_proces)
                    aktualny_proces = None
        
        statystyki = StatystykiSzeregowania(self.procesy_ukonczone, self.wykres_gantta)
        return self.procesy_ukonczone.copy(), statystyki
    
    def _resetuj(self):
        """Resetuj stan algorytmu"""

        self.wykres_gantta.clear()
        self.procesy_ukonczone.clear()
        self.aktualny_czas = 0
        self.liczba_wywlaszczen = 0
    
    def _kopiuj_proces(self, proces: Proces) -> Proces:
        """Skopiuj proces do lokalnej instancji"""

        return Proces(
            pid=proces.pid,
            czas_przybycia=proces.czas_przybycia,
            czas_wykonania=proces.czas_wykonania,
            priorytet=proces.priorytet
        )
    
    def _zapisz_wykonanie(self, proces: Proces, czas_startu: int):
        """Zapisz wykonanie procesu"""

        dlugosc = self.aktualny_czas - czas_startu

        if dlugosc > 0:
            self.wykres_gantta.append(WpisGantta(
                id_procesu=proces.pid,
                czas_startu=czas_startu,
                dlugosc=dlugosc
            ))


if __name__ == "__main__":
    from cpu_generator import GeneratorProcesow

    tytul = "Testowanie algorytmu SRTF (SJF z wywłaszczeniem)"
    print(f'\n{"=" * len(tytul)}')
    print(tytul)
    print(f'{"=" * len(tytul)}\n')

    # tworzy przykładowe dane testowe
    generator = GeneratorProcesow(seed=42)
    procesy_procesora = generator.obciazenie_losowe(8, (0, 16), (2, 24), (1, 5))

    print("\nProcesy testowe:")
    for p in procesy_procesora:
        print(f"\tP{p.pid}: AT={p.czas_przybycia}, BT={p.czas_wykonania}")

    # uruchamia algorytm SRTF
    algorytm = SJFWywlaszczajacy()
    ukonczone, statystyki = algorytm.symuluj(procesy_procesora)

    print(f"\nWyniki czasowe:")
    for p in sorted(ukonczone, key=lambda x: x.pid):
        print(f"\tP{p.pid}: CT={p.czas_zakonczenia}, TAT={p.czas_obrotu}, WT={p.czas_oczekiwania}, RT={p.czas_odpowiedzi}")
    print(f"\twywłaszczenia: {algorytm.liczba_wywlaszczen}")
    
    # tworzy diagram Gantta
    print("\nDiagram Gantta:")
    for wpis in algorytm.wykres_gantta:
        print(f"\tP{wpis.id_procesu}: [{wpis.czas_startu}-{wpis.czas_konca}]")
    print()
    statystyki.wyswietl_podsumowanie("SRTF")
