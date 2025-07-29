"""
Generator zestawów procesów CPU
    generuje zestawy procesów do testowania algorytmów szeregowania sekwencji procesów
    zapisuje wyniki w dane_wejsciowe/ w osobnych katalogach w .csv i .json
"""

import numpy as np
import pandas as pd
import os
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Proces:
    """Podstawowe atrybuty szeregowania procesu CPU"""
    pid: int = 0
    czas_przybycia: int = 0
    czas_wykonania: int = 0
    priorytet: int = 1
    # obliczanie wartości
    czas_zakonczenia: int = 0
    czas_obrotu: int = 0
    czas_oczekiwania: int = 0
    czas_odpowiedzi: int = -1
    # śledzenie stanu
    pozostaly_czas: int = 0
    czas_startu: int = -1
    zostal_uruchomiony: bool = False
    
    def __post_init__(self):
        # Ustawia pozostały czas na czas wykonania przy inicjalizacji
        self.pozostaly_czas = self.czas_wykonania
    
    def oblicz_czasy(self):
        """Oblicz wyniki czasowe po zakończeniu procesu"""

        self.czas_obrotu = self.czas_zakonczenia - self.czas_przybycia
        self.czas_oczekiwania = self.czas_obrotu - self.czas_wykonania
        if self.czas_startu != -1:
            self.czas_odpowiedzi = self.czas_startu - self.czas_przybycia
    
    def ustaw_czas_startu(self, czas: int):
        """Ustaw czas pierwszego przydziału CPU"""

        if not self.zostal_uruchomiony:
            self.czas_startu = czas
            self.zostal_uruchomiony = True
    
    def __repr__(self):
        # Czytelna reprezentacja procesu
        return f"P{self.pid} (AT:{self.czas_przybycia}, BT:{self.czas_wykonania})"


@dataclass
class WpisGantta:
    """Wpis procesu CPU w wykresie Gantta"""

    id_procesu: int
    czas_startu: int
    dlugosc: int
    
    @property
    def czas_konca(self):
        # Zwraca czas zakończenia fragmentu
        return self.czas_startu + self.dlugosc


class StatystykiSzeregowania:
    """Statystyki wydajności algorytmów szeregowania procesów CPU"""
    
    def __init__(self, procesy: List[Proces], wykres_gantta: List[WpisGantta]):
        self.procesy = procesy
        self.wykres_gantta = wykres_gantta
        self.statystyki = self._oblicz()

    def pobierz_wartosc(self, nazwa: str) -> float:
        # Zwraca wartość wybranej statystyki
        return self.statystyki.get(nazwa, 0.0)
    
    def _oblicz(self) -> Dict[str, float]:
        """Oblicz statystyki wydajności algorytmów"""

        n = len(self.procesy)
        if n == 0:
            return {}
        
        # oblicza podstawowe czasy
        sredni_czas_obrotu = sum(p.czas_obrotu for p in self.procesy) / n
        sredni_czas_oczekiwania = sum(p.czas_oczekiwania for p in self.procesy) / n
        sredni_czas_odpowiedzi = sum(p.czas_odpowiedzi for p in self.procesy if p.czas_odpowiedzi >= 0) / n
        sredni_czas_zakonczenia = sum(p.czas_zakonczenia for p in self.procesy) / n
        
        # zwraca podstawowe statystyki
        return {
            'sredni_czas_obrotu': sredni_czas_obrotu,
            'sredni_czas_oczekiwania': sredni_czas_oczekiwania,
            'sredni_czas_odpowiedzi': sredni_czas_odpowiedzi,
            'sredni_czas_zakonczenia': sredni_czas_zakonczenia,
        }
    
    def wyswietl_podsumowanie(self, algorytm: str = ""):
        """
        Wyświetl podsumowanie statystyk dla danego algorytmu.
        """
        if algorytm:
            print(f"Podsumowanie {algorytm}:")
        print(f"\tliczba procesów: {len(self.procesy)}")
        print(f"\tśredni czas zakończenia (ACT): {self.statystyki['sredni_czas_zakonczenia']:.2f}")
        print(f"\tśredni czas cyklu (ATAT): {self.statystyki['sredni_czas_obrotu']:.2f}")
        print(f"\tśredni czas oczekiwania (AWT): {self.statystyki['sredni_czas_oczekiwania']:.2f}")
        print(f"\tśredni czas odpowiedzi (ART): {self.statystyki['sredni_czas_odpowiedzi']:.2f}")


class GeneratorProcesow:
    """Generator zestawów sekwencji procesów zależnych od parametrów"""

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.katalog_wejsciowy = "szeregowanie_cpu/dane_wejsciowe"
        os.makedirs(self.katalog_wejsciowy, exist_ok=True)
        if seed is not None:
            np.random.seed(seed)
    
    def obciazenie_losowe(self, n: int, 
                       zakres_przybycia: Tuple[int, int], 
                       zakres_wykonania: Tuple[int, int], 
                       zakres_priorytetu: Tuple[int, int]) -> List[Proces]:
        """Generuj losowe procesy z typowymi czasami przybycia i wykonania"""

        procesy = []

        # generuje n zadań o losowych parametrach
        for i in range(n):
            proces = Proces(
                pid=i + 1,
                czas_przybycia=np.random.randint(zakres_przybycia[0], zakres_przybycia[1] + 1),
                czas_wykonania=np.random.randint(zakres_wykonania[0], zakres_wykonania[1] + 1),
                priorytet=np.random.randint(zakres_priorytetu[0], zakres_priorytetu[1] + 1)
            )
            procesy.append(proces)
        nazwa_pliku = f"losowy_{n}_cp{zakres_przybycia[0]}-{zakres_przybycia[1]}_cw{zakres_wykonania[0]}-{zakres_wykonania[1]}"

        # zapisuje wygenerowany zestaw do pliku
        self._zapisz_zestaw(procesy, nazwa_pliku, {
            'typ': 'losowy',
            'liczba_procesow': n,
            'opis': 'Losowe procesy z typowymi czasami przybycia i wykonania'
        })
        return procesy
    
    def obciazenie_sekwencyjne(self, n: int, 
                               interwal: int,
                               zakres_wykonania: Tuple[int, int],
                               wzorzec_wykonania: str) -> List[Proces]:
        """Generuj procesy z sekwencyjnymi czasami przybycia"""

        procesy = []
        for i in range(n):
            # wybiera wzorzec wykonania i generuje n zadań
            if wzorzec_wykonania == 'rosnacy':
                czas_wykonania = zakres_wykonania[0] + (i * (zakres_wykonania[1] - zakres_wykonania[0]) // n)
            elif wzorzec_wykonania == 'malejacy':
                czas_wykonania = zakres_wykonania[1] - (i * (zakres_wykonania[1] - zakres_wykonania[0]) // n)
            else:
                czas_wykonania = np.random.randint(zakres_wykonania[0], zakres_wykonania[1] + 1)
            proces = Proces(
                pid=i + 1,
                czas_przybycia=i * interwal,
                czas_wykonania=czas_wykonania,
                priorytet=np.random.randint(1, 6)
            )
            procesy.append(proces)
        if wzorzec_wykonania == 'rosnacy':
            wzorzec_wykonania = '->'
        elif wzorzec_wykonania == 'malejacy':
            wzorzec_wykonania = '<-'
        else:
            wzorzec_wykonania = '--'
        nazwa_pliku = f"sekwencyjny_{n}_i{interwal}_cw{zakres_wykonania[0]}-{zakres_wykonania[1]}_{wzorzec_wykonania}"

        # zapisuje wygenerowany zestaw do pliku
        self._zapisz_zestaw(procesy, nazwa_pliku, {
            'typ': 'sekwencyjny',
            'liczba_procesow': n,
            'opis': 'Procesy z sekwencyjnymi czasami przybycia'
        })
        return procesy

    def obciazenie_robocze(self, n: int,
                            stosunek_krotkich: float,
                            zakres_krotkich_wykonan: Tuple[int, int],
                            zakres_dlugich_wykonan: Tuple[int, int]) -> List[Proces]:
        """Generuj mieszane procesy z krótkimi i długimi zadaniami"""

        procesy = []
        n_krotkich = int(n * stosunek_krotkich)  # liczba krótkich zadań

        # generuje n krótkich zadań
        for i in range(n_krotkich):
            proces = Proces(
                pid=i + 1,
                czas_przybycia=np.random.randint(0, max(10, n // 4)),
                czas_wykonania=np.random.randint(zakres_krotkich_wykonan[0], zakres_krotkich_wykonan[1] + 1),
                priorytet=np.random.randint(1, 3)  # wyższy priorytet dla krótkich zadań
            )
            procesy.append(proces)

        # generuje długie zadania
        for i in range(n - n_krotkich):
            proces = Proces(
                pid=n_krotkich + i + 1,
                czas_przybycia=np.random.randint(0, max(15, n // 3)),
                czas_wykonania=np.random.randint(zakres_dlugich_wykonan[0], zakres_dlugich_wykonan[1] + 1),
                priorytet=np.random.randint(3, 6)  # niższy priorytet dla długich zadań
            )
            procesy.append(proces)

        # sortuje procesy według czasu przybycia i PID
        procesy.sort(key=lambda p: (p.czas_przybycia, p.pid))
        
        nazwa_pliku = f"mieszany_{n}_krotkie{int(stosunek_krotkich*100)}"

        # zapisuje wygenerowany zestaw do pliku
        self._zapisz_zestaw(procesy, nazwa_pliku, {
            'typ': 'mieszany',
            'opis': 'Mieszane procesy z krótkimi i długimi zadaniami'
        })
        return procesy

    def obciazenie_cpu(self, n: int, 
                       min_wykonanie: int,
                       max_wykonanie: int) -> List[Proces]:
        """Generuj obciążenie intensywnie wykorzystujące CPU z długimi czasami wykonania"""

        procesy = []
        for i in range(n):
            # tworzy n zadań o długim czasie wykonania
            proces = Proces(
                pid=i + 1,
                czas_przybycia=np.random.randint(0, n // 2),
                czas_wykonania=np.random.randint(min_wykonanie, max_wykonanie + 1),
                priorytet=np.random.randint(1, 5)
            )
            procesy.append(proces)

        # sortuje procesy według czasu przybycia
        procesy.sort(key=lambda p: p.czas_przybycia)
        nazwa_pliku = f"intensywne_cpu_{n}_cw{min_wykonanie}-{max_wykonanie}"

        # zapisuje wygenerowany zestaw do pliku
        self._zapisz_zestaw(procesy, nazwa_pliku, {
            'typ': 'intensywne_cpu',
            'opis': 'Obciążenie intensywnie wykorzystujące CPU z długimi czasami wykonania'
        })
        return procesy
    
    def obciazenie_io(self, n: int, 
                      max_wykonanie: int,
                      zakres_priorytetu: Tuple[int, int]) -> List[Proces]:
        """Generuj obciążenie intensywnie wykorzystujące I/O z krótkimi czasami wykonania"""

        procesy = []
        for i in range(n):
            # tworzy n zadań o krótkim czasie wykonania
            proces = Proces(
                pid=i + 1,
                czas_przybycia=np.random.randint(0, n),
                czas_wykonania=np.random.randint(1, max_wykonanie + 1),
                priorytet=np.random.randint(zakres_priorytetu[0], zakres_priorytetu[1] + 1)
            )
            procesy.append(proces)

        # sortuje procesy według czasu przybycia
        procesy.sort(key=lambda p: p.czas_przybycia)
        nazwa_pliku = f"intensywne_io_{n}_cw1-{max_wykonanie}"

        # zapisuje wygenerowany zestaw do pliku
        self._zapisz_zestaw(procesy, nazwa_pliku, {
            'typ': 'intensywne_io',
            'opis': 'Obciążenie intensywnie wykorzystujące I/O z krótkimi czasami wykonania'
        })
        return procesy
    

    def _zapisz_zestaw(self, procesy: List[Proces], nazwa_pliku: str, metadane: dict):
        """Zapisz zestaw danych z metadanymi"""

        # tworzy folder dla zestawu
        katalog_zestawu = os.path.join(self.katalog_wejsciowy, nazwa_pliku)
        os.makedirs(katalog_zestawu, exist_ok=True)
        
        # przygotowuje dane procesów do zapisu
        dane = [{
            'PID': p.pid,
            'Czas_Przybycia': p.czas_przybycia,
            'Czas_Wykonania': p.czas_wykonania,
            'Priorytet': p.priorytet
        } for p in procesy]
        df = pd.DataFrame(dane)

        # zapisuje plik CSV z danymi procesów
        sciezka_csv = os.path.join(katalog_zestawu, f"{nazwa_pliku}.csv")
        df.to_csv(sciezka_csv, index=False)

        # przygotowuje metadane zestawu
        metadane = {
            **metadane,
            'nazwa_zestawu': nazwa_pliku,
            'liczba_procesow': len(procesy),
            'zakres_czasow_przybycia': f"{min(p.czas_przybycia for p in procesy)}-{max(p.czas_przybycia for p in procesy)}",
            'zakres_czasow_wykonania': f"{min(p.czas_wykonania for p in procesy)}-{max(p.czas_wykonania for p in procesy)}",
            'zakres_priorytetow': f"{min(p.priorytet for p in procesy)}-{max(p.priorytet for p in procesy)}",
            'sredni_czas_wykonania': sum(p.czas_wykonania for p in procesy) / len(procesy),
            'sredni_czas_przybycia': sum(p.czas_przybycia for p in procesy) / len(procesy),
            'sredni_priorytet': sum(p.priorytet for p in procesy) / len(procesy),
            'znacznik_czasowy': pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S"),
        }

        # zapisuje plik JSON z metadanymi
        sciezka_metadanych = os.path.join(katalog_zestawu, 'metadane.json')
        with open(sciezka_metadanych, 'w', encoding='utf-8') as f:
            json.dump(metadane, f, indent=2, ensure_ascii=False)
        print(f"\t{nazwa_pliku}")


    def generuj_zestawy(self):
        """
        Generuj zestawy danych testowych
        """

        print(">>>>> generowanie zestawów <<<<<\n")

        # podstawowy zestaw - standardowe porównanie działania algorytmów na losowych procesach
        self.obciazenie_losowe(
            n=100,
            zakres_przybycia=(0, 50),
            zakres_wykonania=(10, 50),
            zakres_priorytetu=(1, 5)
        )
        # sekwencyjny zestaw - analiza zachowania przy sekwencyjnym przybyciu procesów
        self.obciazenie_sekwencyjne(
            n=100,
            interwal=2,
            zakres_wykonania=(1, 50),
            wzorzec_wykonania='rosnacy'  # 'malejacy' lub 'losowy'
        )
        # rzeczywisty zestaw - symulacja rzeczywistych wzorców obciążenia systemów
        self.obciazenie_robocze(
            n=100,
            stosunek_krotkich=0.7,
            zakres_krotkich_wykonan=(1, 5),
            zakres_dlugich_wykonan=(10, 20)
        )
        # zestaw obciążenia CPU - intensywne zadania
        self.obciazenie_cpu(
            n=100,
            min_wykonanie=25,
            max_wykonanie=100
        )
        # zestaw obciążenia IO - częste operacje wejścia/wyjścia
        self.obciazenie_io(
            n=100,
            max_wykonanie=10,
            zakres_priorytetu=(1, 5)
        )

        self._generuj_podsumowanie()

    def _generuj_podsumowanie(self):
        """Generuj podsumowanie zestawów danych"""

        dane_podsumowania = []
        
        # przegląda wszystkie katalogi zestawów i zbiera metadane
        for element in os.listdir(self.katalog_wejsciowy):
            sciezka_elementu = os.path.join(self.katalog_wejsciowy, element)
            if os.path.isdir(sciezka_elementu):
                plik_metadanych = os.path.join(sciezka_elementu, 'metadane.json')
                if os.path.exists(plik_metadanych):
                    with open(plik_metadanych, 'r', encoding='utf-8') as f:
                        metadane = json.load(f)
                    dane_podsumowania.append(metadane)

        # zapisuje podsumowanie do pliku CSV
        if dane_podsumowania:
            df_podsumowania = pd.DataFrame(dane_podsumowania)
            plik_podsumowania = os.path.join(self.katalog_wejsciowy, 'podsumowanie_zestawow.csv')
            df_podsumowania.to_csv(plik_podsumowania, index=False)
            print(f"{plik_podsumowania}")


if __name__ == "__main__":
    generator = GeneratorProcesow(seed=42)
    tytul = "Generowanie zestawów procesów CPU"
    print(f'\n{"=" * len(tytul)}')
    print(tytul)
    print(f'{"=" * len(tytul)}\n')
    
    generator.generuj_zestawy()
    print(f"\nPliki zapisane w katalogu: {generator.katalog_wejsciowy}")
