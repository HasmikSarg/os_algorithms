"""
Symulator działania algorytmów planowania czasu procesora.
    rejestruje, uruchamia i porównuje algorytmy szeregowania sekwencji procesów
    zwraca statystyki dotyczące wydajności algorytmów w zależności od wartości kwantu i zestawu danych
"""

import pandas as pd
import time
import os
from typing import List, Optional
from dataclasses import dataclass
from cpu_generator import Proces, GeneratorProcesow


@dataclass
class WynikSymulacji:
    """Wynik pojedynczej symulacji"""

    algorytm: str
    typ_zestawu: str
    liczba_procesow: int
    sredni_czas_oczekiwania: float
    sredni_czas_obrotu: float
    sredni_czas_odpowiedzi: float
    sredni_czas_zakonczenia: float
    czas_wykonania: float
    kwant: Optional[int] = None


class SymulatorSzeregowania(GeneratorProcesow, Proces):
    """
    Symuluj algorytmy szeregowania procesów CPU
    """

    def __init__(self):
        super().__init__()

        self.algorytmy: dict = {}
        self.wartosci_kwantu = [2, 4, 8, 16, 32]
        self.historia_wynikow = []

        self.zestawy_wejsciowe = []
        self.katalog_wejsciowy = os.path.join(os.path.dirname(__file__), 'dane_wejsciowe')
        self.katalog_wyjsciowy = os.path.join(os.path.dirname(__file__), 'dane_wyjsciowe')

        # tworzy katalogi jeśli nie istnieją
        os.makedirs(self.katalog_wejsciowy, exist_ok=True)
        os.makedirs(self.katalog_wyjsciowy, exist_ok=True)
    
    def zarejestruj_algorytm(self, nazwa: str, klasa_algorytmu, **domyslne_argumenty):
        """
        Zarejestruj algorytm szeregowania
        """

        # dodaje algorytm do słownika pod podaną nazwą
        self.algorytmy[nazwa] = {
            'klasa': klasa_algorytmu,
            'domyslne_argumenty': domyslne_argumenty
        }

    def _zaladuj_algorytmy(self):
        """Załaduj dostępne algorytmy szeregowania"""

        zaladowane_algorytmy = []
        # próbuje załadować i zarejestrować algorytmy, jeśli są dostępne
        try:
            from Round_Robin import RoundRobin
            self.zarejestruj_algorytm('RR', RoundRobin)
            zaladowane_algorytmy.append('Round-Robin')
        except ImportError:
            print("Round-Robin niedostępny")
        try:
            from SJF_niewywl import SJFNiewywlaszczajacy
            self.zarejestruj_algorytm('SJF', SJFNiewywlaszczajacy)
            zaladowane_algorytmy.append('SJF Niewywłaszczający')
        except ImportError:
            print("SJF Niewywłaszczający niedostępny")
        try:
            from SJF_wywl import SJFWywlaszczajacy
            self.zarejestruj_algorytm('SRTF', SJFWywlaszczajacy)
            zaladowane_algorytmy.append('SJF Wywłaszczający (SRTF)')
        except ImportError:
            print("SJF Wywłaszczający (SRTF) niedostępny")
        return zaladowane_algorytmy
    
    def wyswietl_zestawy(self) -> List[str]:
        """Wyświetl dostępne zestawy danych"""

        # przeszukuje katalog wejściowy i zwraca listę folderów z plikiem CSV
        if not os.path.exists(self.katalog_wejsciowy):
            return []
        
        foldery_zestawow = []
        for element in os.listdir(self.katalog_wejsciowy):
            sciezka_elementu = os.path.join(self.katalog_wejsciowy, element)
            if os.path.isdir(sciezka_elementu):
                plik_csv = os.path.join(sciezka_elementu, f"{element}.csv")
                if os.path.exists(plik_csv):
                    foldery_zestawow.append(element)

        return sorted(foldery_zestawow)
    
    def wczytaj_zestaw(self, nazwa_zestawu: str) -> List[Proces]:
        """Wczytaj zestaw danych z pliku CSV"""

        # buduje ścieżkę do pliku CSV na podstawie nazwy zestawu
        folder_zestawu = os.path.join(self.katalog_wejsciowy, nazwa_zestawu)
        sciezka_csv = os.path.join(folder_zestawu, f"{nazwa_zestawu}.csv")
        # sprawdza czy plik istnieje
        if not os.path.exists(sciezka_csv):
            raise FileNotFoundError(f"Plik nie istnieje: {sciezka_csv}")
        # wczytuje dane z pliku CSV do DataFrame
        df = pd.read_csv(sciezka_csv)
        procesy = []
        # tworzy obiekty Proces na podstawie wierszy z pliku
        for _, wiersz in df.iterrows():
            proces = Proces(
                pid=int(wiersz['PID']),
                czas_przybycia=int(wiersz['Czas_Przybycia']),
                czas_wykonania=int(wiersz['Czas_Wykonania']),
                priorytet=int(wiersz['Priorytet'])
            )
            procesy.append(proces)
        
        return procesy
        
    def uruchom_test_algorytmu(self, nazwa_algorytmu: str, procesy: List[Proces], 
                        typ_zestawu: str = "nieznany", **kwargs) -> WynikSymulacji:
        """
        Uruchom pojedynczy test algorytmu
        """

        # sprawdza czy algorytm jest zarejestrowany
        if nazwa_algorytmu not in self.algorytmy:
            raise ValueError(f"Nieznany algorytm: {nazwa_algorytmu}")
        # przygotowuje argumenty dla algorytmu (łączy domyślne i przekazane)
        info_alg = self.algorytmy[nazwa_algorytmu]
        argumenty_alg = info_alg['domyslne_argumenty'].copy()
        argumenty_alg.update(kwargs)
        
        # tworzy instancję algorytmu
        algorytm = info_alg['klasa'](**argumenty_alg)
        # tworzy kopie procesów do testu (nie modyfikuje oryginału)
        procesy_procesora = [self._kopiuj_proces(p) for p in procesy]
        # mierzy czas wykonania algorytmu
        czas_startu = time.time()
        try:
            ukonczone_procesy, statystyki = algorytm.symuluj(procesy_procesora)
            czas_wykonania = time.time() - czas_startu
            # tworzy wynik symulacji na podstawie statystyk
            wynik = WynikSymulacji(
                algorytm=nazwa_algorytmu,
                typ_zestawu=typ_zestawu,
                liczba_procesow=len(procesy),
                sredni_czas_oczekiwania=statystyki.pobierz_wartosc('sredni_czas_oczekiwania'),
                sredni_czas_obrotu=statystyki.pobierz_wartosc('sredni_czas_obrotu'),
                sredni_czas_odpowiedzi=statystyki.pobierz_wartosc('sredni_czas_odpowiedzi'),
                sredni_czas_zakonczenia=statystyki.pobierz_wartosc('sredni_czas_zakonczenia'),
                czas_wykonania=czas_wykonania,
                kwant=kwargs.get('kwant_czasu')
            )

            return wynik 
        except Exception as e:
            print(f"Błąd wykonania {nazwa_algorytmu}: {str(e)}")

            # zwraca pusty wynik w przypadku błędu
            return WynikSymulacji(
                algorytm=nazwa_algorytmu,
                typ_zestawu=typ_zestawu,
                liczba_procesow=len(procesy),
                sredni_czas_oczekiwania=float('inf'),
                sredni_czas_obrotu=float('inf'),
                sredni_czas_odpowiedzi=float('inf'),
                sredni_czas_zakonczenia=float('inf'),
                czas_wykonania=time.time() - czas_startu,
                kwant=kwargs.get('kwant_czasu')
            )
    
    def uruchom_test_zestawu(self, procesy: List['Proces'],
                               typ_zestawu: str = "nieznany", algorytmy: Optional[List[str]] = None, **kwargs) -> List[WynikSymulacji]:
        """
        Uruchom pojedynczy test zestawu danych
        """

        wyniki = []  # lista wyników dla wszystkich algorytmów
        # określa które algorytmy testować (wszystkie jeśli nie podano)
        algorytmy_do_testu = algorytmy if algorytmy is not None else list(self.algorytmy.keys())

        for algorytm in algorytmy_do_testu:
            if algorytm in self.algorytmy:
                kwargs_clean = {k: v for k, v in kwargs.items() if k != 'algorytmy'}
                # uruchamia test pojedynczego algorytmu
                wynik = self.uruchom_test_algorytmu(algorytm, procesy, typ_zestawu, **kwargs_clean)
                wyniki.append(wynik)
                self.historia_wynikow.append(wynik)
            else:
                print(f"Algorytm {algorytm} niedostępny")
        return wyniki

    def uruchom_testy(self) -> pd.DataFrame:
        """
        Uruchom testy dla wszystkich dostępnych zestawów i algorytmów.
        """

        print("\n>>>>>> symulowanie pracy <<<<<")
        self.zestawy_wejsciowe = self.wyswietl_zestawy()
        # jeśli nie ma zestawów, generuje je automatycznie
        if not self.zestawy_wejsciowe:
            print("Brak zestawów danych w katalogu wejściowym")
            GeneratorProcesow().generuj_zestawy()
            self.zestawy_wejsciowe = self.wyswietl_zestawy()

        wszystkie_wyniki = []  # zbiera wszystkie wyniki

        # przetwarza każdy zestaw danych
        for nazwa_zestawu in self.zestawy_wejsciowe:
            # wczytuje procesy z pliku
            procesy = self.wczytaj_zestaw(nazwa_zestawu)
            # określa typ zestawu na podstawie nazwy
            typ_zestawu = nazwa_zestawu.split('_')[0] if '_' in nazwa_zestawu else 'nieznany'
            wyniki_zestawu = []  # wyniki dla tego zestawu
            for algorytm in self.algorytmy:
                if algorytm == 'RR':
                    # dla RR testuje różne wartości kwantu
                    for kwant in self.wartosci_kwantu:
                        wyniki = self.uruchom_test_zestawu(
                            procesy=procesy,
                            typ_zestawu=typ_zestawu,
                            algorytmy=[algorytm],
                            kwant_czasu=kwant
                        )
                        wyniki_zestawu.extend(wyniki)
                        wszystkie_wyniki.extend(wyniki)
                else:
                    # dla pozostałych algorytmów testuje tylko raz
                    wyniki = self.uruchom_test_zestawu(
                        procesy=procesy,
                        typ_zestawu=typ_zestawu,
                        algorytmy=[algorytm]
                    )
                    wyniki_zestawu.extend(wyniki)
                    wszystkie_wyniki.extend(wyniki)
            # generuje raport tekstowy dla zestawu
            self.generuj_raport(procesy, wyniki_zestawu, nazwa_zestawu)

        # zwraca wyniki jako DataFrame
        return self.wyniki_do_dataframe(wszystkie_wyniki)

    
    def wyniki_do_dataframe(self, wyniki: List[WynikSymulacji]) -> pd.DataFrame:
        """Konwertuj wyniki do DataFrame"""

        # tworzy listę słowników z wynikami
        dane = []

        for wynik in wyniki:
            dane.append({
                'algorytm': wynik.algorytm,
                'typ_zestawu': wynik.typ_zestawu,
                'liczba_procesow': wynik.liczba_procesow,
                'sredni_czas_oczekiwania': wynik.sredni_czas_oczekiwania,
                'sredni_czas_obrotu': wynik.sredni_czas_obrotu,
                'sredni_czas_odpowiedzi': wynik.sredni_czas_odpowiedzi,
                'sredni_czas_zakonczenia': wynik.sredni_czas_zakonczenia,
                'czas_wykonania': wynik.czas_wykonania,
                'kwant': wynik.kwant
            })
        # Zwraca DataFrame z wynikami
        return pd.DataFrame(dane)
    
    def _kopiuj_proces(self, proces: 'Proces') -> 'Proces':
        """Kopiuj proces"""

        # tworzy nową instancję procesu z tymi samymi parametrami
        return Proces(
            pid=proces.pid,
            czas_przybycia=proces.czas_przybycia,
            czas_wykonania=proces.czas_wykonania,
            priorytet=proces.priorytet
        )

    def generuj_raport(self, procesy_procesora, wyniki: List[WynikSymulacji], nazwa_zestawu: str = ""):
        """
        Generuj raport z symulacji.
        """

        if not wyniki:
            return "Brak wyników symulacji."
        
        # tworzy DataFrame z wyników
        df = symulator.wyniki_do_dataframe(wyniki)
        print(f"\nWyniki porównania ({nazwa_zestawu}):")

        # wyświetla tabelę z najważniejszymi statystykami
        df_table = df[['algorytm', 'kwant', 'sredni_czas_oczekiwania', 'sredni_czas_obrotu', 'sredni_czas_odpowiedzi', 'sredni_czas_zakonczenia']].round(8).to_string(index=False)
        print('\n'.join(['\t' + line for line in df_table.split('\n')]))


if __name__ == "__main__":
    symulator = SymulatorSzeregowania()
    tytul = "Symulowanie algorytmów szeregowania CPU"
    print(f'\n{"=" * len(tytul)}')
    print(tytul)
    print(f'{"=" * len(tytul)}\n')

    generator = GeneratorProcesow(seed=42)

    print(f"\t{', '.join(symulator._zaladuj_algorytmy())}")
    print(f"\t{', '.join(symulator.wyswietl_zestawy())}")

    wyniki = symulator.uruchom_testy()
