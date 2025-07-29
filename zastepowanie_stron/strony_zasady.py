"""
Symulator algorytmów wymiany stron
    rejestruje, uruchamia i porównuje algorytmy wymiany stron
    zwraca statystyki dotyczące wydajności algorytmów w zależności od ilości ramek i zestawu danych
"""

import pandas as pd
import time
import os
from typing import List
from dataclasses import dataclass
from strony_generator import GeneratorStron, Strony


@dataclass
class WynikSymulacji:
    """Wynik pojedynczej symulacji."""

    algorytm: str
    liczba_ramek: int
    typ_sekwencji: str
    dlugosc_sekwencji: int
    bledy_stron: int
    trafienia_stron: int
    procent_wskaznika_trafien: float
    czas_wykonania: float


class SymulatorWymiany(GeneratorStron, Strony):
    """
    Symuluj algorytmy wymiany stron
    """
    
    def __init__(self):
        self.algorytmy: dict = {}
        self.liczba_ramek = [2, 4, 8, 16, 32]
        self.historia_wynikow = []

        self.zestawy_wejsciowe = []
        self.katalog_wejsciowy = os.path.join(os.path.dirname(__file__), 'dane_wejsciowe')
        self.katalog_wyjsciowy = os.path.join(os.path.dirname(__file__), 'dane_wyjsciowe')

        # tworzy katalogi jeśli nie istnieją
        os.makedirs(self.katalog_wejsciowy, exist_ok=True)
        os.makedirs(self.katalog_wyjsciowy, exist_ok=True)

    def zarejestruj_algorytm(self, nazwa: str, klasa_algorytmu):
        """
        Rejestruj nowy algorytm wymiany stron.
        """

        self.algorytmy[nazwa] = klasa_algorytmu

    def _zaladuj_algorytmy(self) -> List[str]:
        """
        Zwróć listę dostępnych algorytmów wymiany stron.
        """

        zaladowane_algorytmy = []

        # próbuje załadować i zarejestrować algorytmy, jeśli są dostępne
        try:
            from LFU import LFU
            self.zarejestruj_algorytm('LFU', LFU)
            zaladowane_algorytmy.append('LFU')
        except ImportError:
            print("LFU niedostępny")
        try:
            from LRU import LRU
            self.zarejestruj_algorytm('LRU', LRU)
            zaladowane_algorytmy.append('LRU')
        except ImportError:
            print("LRU niedostępny")
        return list(self.algorytmy.keys())

    def wyswietl_zestawy(self) -> List[str]:
        """
        Zwróć listę dostępnych zestawów sekwencji stron (folderów z plikiem CSV).
        """

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

    def wczytaj_zestaw(self, nazwa_zestawu: str) -> List[int]:
        """
        Wczytaj sekwencję stron z wybranego zestawu (pliku CSV).
        """

        folder_zestawu = os.path.join(self.katalog_wejsciowy, nazwa_zestawu)
        sciezka_csv = os.path.join(folder_zestawu, f"{nazwa_zestawu}.csv")
        if not os.path.exists(sciezka_csv):
            raise FileNotFoundError(f"Plik nie istnieje: {sciezka_csv}")
        df = pd.read_csv(sciezka_csv)
        return df['numer_strony'].tolist()

    def uruchom_test_algorytmu(self, nazwa_algorytmu: str, liczba_ramek: int, 
                       sekwencja: List[int], typ_sekwencji: str = "nieznany") -> WynikSymulacji:
        """
        Uruchom pojedynczy test algorytmu.
        """

        if nazwa_algorytmu not in self.algorytmy:
            raise ValueError(f"nieznany algorytm: {nazwa_algorytmu}")
        
        # tworzy instancję algorytmu
        algorytm = self.algorytmy[nazwa_algorytmu](liczba_ramek)

        # mierzy czas wykonania
        czas_start = time.time()
        bledy_stron, trafienia_stron = algorytm.symuluj(sekwencja)
        czas_wykonania = time.time() - czas_start

        # oblicza wskaźniki
        laczne_dostepy = len(sekwencja)
        procent_wskaznika_trafien = (trafienia_stron / laczne_dostepy * 100) if laczne_dostepy > 0 else 0
        return WynikSymulacji(
            algorytm=nazwa_algorytmu,
            liczba_ramek=liczba_ramek,
            typ_sekwencji=typ_sekwencji,
            dlugosc_sekwencji=laczne_dostepy,
            bledy_stron=bledy_stron,
            trafienia_stron=trafienia_stron,
            procent_wskaznika_trafien=procent_wskaznika_trafien,
            czas_wykonania=czas_wykonania
        )
    
    def uruchom_test_zestawu(self, algorytmy: List[str], liczba_ramek: List[int],
                           sekwencja: List[int], typ_sekwencji: str = "nieznany") -> List[WynikSymulacji]:
        """
        Porównuje wiele algorytmów na tej samej sekwencji.
        """

        wyniki = []
        for algorytm in algorytmy:
            for liczba_ramka in liczba_ramek:
                wynik = self.uruchom_test_algorytmu(algorytm, liczba_ramka, sekwencja, typ_sekwencji)
                wyniki.append(wynik)
        return wyniki
    
    def uruchom_testy(self) -> pd.DataFrame:
        """
        Uruchom testy dla wszystkich dostępnych zestawów i algorytmów.
        """

        print("\n>>>>>> symulowanie pracy <<<<<")
        zestawy_wejsciowe = self.wyswietl_zestawy()
        if not zestawy_wejsciowe:
            print("Brak zestawów danych w katalogu wejściowym")
            return pd.DataFrame()  # nie generuje automatycznie, tylko zwraca pusty DataFrame

        wszystkie_wyniki = []

        for nazwa_zestawu in zestawy_wejsciowe:
            sekwencja = self.wczytaj_zestaw(nazwa_zestawu)
            typ_sekwencji = nazwa_zestawu.split('_')[0] if '_' in nazwa_zestawu else 'nieznany'
            algorytmy = self._zaladuj_algorytmy()
            wyniki = self.uruchom_test_zestawu(
                algorytmy=algorytmy,
                liczba_ramek=self.liczba_ramek,
                sekwencja=sekwencja,
                typ_sekwencji=typ_sekwencji
            )
            wszystkie_wyniki.extend(wyniki)
            self.generuj_raport(wyniki, nazwa_zestawu)

        return self.wyniki_do_dataframe(wszystkie_wyniki)
    
    def wyniki_do_dataframe(self, wyniki: List[WynikSymulacji]) -> pd.DataFrame:
        """Konwertuj wyniki do DataFrame."""

        dane = []
        for wynik in wyniki:
            dane.append({
                'algorytm': wynik.algorytm,
                'liczba_ramek': wynik.liczba_ramek,
                'typ_sekwencji': wynik.typ_sekwencji,
                'dlugosc_sekwencji': wynik.dlugosc_sekwencji,
                'bledy_stron': wynik.bledy_stron,
                'trafienia_stron': wynik.trafienia_stron,
                'procent_wskaznika_trafien': wynik.procent_wskaznika_trafien,
                'czas_wykonania': wynik.czas_wykonania
            })
        return pd.DataFrame(dane)
    
    def generuj_raport(self, wyniki: List[WynikSymulacji], nazwa_zestawu: str = ""):
        """
        Generuj raport z symulacji algorytmów wymiany stron.
        """

        if not wyniki:
            return "Brak wyników symulacji."
        
        # tworzy DataFrame z wyników
        df = self.wyniki_do_dataframe(wyniki)
        print(f"\nWyniki porównania ({nazwa_zestawu}):")

        # wyświetla tabelę z najważniejszymi statystykami
        df_table = df[['algorytm', 'dlugosc_sekwencji', 'liczba_ramek', 'typ_sekwencji', 'bledy_stron', 'trafienia_stron', 'procent_wskaznika_trafien', 'czas_wykonania']].round(8).to_string(index=False)
        print('\n'.join(['\t' + line for line in df_table.split('\n')]))


if __name__ == "__main__":
    symulator = SymulatorWymiany()

    tytul = "Symulowanie algorytmów wymiany stron"
    print(f'\n{"=" * len(tytul)}')
    print(tytul)
    print(f'{"=" * len(tytul)}\n')

    generator = GeneratorStron(ziarno=42)

    print(f"\t{', '.join(symulator._zaladuj_algorytmy())}")
    print(f"\t{', '.join(symulator.wyswietl_zestawy())}")

    wyniki = symulator.uruchom_testy()
