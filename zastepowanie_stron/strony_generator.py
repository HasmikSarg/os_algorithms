"""
Generator sekwencji odwołań do stron
    generuje zestawy stron do testowania algorytmów wymiany stron
    zapisuje wyniki w dane_wejsciowe/ w osobnych katalogach w .csv i .json
"""

import numpy as np
import pandas as pd
import os
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Strony:
    """Konfiguruj generowanie sekwencji stron"""

    dlugosc: int
    zakres_stron: Tuple[int, int]
    typ_wzorca: str = "losowy"
    ziarno: Optional[int] = None
    wspolczynnik_lokalnosci: float = 0.8
    rozmiar_lokalnosci: int = 5
    rozmiar_roboczy: int = 10


class GeneratorStron:
    """Generuj sekwencje odwołań do stron o różnych charakterystykach"""

    def __init__(self, ziarno: Optional[int] = None):
        if ziarno is not None:
            np.random.seed(ziarno)
        self.ziarno = ziarno
        self.katalog_wejsciowy = "zastepowanie_stron/dane_wejsciowe"
        os.makedirs(self.katalog_wejsciowy, exist_ok=True)

    def _zapisz_csv(self, sekwencja: List[int], nazwa_pliku: str):
        """Zapisz sekwencję do CSV w osobnym folderze dla zestawu danych"""

        # wyodrębnia nazwę zestawu z nazwa_pliku
        nazwa_zestawu = nazwa_pliku.replace('.csv', '')

        # tworzy folder dla zestawu danych
        katalog_zestawu = os.path.join(self.katalog_wejsciowy, nazwa_zestawu)
        os.makedirs(katalog_zestawu, exist_ok=True)
        df = pd.DataFrame({
            'krok': range(1, len(sekwencja) + 1),
            'numer_strony': sekwencja
        })

        # zapisuje CSV w folderze zestawu danych
        sciezka_csv = os.path.join(katalog_zestawu, f"{nazwa_zestawu}.csv")
        df.to_csv(sciezka_csv, index=False)

        # zapisuje metadane zestawu
        metadane = {
            'typ_wzorca': nazwa_zestawu.split('_')[0],  # wyodrębnia typ wzorca z nazwy pliku
            'nazwa_zestawu': nazwa_zestawu,
            'dlugosc_sekwencji': len(sekwencja),
            'zakres_stron': f"{min(sekwencja)}-{max(sekwencja)}",
            'unikalne_strony': len(set(sekwencja)),
            'znacznik_czasowy_generowania': pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S"),
        }
        sciezka_metadanych = os.path.join(katalog_zestawu, 'metadane.json')
        import json
        with open(sciezka_metadanych, 'w') as f:
            json.dump(metadane, f, indent=2)
        print(f"\t{nazwa_zestawu}")

    def obciazenie_losowe(self, dlugosc: int, zakres_stron: Tuple[int, int]) -> List[int]:
        """Generuj losową sekwencję odwołań do stron"""

        konfiguracja = Strony(
            dlugosc=dlugosc,  # pptymalny rozmiar dla analizy statystycznej
            zakres_stron=zakres_stron,
            typ_wzorca="losowy",
            ziarno=self.ziarno,
            wspolczynnik_lokalnosci=0.5,  # neutralne - brak preferencji
            rozmiar_lokalnosci=5,
            rozmiar_roboczy=8
        )
        min_strona, max_strona = konfiguracja.zakres_stron
        sekwencja = np.random.randint(min_strona, max_strona + 1, size=konfiguracja.dlugosc).tolist()
        nazwa_pliku = f"losowy_{konfiguracja.dlugosc}_cp{max_strona}-{min_strona}_cw{konfiguracja.rozmiar_lokalnosci}-{konfiguracja.rozmiar_roboczy}"

        # zapisuje wygenerowany zestaw do pliku
        self._zapisz_csv(sekwencja, nazwa_pliku)
        return sekwencja

    def obciazenie_sekwencyjne(self, dlugosc: int, zakres_stron: Tuple[int, int]) -> List[int]:
        """Generuj sekwencyjną sekwencję odwołań do stron"""

        konfiguracja = Strony(
            dlugosc=dlugosc,
            zakres_stron=zakres_stron,
            typ_wzorca="sekwencyjny",
            ziarno=self.ziarno,
            wspolczynnik_lokalnosci=0.6,
            rozmiar_lokalnosci=4,
            rozmiar_roboczy=6
        )
        min_strona, max_strona = konfiguracja.zakres_stron
        sekwencja = []
        biezaca_strona = min_strona

        for _ in range(konfiguracja.dlugosc):
            sekwencja.append(biezaca_strona)
            biezaca_strona += 1
            if biezaca_strona > max_strona:
                biezaca_strona = min_strona
        nazwa_pliku = f"sekwencyjny_{konfiguracja.dlugosc}_cp{max_strona}-{min_strona}_cw{konfiguracja.rozmiar_lokalnosci}-{konfiguracja.rozmiar_roboczy}"

        # zapisuje wygenerowany zestaw do pliku
        self._zapisz_csv(sekwencja, nazwa_pliku)
        return sekwencja

    def obciazenie_lokalne(self, dlugosc: int, zakres_stron: Tuple[int, int], wspolczynnik_lokalnosci: float, rozmiar_lokalnosci: int, rozmiar_roboczy: int) -> List[int]:
        """generuje sekwencję z lokalnością czasową i przestrzenną."""
        konfiguracja = Strony(
            dlugosc=dlugosc,
            zakres_stron=zakres_stron,
            typ_wzorca="lokalny",
            ziarno=self.ziarno,
            wspolczynnik_lokalnosci=wspolczynnik_lokalnosci,  # silna lokalność temporalna
            rozmiar_lokalnosci=rozmiar_lokalnosci,   # większe skupiska
            rozmiar_roboczy=rozmiar_roboczy
        )
        min_strona, max_strona = konfiguracja.zakres_stron
        sekwencja = []

        # inicjalne okno lokalności
        aktualna_lokalnosc = set(np.random.choice(
            range(min_strona, max_strona + 1),
            size=min(konfiguracja.rozmiar_lokalnosci, max_strona - min_strona + 1),
            replace=False
        ))
        for _ in range(konfiguracja.dlugosc):
            if np.random.random() < konfiguracja.wspolczynnik_lokalnosci and aktualna_lokalnosc:
                # dostęp w obrębie lokalności
                strona = np.random.choice(list(aktualna_lokalnosc))
            else:
                # dostęp poza lokalnością
                strona = np.random.randint(min_strona, max_strona + 1)
                if np.random.random() < 0.3:
                    if len(aktualna_lokalnosc) >= konfiguracja.rozmiar_lokalnosci:
                        aktualna_lokalnosc.remove(np.random.choice(list(aktualna_lokalnosc)))
                    aktualna_lokalnosc.add(strona)
            sekwencja.append(strona)
        nazwa_pliku = f"lokalny_{konfiguracja.dlugosc}_cp{max_strona}-{min_strona}_cw{konfiguracja.rozmiar_lokalnosci}-{konfiguracja.rozmiar_roboczy}"

        # zapisuje wygenerowany zestaw do pliku
        self._zapisz_csv(sekwencja, nazwa_pliku)
        return sekwencja

    def obciazenie_robocze(self, dlugosc: int, zakres_stron: Tuple[int, int], wspolczynnik_lokalnosci: float, rozmiar_lokalnosci: int, rozmiar_roboczy: int) -> List[int]:
        """generuje sekwencję zgodną z zasadą zbioru roboczego"""
        konfiguracja = Strony(
            dlugosc=dlugosc,
            zakres_stron=zakres_stron,
            typ_wzorca="roboczy",
            ziarno=self.ziarno,
            wspolczynnik_lokalnosci=wspolczynnik_lokalnosci,
            rozmiar_lokalnosci=rozmiar_lokalnosci,
            rozmiar_roboczy=rozmiar_roboczy # większy zbiór roboczy
        )
        min_strona, max_strona = konfiguracja.zakres_stron
        sekwencja = []

        # dzieli na fazy ze zbiorem roboczym
        dlugosc_fazy = konfiguracja.dlugosc // 4
        zbiory_robocze = []

        for _ in range(4):
            zbior_roboczy = set(np.random.choice(
                range(min_strona, max_strona + 1),
                size=min(konfiguracja.rozmiar_roboczy, max_strona - min_strona + 1),
                replace=False
            ))
            zbiory_robocze.append(zbior_roboczy)
        aktualna_faza = 0
        for i in range(konfiguracja.dlugosc):
            # zmiana zbioru roboczego co dlugosc_fazy
            if i > 0 and i % dlugosc_fazy == 0:
                aktualna_faza = min(aktualna_faza + 1, len(zbiory_robocze) - 1)
            # 90% szans na dostęp w zbiorze roboczym
            if np.random.random() < 0.9 and zbiory_robocze[aktualna_faza]:
                strona = np.random.choice(list(zbiory_robocze[aktualna_faza]))
            else:
                strona = np.random.randint(min_strona, max_strona + 1)
            sekwencja.append(strona)
        nazwa_pliku = f"roboczy_{konfiguracja.dlugosc}_cp{max_strona}-{min_strona}_cw{konfiguracja.rozmiar_lokalnosci}-{konfiguracja.rozmiar_roboczy}"

        # zapisuje wygenerowany zestaw do pliku
        self._zapisz_csv(sekwencja, nazwa_pliku)
        return sekwencja

    def generuj_zestawy(self):
        """
        Generuj zestawy danych testowych
        """

        print(">>>>> generowanie zestawów <<<<<\n")

        # podstawowy zestaw - standardowe porównanie działania algorytmów na losowych sekwencjach
        self.obciazenie_losowe(dlugosc=500, zakres_stron=(0, 20))
        # lokalny zestaw - analiza adaptacji do zmian lokalności czasowej i przestrzennej
        self.obciazenie_lokalne(dlugosc=750, zakres_stron=(0, 20), wspolczynnik_lokalnosci=0.8, rozmiar_lokalnosci=6, rozmiar_roboczy=8)
        # roboczy zestaw - symulacja rzeczywistych wzorców testująca adaptację do lokalnych zmian i odporność na zjawisko thrashingu
        self.obciazenie_robocze(dlugosc=1000, zakres_stron=(0, 20), wspolczynnik_lokalnosci=0.7, rozmiar_lokalnosci=5, rozmiar_roboczy=10)
        # sekwencyjny zestaw - analiza zachowania stosu testująca właściwość inkluzji i odporność na anomalię Belady'ego
        self.obciazenie_sekwencyjne(dlugosc=500, zakres_stron=(0, 20))

        self._generuj_podsumowanie()

    def wczytaj_csv(self, nazwa_pliku: str) -> List[int]:
        """wczytuje sekwencję z CSV"""
        sciezka = os.path.join(self.katalog_wejsciowy, nazwa_pliku)
        df = pd.read_csv(sciezka)
        return df['numer_strony'].tolist()
    
    def _generuj_podsumowanie(self):
        """Generuje podsumowanie zestawów danych na podstawie metadanych"""


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
    generator = GeneratorStron(ziarno=42)
    tytul = "Generowanie sekwencji odwołań do stron"
    print(f'\n{"=" * len(tytul)}')
    print(tytul)
    print(f'{"=" * len(tytul)}\n')

    generator.generuj_zestawy()
    print(f"\nPliki zapisane w katalogu: {generator.katalog_wejsciowy}")
