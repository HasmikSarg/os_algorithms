"""
Symulacja działania algorytmów zastępowania stron dla zestawu stron.
    uruchamia testy na danych z dane_wejsciowe/ i eksportuje wyniki do dane_wyjsciowe/
    analizuje i tworzy zestawienia wyników działania algorytmów w .csv, wykresy wydajności w .png oraz raporty końcowe w .txt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
from datetime import datetime

from strony_zasady import SymulatorWymiany, Strony
from strony_generator import GeneratorStron


class AnalizaWymiany(SymulatorWymiany, GeneratorStron, Strony):
    """
    Analizuj algorytmy wymiany stron.
    """
    
    def __init__(self):
        super().__init__()

        self.generator = GeneratorStron()
        self.symulator = SymulatorWymiany()

        self.katalog_wejsciowy = os.path.join(os.path.dirname(__file__), 'dane_wejsciowe')
        self.katalog_wyjsciowy = os.path.join(os.path.dirname(__file__), 'dane_wyjsciowe')
        os.makedirs(self.katalog_wejsciowy, exist_ok=True)
        os.makedirs(self.katalog_wyjsciowy, exist_ok=True)

        # ustawienia wykresów
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        self.schemat_kolorow = {
            'wskaznik_trafien': '#2ecc71',
            'bledy_stron': '#e74c3c',
            'czas_wykonania': '#f39c12',
            'histogram': '#9b59b6',
            'optymalny': '#27ae60',
            'linia_srednia': '#c0392b',
            'linia_mediana': '#d35400',
            'slupki_przedzialow': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        }

        self.kolory_algorytmow = {
            'LRU': '#3498db',
            'LFU': '#e67e22',
        }

        self._zaladuj_algorytmy()
        
    def stworz_wykresy(self, wyniki: List, nazwa_zestawu: str):
        """Stwórz wykresy dla zestawu danych"""

        katalog_wyjsciowy_zestawu = os.path.join(self.katalog_wyjsciowy, nazwa_zestawu)
        os.makedirs(katalog_wyjsciowy_zestawu, exist_ok=True)
        
        df_wyniki = self.symulator.wyniki_do_dataframe(wyniki)
        
        # przygotowuje dane
        df_wykres = df_wyniki.copy()
        df_wykres['nazwa_algorytmu'] = df_wykres.apply(
            lambda wiersz: f"{wiersz['algorytm']} (q={int(wiersz['kwant'])})" 
            if wiersz['algorytm'] == 'RR' and pd.notna(wiersz['kwant']) 
            else wiersz['algorytm'], axis=1
        )
        
        # ANALIZA 1: Analiza wynikow
        self._stworz_wykresy_wynikow(wyniki, nazwa_zestawu)

        # ANALIZA 2: Analiza ramek
        self._stworz_analiza_ramek(df_wyniki, nazwa_zestawu)

        df_wyniki.to_csv(os.path.join(katalog_wyjsciowy_zestawu, f'{nazwa_zestawu}_wyniki.csv'), index=False)

    def _stworz_wykresy_wynikow(self, wyniki: List, nazwa_zestawu: str):
        """Wykres wyników PH, PF, HR, FR dla każdego algorytmu"""

        # tworzy folder dla wykresów zestawu
        katalog_wyjsciowy_zestawu = os.path.join(self.katalog_wyjsciowy, nazwa_zestawu)
        os.makedirs(katalog_wyjsciowy_zestawu, exist_ok=True)
        df_wyniki = self.symulator.wyniki_do_dataframe(wyniki)
        if df_wyniki.empty:
            print(f"Brak danych dla zestawu {nazwa_zestawu}")
            return
        dane_wykresu = df_wyniki.groupby('algorytm', observed=False).agg({
            'procent_wskaznika_trafien': 'mean',
            'bledy_stron': 'mean',
            'czas_wykonania': 'mean'
        }).reset_index()
        plt.figure(figsize=(15, 10), constrained_layout=True)

        # Wykres 1: wskaźnik trafień vs liczba ramek
        plt.subplot(2, 2, 1)
        for algorytm in df_wyniki['algorytm'].unique():
            dane_alg = df_wyniki[df_wyniki['algorytm'] == algorytm]
            dane_srednie = dane_alg.groupby('liczba_ramek', observed=False)['procent_wskaznika_trafien'].mean().reset_index()
            kolor = self.kolory_algorytmow.get(algorytm, self.schemat_kolorow['wskaznik_trafien'])
            plt.plot(dane_srednie['liczba_ramek'], dane_srednie['procent_wskaznika_trafien'], 
                    marker='o', label=algorytm, linewidth=2, markersize=6, color=kolor)
        plt.xlabel('Liczba ramek')
        plt.ylabel('Wskaźnik trafień (%)')
        plt.title(f'Wskaźnik trafień', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Wykres 2: błędy stron vs liczba ramek
        plt.subplot(2, 2, 2)
        for algorytm in df_wyniki['algorytm'].unique():
            dane_alg = df_wyniki[df_wyniki['algorytm'] == algorytm]
            dane_srednie = dane_alg.groupby('liczba_ramek', observed=False)['bledy_stron'].mean().reset_index()
            kolor = self.kolory_algorytmow.get(algorytm, self.schemat_kolorow['bledy_stron'])
            plt.plot(dane_srednie['liczba_ramek'], dane_srednie['bledy_stron'], 
                    marker='s', label=algorytm, linewidth=2, markersize=6, color=kolor)
        plt.xlabel('Liczba ramek')
        plt.ylabel('Liczba błędów stron')
        plt.title(f'Błędy stron', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Wykres 3: porównanie średnich wskaźników trafień
        plt.subplot(2, 2, 3)
        kolory = [self.kolory_algorytmow.get(alg, self.schemat_kolorow['wskaznik_trafien']) for alg in dane_wykresu['algorytm']]
        slupki = plt.bar(dane_wykresu['algorytm'], dane_wykresu['procent_wskaznika_trafien'], 
                      color=kolory, alpha=0.8)
        plt.xlabel('Algorytm')
        plt.ylabel('Średni wskaźnik trafień (%)')
        plt.title(f'Średni wskaźnik trafień', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        for slupek in slupki:
            wysokosc = slupek.get_height()
            plt.text(slupek.get_x() + slupek.get_width()/2., wysokosc,
                    f'{wysokosc:.1f}%', ha='center', va='bottom', fontsize=10)
        plt.grid(True, alpha=0.3)

        # Wykres 4: przedziały średniej ilości błędów stron
        plt.subplot(2, 2, 4)
        statystyki_przedzialow = self._oblicz_przedzialy_bledow(df_wyniki, liczba_przedzialow=7)
        if not statystyki_przedzialow.empty:
            algorytmy = statystyki_przedzialow['algorytm'].unique()
            przedzialy = statystyki_przedzialow['przedzial_bledow'].unique()
            przedzialy = [x for x in przedzialy if pd.notna(x)]
            if len(przedzialy) > 0 and len(algorytmy) > 0:
                poz_x = np.arange(len(przedzialy))
                szerokosc = 0.35
                # dla każdego algorytmu tworzy słupki
                for i, algorytm in enumerate(algorytmy):
                    dane_alg = statystyki_przedzialow[statystyki_przedzialow['algorytm'] == algorytm]
                    liczebnosci = []
                    for przedzial in przedzialy:
                        liczba_przedzial = dane_alg[dane_alg['przedzial_bledow'] == przedzial]['liczba'].sum()
                        liczebnosci.append(liczba_przedzial)
                    # oblicza pozycję słupków
                    pozycje = poz_x + (i - len(algorytmy)/2 + 0.5) * szerokosc / len(algorytmy)
                    kolor = self.kolory_algorytmow.get(algorytm, self.schemat_kolorow['slupki_przedzialow'][i % len(self.schemat_kolorow['slupki_przedzialow'])])
                    plt.bar(pozycje, liczebnosci, szerokosc/len(algorytmy), 
                           label=algorytm, alpha=0.8, color=kolor)
                plt.xlabel('Przedziały błędów stron')
                plt.ylabel('Liczba przypadków')
                plt.title(f'Rozkład błędów stron', fontsize=12, fontweight='bold')
                plt.xticks(poz_x, przedzialy, rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'Brak danych\ndo przedziałów', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes, fontsize=12)
                plt.title(f'Przedziały błędów stron - {nazwa_zestawu}')
        else:
            plt.text(0.5, 0.5, 'Brak danych\ndo przedziałów błędów stron', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'Przedziały błędów stron - {nazwa_zestawu}')
        plt.suptitle(f'Analiza wyników - {nazwa_zestawu}', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(katalog_wyjsciowy_zestawu, f'{nazwa_zestawu}_przeglad_wydajnosci.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _stworz_analiza_ramek(self, df_wyniki: pd.DataFrame, nazwa_zestawu: str):
        """Szczegółowa analiza ramek"""

        katalog_wyjsciowy_zestawu = os.path.join(self.katalog_wyjsciowy, nazwa_zestawu)
        plt.figure(figsize=(10, 15), constrained_layout=True)

        # Wykres 1: Czas wykonania algorytmów vs liczba ramek
        plt.subplot(2, 1, 1)
        for algorytm in df_wyniki['algorytm'].unique():
            dane_alg = df_wyniki[df_wyniki['algorytm'] == algorytm]
            dane_srednie = dane_alg.groupby('liczba_ramek', observed=False)['czas_wykonania'].mean().reset_index()
            kolor = self.kolory_algorytmow.get(algorytm, self.schemat_kolorow['czas_wykonania'])
            plt.plot(dane_srednie['liczba_ramek'], dane_srednie['czas_wykonania'], marker='o', label=algorytm, linewidth=2, markersize=6, color=kolor)
        plt.xlabel('Liczba ramek')
        plt.ylabel('Średni czas wykonania (s)')
        plt.title(f'Czas wykonania algorytmów vs liczba ramek', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Wykres 2: Czas wykonania algorytmów vs błędy stron
        plt.subplot(2, 1, 2)
        for algorytm in df_wyniki['algorytm'].unique():
            dane_alg = df_wyniki[df_wyniki['algorytm'] == algorytm]
            kolor = self.kolory_algorytmow.get(algorytm, self.schemat_kolorow['czas_wykonania'])
            plt.plot(dane_alg['bledy_stron'], dane_alg['czas_wykonania'], marker='o', label=algorytm, linewidth=2, markersize=6, color=kolor)
        plt.xlabel('Liczba błędów stron')
        plt.ylabel('Czas wykonania (s)')
        plt.title(f'Czas wykonania vs. liczba błędów stron', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'Analiza czasu wykonania - {nazwa_zestawu}', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(katalog_wyjsciowy_zestawu, f'{nazwa_zestawu}_czas_vs_bledy.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def stworz_wykresy_podsumowujace(self, df_wyniki: pd.DataFrame):
        """
        Stwórz wykresy podsumowujące wydajności algorytmów wymiany stron
        """

        print("\n>>>>> tworzenie wykresów <<<<<")
        if df_wyniki.empty:
            print("Brak danych do utworzenia wykresów")
            return
        plt.figure(figsize=(16, 12))

        # Wykres 1: Wskaźnik trafień vs liczba ramek
        plt.subplot(3, 2, 1)
        for algorytm in df_wyniki['algorytm'].unique():
            dane_alg = df_wyniki[df_wyniki['algorytm'] == algorytm]
            dane_srednie = dane_alg.groupby('liczba_ramek', observed=False)['procent_wskaznika_trafien'].mean().reset_index()
            kolor = self.kolory_algorytmow.get(algorytm, self.schemat_kolorow['wskaznik_trafien'])
            plt.plot(dane_srednie['liczba_ramek'], dane_srednie['procent_wskaznika_trafien'], 
                    marker='o', label=algorytm, linewidth=2, color=kolor)
        plt.xlabel('Liczba ramek')
        plt.ylabel('Średni wskaźnik trafień (%)')
        plt.title('Wydajność algorytmów vs liczba ramek')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Wykres 2: Porównanie według typu sekwencji
        plt.subplot(3, 2, 2)
        dane_pivot = df_wyniki.groupby(['algorytm', 'typ_sekwencji'], observed=False)['procent_wskaznika_trafien'].mean().reset_index()
        if not dane_pivot.empty:
            tabela_pivot = dane_pivot.pivot(index='algorytm', columns='typ_sekwencji', values='procent_wskaznika_trafien')
            tabela_pivot.plot(kind='bar', ax=plt.gca(), alpha=0.8, 
                           color=[self.kolory_algorytmow.get(alg, self.schemat_kolorow['wskaznik_trafien']) for alg in tabela_pivot.index])
            plt.xlabel('Algorytm')
            plt.ylabel('Średni wskaźnik trafień (%)')
            plt.title('Wydajność według typu sekwencji')
            plt.xticks(rotation=45)
            plt.legend(title='Typ sekwencji', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Wykres 3: Przedziały błędów stron (globalnie)
        plt.subplot(3, 2, 3)
        statystyki_przedzialow = self._oblicz_przedzialy_bledow(df_wyniki, liczba_przedzialow=7)
        if not statystyki_przedzialow.empty:
            algorytmy = statystyki_przedzialow['algorytm'].unique()
            przedzialy = statystyki_przedzialow['przedzial_bledow'].unique()
            przedzialy = [x for x in przedzialy if pd.notna(x)]
            if len(przedzialy) > 0 and len(algorytmy) > 0:
                poz_x = np.arange(len(przedzialy))
                szerokosc = 0.35
                for i, algorytm in enumerate(algorytmy):
                    dane_alg = statystyki_przedzialow[statystyki_przedzialow['algorytm'] == algorytm]
                    liczebnosci = []
                    for przedzial in przedzialy:
                        liczba_przedzial = dane_alg[dane_alg['przedzial_bledow'] == przedzial]['liczba'].sum()
                        liczebnosci.append(liczba_przedzial)
                    pozycje = poz_x + (i - len(algorytmy)/2 + 0.5) * szerokosc / len(algorytmy)
                    kolor = self.kolory_algorytmow.get(algorytm, self.schemat_kolorow['slupki_przedzialow'][i % len(self.schemat_kolorow['slupki_przedzialow'])])
                    plt.bar(pozycje, liczebnosci, szerokosc/len(algorytmy), 
                           label=algorytm, alpha=0.8, color=kolor)
                plt.xlabel('Przedziały błędów stron')
                plt.ylabel('Liczba przypadków')
                plt.title('Globalne przedziały błędów stron')
                plt.xticks(poz_x, przedzialy, rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)

        # Wykres 4: Błędy stron vs liczba ramek
        plt.subplot(3, 2, 4)
        for algorytm in df_wyniki['algorytm'].unique():
            dane_alg = df_wyniki[df_wyniki['algorytm'] == algorytm]
            dane_srednie = dane_alg.groupby('liczba_ramek', observed=False)['bledy_stron'].mean().reset_index()
            kolor = self.kolory_algorytmow.get(algorytm, self.schemat_kolorow['bledy_stron'])
            plt.plot(dane_srednie['liczba_ramek'], dane_srednie['bledy_stron'], 
                    marker='s', label=algorytm, linewidth=2, color=kolor)
        plt.xlabel('Liczba ramek')
        plt.ylabel('Średnia liczba błędów stron')
        plt.title('Błędy stron vs liczba ramek')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Wykres 5: Czas wykonania
        plt.subplot(3, 2, 5)
        dane_wykonania = df_wyniki.groupby('algorytm', observed=False)['czas_wykonania'].mean().reset_index()
        kolory = [self.kolory_algorytmow.get(alg, self.schemat_kolorow['czas_wykonania']) for alg in dane_wykonania['algorytm']]
        slupki = plt.bar(dane_wykonania['algorytm'], dane_wykonania['czas_wykonania'], 
                      alpha=0.8, color=kolory)
        plt.xlabel('Algorytm')
        plt.ylabel('Średni czas wykonania (ms)')
        plt.title('Czas wykonania algorytmów')
        plt.xticks(rotation=45)
        for slupek in slupki:
            wysokosc = slupek.get_height()
            plt.text(slupek.get_x() + slupek.get_width()/2., wysokosc,
                    f'{wysokosc:.8f}', ha='center', va='bottom', fontsize=9)
            
        # Wykres 6: Podsumowanie wydajności
        plt.subplot(3, 2, 6)
        dane_podsumowania = df_wyniki.groupby('algorytm', observed=False).agg({
            'procent_wskaznika_trafien': 'mean',
            'bledy_stron': 'mean'
        }).reset_index()
        x = range(len(dane_podsumowania))
        szerokosc = 0.35
        os1 = plt.gca()
        slupki1 = os1.bar([i - szerokosc/2 for i in x], dane_podsumowania['procent_wskaznika_trafien'], 
                       szerokosc, label='Wskaźnik trafień (%)', alpha=0.8, color=self.schemat_kolorow['wskaznik_trafien'])
        os1.set_ylabel('Wskaźnik trafień (%)', color=self.schemat_kolorow['wskaznik_trafien'])
        os1.tick_params(axis='y', labelcolor=self.schemat_kolorow['wskaznik_trafien'])
        os2 = os1.twinx()
        slupki2 = os2.bar([i + szerokosc/2 for i in x], dane_podsumowania['bledy_stron'], 
                       szerokosc, label='Błędy stron', alpha=0.8, color=self.schemat_kolorow['bledy_stron'])
        os2.set_ylabel('Błędy stron', color=self.schemat_kolorow['bledy_stron'])
        os2.tick_params(axis='y', labelcolor=self.schemat_kolorow['bledy_stron'])
        plt.title('Podsumowanie wydajności')
        os1.set_xticks(x)
        os1.set_xticklabels(dane_podsumowania['algorytm'], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.katalog_wyjsciowy, 'przeglad_wydajnosci_stron.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    
    def _oblicz_przedzialy_bledow(self, df_wyniki: pd.DataFrame, liczba_przedzialow: int = 7):
        """
        OOblicz przedziały błędów stron
        """

        if df_wyniki.empty:
            return pd.DataFrame()
        
        # znajduje zakres błędów stron
        min_bledow = df_wyniki['bledy_stron'].min()
        max_bledow = df_wyniki['bledy_stron'].max()

        # tworzy przedziały
        krawedzie_przedzialow = np.linspace(min_bledow, max_bledow + 1, liczba_przedzialow + 1)
        etykiety_przedzialow = []
        for i in range(liczba_przedzialow):
            start = int(krawedzie_przedzialow[i])
            koniec = int(krawedzie_przedzialow[i + 1]) - 1
            if i == liczba_przedzialow - 1:
                koniec = int(krawedzie_przedzialow[i + 1])
            etykiety_przedzialow.append(f"{start}-{koniec}")

        # przypisuje każdy wynik do przedziału
        df_wyniki['przedzial_bledow'] = pd.cut(
            df_wyniki['bledy_stron'], 
            bins=krawedzie_przedzialow, 
            labels=etykiety_przedzialow, 
            include_lowest=True,
            right=False
        )

        # oblicza średnią liczbę błędów dla każdego algorytmu w każdym przedziale
        statystyki_przedzialow = df_wyniki.groupby(['algorytm', 'przedzial_bledow'], observed=False).agg({
            'bledy_stron': ['mean', 'count'],
            'procent_wskaznika_trafien': 'mean'
        }).round(4)
        statystyki_przedzialow.columns = ['srednie_bledy', 'liczba', 'sredni_wskaznik_trafien']
        statystyki_przedzialow = statystyki_przedzialow.reset_index()
        return statystyki_przedzialow

    def uruchom_porownanie(self, algorytmy: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Porównaj zestawy z wykresami i raportem
        """

        print("\n>>>>> uruchamianie porównania <<<<<")
        zestawy_wejsciowe = self.wyswietl_zestawy()
        if not zestawy_wejsciowe:
            print("Brak zestawów danych w katalogu wejściowym")
            self.generator.generuj_zestawy()
            zestawy_wejsciowe = self.wyswietl_zestawy()

        if algorytmy is None:
            algorytmy = list(self.symulator.algorytmy.keys())

        wszystkie_wyniki = []

        # przetwarza każdy zestaw danych osobno
        for nazwa_zestawu in zestawy_wejsciowe:
            print(f"Przetwarzanie zestawu: {nazwa_zestawu}")
            # wczytuje sekwencję
            sekwencja = self.wczytaj_zestaw(nazwa_zestawu)
            typ_zestawu = nazwa_zestawu.split('_')[0] if '_' in nazwa_zestawu else 'nieznany'
            # uruchamia testy dla tego zestawu
            wyniki_zestawu = []
            for algorytm in algorytmy:
                if algorytm in self.symulator.algorytmy:
                    wyniki = self.symulator.uruchom_test_zestawu(
                        algorytmy=[algorytm],
                        liczba_ramek=self.liczba_ramek,
                        sekwencja=sekwencja,
                        typ_sekwencji=typ_zestawu
                    )
                    wyniki_zestawu.extend(wyniki)
                    wszystkie_wyniki.extend(wyniki)
            # tworzy wykresy i raport tekstowy dla zestawu
            if wyniki_zestawu:
                self.stworz_wykresy(wyniki_zestawu, nazwa_zestawu)
                df_zestawu = self.symulator.wyniki_do_dataframe(wyniki_zestawu)
                raport_zestawu = self.stworz_raport(df_zestawu, tytul=f"RAPORT ZESTAWU: {nazwa_zestawu}")
                katalog_wyjsciowy_zestawu = os.path.join(self.katalog_wyjsciowy, nazwa_zestawu)
                os.makedirs(katalog_wyjsciowy_zestawu, exist_ok=True)
                plik_raportu = os.path.join(katalog_wyjsciowy_zestawu, f"{nazwa_zestawu}_raport.txt")
                with open(plik_raportu, 'w', encoding='utf-8') as f:
                    f.write(raport_zestawu)

        # zapisuje połączone wyniki
        df_wyniki = self.symulator.wyniki_do_dataframe(wszystkie_wyniki)
        plik_wyjsciowy = os.path.join(self.katalog_wyjsciowy, 'wyniki_wymianastron.csv')
        df_wyniki.to_csv(plik_wyjsciowy, index=False)

        # tworzy raport dla każdego typu zestawu
        if 'typ_sekwencji' in df_wyniki.columns:
            for typ_zestawu in sorted(df_wyniki['typ_sekwencji'].unique()):
                dane_typu = df_wyniki[df_wyniki['typ_sekwencji'] == typ_zestawu]
                if dane_typu.empty:
                    continue
                raport_typu = self.stworz_raport(dane_typu, tytul=f"RAPORT TYPU ZESTAWU: {typ_zestawu}")
                plik_raportu_typu = os.path.join(self.katalog_wyjsciowy, f"{typ_zestawu}_raport.txt")
                with open(plik_raportu_typu, 'w', encoding='utf-8') as f:
                    f.write(raport_typu)

        # tworzy wykresy podsumowujące
        self.stworz_wykresy_podsumowujace(df_wyniki)
        return df_wyniki


    def stworz_raport(self, df_wyniki: pd.DataFrame, tytul: str = "RAPORT") -> str:
        """
        Generuj raport tekstowy.
        """
        
        if df_wyniki.empty:
            return f"{tytul}\nBrak danych do raportu."
        tresc_raportu = []

        tresc_raportu = [
            f" {'_'*(len(tytul)+8)} ",
            f"|{' '*(len(tytul)+8)}|",
            f"|    {tytul}    |",
            f"|{'_'*(len(tytul)+8)}|",
            "",
            ">>>>>> LEGENDA <<<<<<",
            "\tPA (page accesses): żądania dostępu stron",
            "\tPH (page hits): trafienia stron",
            "\tPF (page faults): błędy stron",
            "\tHR (hit rate): wskaźnik trafień = (PH / PA) * 100%",
            "\tFR (fault rate): wskaźnik pudeł = (PF / PA) * 100%",
            "\tRT (response time): średni czas odpowiedzi",
            "",
            ">>>>>> PARAMETRY <<<<<<",
            f"\tLiczba testów: {len(df_wyniki)}",
            f"\tTestowane algorytmy: {sorted(df_wyniki['algorytm'].unique())}",
            f"\tTypy zestawów: {sorted(df_wyniki['typ_sekwencji'].unique())}",
            "",
        ]
        
        # analiza algorytmów
        if not df_wyniki.empty:
            tresc_raportu.extend([
                ">>>>>> STATYSTYKA <<<<<<",
            ])
            
            analiza_algorytmow = df_wyniki.groupby('algorytm', observed=False).agg({
                'bledy_stron': ['mean', 'min', 'max'],
                'procent_wskaznika_trafien': ['mean', 'min', 'max'],
                'czas_wykonania': ['mean', 'min', 'max'],
            }).round(8)
            
            # formatuje nagłówki kolumn
            analiza_algorytmow.columns = [f'{metric}_{stat}' for metric, stat in analiza_algorytmow.columns]
            
            # tworzy sformatowaną tabelę
            header_line1 = f"\t{'':<16}|{'błędy strony':<24}{' ':<4}{'wskaźnik trafień':<24}"
            header_line2 = f"\t{'algorytm':<8}{'':<8}|{'mean':<8}{'min':<8}{'max':<8}{' ':<4}{'mean':<8}{'min':<8}{'max':<8}"
            separator_line = f"\t{'-'*8}{'-'*8}|{'-'*8*7}"
            tresc_raportu.extend([
                header_line1,
                header_line2,
                separator_line,
            ])
            
            # dodaje dane dla każdego algorytmu
            bledy_srednie = []
            trafienia_srednie = []
            wykonanie_średnie = []
            for algorytm in analiza_algorytmow.index:
                dane_algorytmu = analiza_algorytmow.loc[algorytm]
                # formatuje dane
                bledy_mean = f"{dane_algorytmu['bledy_stron_mean']:<8}"
                bledy_min = f"{dane_algorytmu['bledy_stron_min']:<8}"
                bledy_max = f"{dane_algorytmu['bledy_stron_max']:<4}"
                
                trafienia_mean = f"{dane_algorytmu['procent_wskaznika_trafien_mean']:<8}"
                trafienia_min = f"{dane_algorytmu['procent_wskaznika_trafien_min']:<8}"
                trafienia_max = f"{dane_algorytmu['procent_wskaznika_trafien_max']:<4}"

                bledy_srednie += [dane_algorytmu['bledy_stron_mean']]
                trafienia_srednie += [dane_algorytmu['procent_wskaznika_trafien_mean']]
                wykonanie_średnie += [dane_algorytmu['czas_wykonania_mean']]
                
                linia_algorytmu = f"{algorytm:<16}|{bledy_mean}{bledy_min}{bledy_max}{' ':<8}{trafienia_mean}{trafienia_min}{trafienia_max}"
                tresc_raportu.append(f"\t{linia_algorytmu}")
            
            tresc_raportu.append("")
        
        # najlepsze wyniki czasowe
        if not df_wyniki.empty:
            najlepszy_trafienia = df_wyniki.loc[df_wyniki['procent_wskaznika_trafien'].idxmax()]
            najmniej_bledow = df_wyniki.loc[df_wyniki['bledy_stron'].idxmin()]
            najszybszy = df_wyniki.loc[df_wyniki['czas_wykonania'].idxmin()]
            tresc_raportu.extend([
                ">>>>>> WYNIKI <<<<<<",
                f"\tNajwyższy AHR dla: {najlepszy_trafienia['algorytm']} = {najlepszy_trafienia['procent_wskaznika_trafien']:.2f}%",
                f"\tNajniższy APF dla: {najmniej_bledow['algorytm']} = {najmniej_bledow['bledy_stron']:.0f}",
                f"\tNajkrótszy ART dla: {najszybszy['algorytm']} = {najszybszy['czas_wykonania']:.6f}s",
                "",
            ])
        tresc_raportu.extend([
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ])
        return "\n".join(tresc_raportu)

    def generuj_analize(self):
        """Generuj analizę z wykresami i raportami"""
        
        df_wyniki = self.uruchom_porownanie()
        print("\n>>>>> tworzenie raportu <<<<<")
        raport_koncowy = self.stworz_raport(df_wyniki, tytul="ANALIZA ALGORYTMÓW WYMIANY STRON")
        plik_raportu_koncowego = os.path.join(self.katalog_wyjsciowy, "raport_wymianastron.txt")
        with open(plik_raportu_koncowego, "w", encoding="utf-8") as f:
            f.write(raport_koncowy)
            
        print("\nPliki wyjściowe:")
        for plik in sorted(os.listdir(self.katalog_wyjsciowy)):
            if os.path.isfile(os.path.join(self.katalog_wyjsciowy, plik)):
                print(f"\t{plik}")
            elif os.path.isdir(os.path.join(self.katalog_wyjsciowy, plik)):
                print(f"\t{plik}/ (folder z wykresami zestawu)")

        return df_wyniki


if __name__ == "__main__":
    banner=r'''

 ___                   ___            _                                   _      ___  _                 _    _    _              
| . \ ___  ___  ___   | . \ ___  ___ | | ___  ___  ___ ._ _ _  ___ ._ _ _| |_   | . || | ___  ___  _ _ <_> _| |_ | |_ ._ _ _  ___
|  _/<_> |/ . |/ ._>  |   // ._>| . \| |<_> |/ | '/ ._>| ' ' |/ ._>| ' | | |    |   || |/ . |/ . \| '_>| |  | |  | . || ' ' |<_-<
|_|  <___|\_. |\___.  |_\_\\___.|  _/|_|<___|\_|_.\___.|_|_|_|\___.|_|_| |_|    |_|_||_|\_. |\___/|_|  |_|  |_|  |_|_||_|_|_|/__/
          <___'                 |_|                                                     <___'                                    
'''

    print(banner)
    symulacja = AnalizaWymiany()
    tytul = "Analizowanie algorytmów wymiany stron"
    print(f'\n{"=" * len(tytul)}')
    print(tytul)
    print(f'{"=" * len(tytul)}\n')

    print(f"\t{', '.join(symulacja.symulator._zaladuj_algorytmy())}")
    print(f"\t{', '.join(symulacja.symulator.wyswietl_zestawy())}")

    symulacja.generuj_analize()
