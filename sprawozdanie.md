# Sprawozdanie – Sztuczne Sieci Neuronowe
# Projekt cz. I

**Zbiór danych:** marriage_data_india.csv (10 000 obserwacji, 16 cech wejściowych)

---

## 1. Opis podjętych problemów

### Problem 1 – Klasyfikacja: Przewidywanie rozwodu

Celem jest przewidzenie, czy dane małżeństwo zakończyło się rozwodem, na podstawie 16 cech opisujących małżonków i okoliczności ślubu (m.in. typ małżeństwa, wiek w chwili zawarcia związku, poziom wykształcenia, religia, akceptacja rodziców, poziom dochodów).

Zmienna docelowa: **Divorce_Status** (wartości: Tak / Nie → zakodowane jako 1 / 0).

Jest to **klasyfikacja binarna** – sieć na wyjściu zwraca jedną liczbę z przedziału [0, 1], interpretowaną jako prawdopodobieństwo rozwodu. Jeśli wartość przekracza 0,5 – sieć przewiduje rozwód.

Dane są **silnie niezbalansowane**: około 90% obserwacji to przypadki bez rozwodu. Oznacza to, że sieć mogłaby osiągnąć 90% dokładności (accuracy) po prostu zawsze przewidując brak rozwodu – bez nauczenia się czegokolwiek. Wyniki dla tego zadania należy interpretować ostrożnie.

Podział danych: 80% zbiór uczący (8 000 obserwacji), 20% zbiór testowy (2 000 obserwacji).

---

### Problem 2 – Regresja: Przewidywanie liczby lat od ślubu

Celem jest przewidzenie, ile lat minęło od zawarcia małżeństwa (**Years_Since_Marriage**), na podstawie tych samych 16 cech co powyżej.

Jest to **regresja** – sieć zwraca jedną wartość liczbową (ciągłą), a nie klasę.

Jako metrykę błędu stosujemy **MSE** (Mean Squared Error – średni błąd kwadratowy). Im niższe MSE, tym lepiej sieć przewiduje.

Podział danych: 80% zbiór uczący (8 000 obserwacji), 20% zbiór testowy (2 000 obserwacji).

---

### Opis implementacji sieci neuronowej

Sieć neuronowa została zaimplementowana od podstaw w języku Python przy użyciu wyłącznie biblioteki NumPy (bez gotowych frameworków jak TensorFlow czy Keras). Sieć obsługuje:
- dowolną liczbę warstw ukrytych i neuronów,
- funkcje aktywacji: ReLU, Sigmoid, Tanh, Leaky ReLU,
- metody inicjalizacji wag: Xavier, He, LeCun, losową (random),
- mini-batch stochastyczny gradient prosty (SGD),
- dwa tryby: klasyfikacja binarna (funkcja straty: BCE + Sigmoid na wyjściu) oraz regresja (MSE + liniowe wyjście).

Po wstępnym przetworzeniu danych zmienne liczbowe zostały znormalizowane (StandardScaler), a zmienne kategoryczne zakodowane metodą one-hot encoding, co dało łącznie 39 cech wejściowych.

Każdy eksperyment był powtarzany **3-krotnie** z różnymi zarodkami losowości (seed = 42, 43, 44), a wyniki uśrednione.

---

## 3. Analiza wpływu parametrów na skuteczność sieci

Poniżej przeanalizowano wpływ **8 parametrów** na wyniki sieci – oddzielnie dla zadania klasyfikacji i regresji. W tabelach podano wartości uśrednione z 3 powtórzeń.

> **Uwaga do klasyfikacji:** Accuracy ~90% we wszystkich konfiguracjach wynika z niezbalansowania danych – sieć nauczyła się przewidywać dominującą klasę (brak rozwodu). Różnice między konfiguracjami są minimalne. Bardziej miarodajną metryką byłaby balanced accuracy lub F1-score.

---

### Parametr 1 – Architektura sieci (liczba warstw i neuronów)

Testowano cztery architektury warstw ukrytych przy domyślnych pozostałych parametrach (learning rate = 0,01, epochs = 100, batch size = 32, aktywacja ReLU, inicjalizacja Xavier).

| Architektura | Klasyfikacja Train Acc | Klasyfikacja Test Acc | Regresja Train MSE | Regresja Test MSE |
|---|---|---|---|---|
| [8] (1 warstwa, 8 neuronów) | 0,8999 | 0,9000 | 190,32 | 204,33 |
| [16, 8] (2 warstwy) | 0,8999 | 0,9000 | 191,32 | 202,95 |
| [32, 16] (2 warstwy) | 0,8999 | 0,9000 | 182,75 | 214,09 |
| [32, 16, 8] (3 warstwy) | 0,8999 | 0,9000 | 197,03 | 199,82 |

**Wnioski:** Dla klasyfikacji brak różnic – efekt niezbalansowanych danych. Dla regresji najlepszy wynik testowy osiągnęła architektura [32, 16, 8] (MSE = 199,82). Sieć z warstwami [32, 16] wykazała lekkie przetrenowanie (niższy MSE na zbiorze uczącym, wyższy na testowym).

---

### Parametr 2 – Liczba epok

Epoka to jedno pełne przejście przez zbiór uczący. Testowano: 10, 50, 100, 200 epok.

| Liczba epok | Klasyfikacja Train Acc | Klasyfikacja Test Acc | Regresja Train MSE | Regresja Test MSE |
|---|---|---|---|---|
| 10 | 0,8999 | 0,9000 | 196,67 | 199,85 |
| 50 | 0,8999 | 0,9000 | 192,58 | 201,27 |
| 100 | 0,8999 | 0,9000 | 191,32 | 202,95 |
| 200 | 0,8999 | 0,9000 | 189,36 | 205,56 |

**Wnioski:** Dla klasyfikacji – brak wpływu. Dla regresji: więcej epok powoduje nieznaczne **przetrenowanie** – MSE na zbiorze uczącym maleje (sieć lepiej uczy się danych uczących), ale MSE na zbiorze testowym rośnie (gorzej generalizuje na nowe dane). Najlepszy wynik testowy uzyskano przy 10 epokach, choć różnice są niewielkie.

---

### Parametr 3 – Rozmiar batcha (batch size)

Batch size to liczba przykładów uczących przetwarzanych naraz przed aktualizacją wag. Testowano: 16, 32, 64, 128.

| Batch size | Klasyfikacja Train Acc | Klasyfikacja Test Acc | Regresja Train MSE | Regresja Test MSE |
|---|---|---|---|---|
| 16 | 0,8999 | 0,9000 | 195,90 | 200,87 |
| 32 | 0,8999 | 0,9000 | 191,32 | 202,95 |
| 64 | 0,8999 | 0,9000 | 193,79 | 202,66 |
| 128 | 0,8999 | 0,9000 | 192,53 | 205,28 |

**Wnioski:** Małe batche (16) dają najlepszy wynik testowy dla regresji (MSE = 200,87), ponieważ częstsze aktualizacje wag pozwalają sieci lepiej generalizować. Duże batche (128) pogorszyły wyniki testowe. Dla klasyfikacji brak widocznego efektu.

---

### Parametr 4 – Współczynnik uczenia (learning rate)

Learning rate określa, jak duże kroki wykonuje sieć podczas nauki. Zbyt mały – nauka wolna i może utknąć; zbyt duży – sieć staje się niestabilna. Testowano: 0,001, 0,01, 0,05, 0,1.

| Learning rate | Klasyfikacja Train Acc | Klasyfikacja Test Acc | Regresja Train MSE | Regresja Test MSE |
|---|---|---|---|---|
| 0,001 | 0,8999 | 0,9000 | 182,42 | 213,54 |
| 0,01 | 0,8999 | 0,9000 | 191,32 | 202,95 |
| 0,05 | 0,9004 | 0,8988 | 197,30 | 200,12 |
| 0,1 | 0,9019 | 0,9027 | 197,52 | 200,36 |

**Wnioski:** Dla regresji: przy małym learning rate (0,001) sieć zbytnio się przetrenowuje – bardzo niski MSE treningowy, ale wysoki testowy. Przy większych wartościach (0,05; 0,1) oba błędy są zbliżone i niskie. Dla klasyfikacji: przy learning rate = 0,05 i 0,1 widać pierwsze sygnały rzeczywistego uczenia się (train accuracy nieznacznie przekroczyła 0,9, co wcześniej się nie zdarzało), choć wyniki testowe są podobne.

---

### Parametr 5 – Funkcja aktywacji

Funkcja aktywacji decyduje, jak neuron „reaguje" na sygnał wejściowy. Testowano: ReLU, Sigmoid, Tanh, Leaky ReLU.

| Funkcja aktywacji | Klasyfikacja Train Acc | Klasyfikacja Test Acc | Regresja Train MSE | Regresja Test MSE |
|---|---|---|---|---|
| ReLU | 0,8999 | 0,9000 | 191,32 | 202,95 |
| Sigmoid | 0,8999 | 0,9000 | 175,98 | 219,63 |
| Tanh | 0,8999 | 0,9000 | 181,07 | 213,64 |
| Leaky ReLU | 0,8999 | 0,9000 | 185,22* | 210,23* |

*\*Leaky ReLU w 2 z 3 powtórzeń dla regresji dało wynik NaN (niestabilność numeryczna – eksplodujący gradient). Podano wynik z jedynego udanego powtórzenia.*

**Wnioski:** Dla klasyfikacji – brak różnic. Dla regresji: ReLU jest **najstabilniejszą i najlepiej generalizującą** funkcją aktywacji (najniższy MSE testowy). Sigmoid i Tanh pokazują silne przetrenowanie (niski MSE uczący, wysoki testowy). Leaky ReLU okazała się niestabilna numerycznie w tej konfiguracji.

---

### Parametr 6 – Metoda inicjalizacji wag

Inicjalizacja wag to sposób nadania sieci wartości początkowych przed uczeniem. Testowano: Xavier, He, Random (małe losowe), LeCun.

| Inicjalizacja | Klasyfikacja Train Acc | Klasyfikacja Test Acc | Regresja Train MSE | Regresja Test MSE |
|---|---|---|---|---|
| Xavier | 0,8999 | 0,9000 | 191,32 | 202,95 |
| He | 0,8999 | 0,9000 | 191,55 | 202,97 |
| Random | 0,8999 | 0,9000 | 191,93 | 206,39 |
| LeCun | 0,8999 | 0,9000 | 196,66 | 200,16 |

**Wnioski:** Xavier, He i LeCun dają podobne wyniki, gdyż wszystkie są metodami zaprojektowanymi do stabilnego uczenia. LeCun osiągnął najlepszy wynik testowy dla regresji (MSE = 200,16). Inicjalizacja Random (małe wartości bliskie zeru) dała nieznacznie gorsze wyniki – bardziej zróżnicowane między powtórzeniami.

---

### Parametr 7 – Liczba warstw ukrytych

Testowano sieci z 1, 2, 3 i 4 warstwami ukrytymi, przy stałej liczbie 32 neuronów w każdej warstwie.

| Liczba warstw | Klasyfikacja Train Acc | Klasyfikacja Test Acc | Regresja Train MSE | Regresja Test MSE |
|---|---|---|---|---|
| 1 × [32] | 0,8999 | 0,9000 | 168,75 | 229,83 |
| 2 × [32, 32] | 0,8999 | 0,9000 | 190,99 | 207,27 |
| 3 × [32, 32, 32] | 0,8999 | 0,9000 | 191,05 | 209,30 |
| 4 × [32, 32, 32, 32] | 0,8999 | 0,9000 | NaN | NaN |

**Wnioski:** Sieć z 1 warstwą wykazała silne przetrenowanie (duża różnica między MSE uczącym a testowym). Sieci z 2 i 3 warstwami osiągnęły podobne wyniki. Sieć z 4 warstwami zwróciła wartości NaN – sieć stała się zbyt głęboka i doszło do **eksplodującego gradientu** (wartości podczas uczenia wymknęły się spod kontroli). Optymalna głębokość to 2–3 warstwy ukryte.

---

### Parametr 8 – Wielkość zbioru testowego

Testowano różne podziały danych na zbiór uczący i testowy: 90%/10%, 80%/20%, 70%/30%, 60%/40%.

| Podział (test_size) | Klasyfikacja Train Acc | Klasyfikacja Test Acc | Regresja Train MSE | Regresja Test MSE |
|---|---|---|---|---|
| 10% testowy | 0,8999 | 0,9000 | 193,67 | 202,19 |
| 20% testowy | 0,8999 | 0,9000 | 191,32 | 202,95 |
| 30% testowy | 0,8999 | 0,9000 | 190,36 | 205,40 |
| 40% testowy | 0,8999 | 0,9000 | 190,69 | 205,60 |

**Wnioski:** Przy większym zbiorze testowym (mniej danych uczących) wyniki testowe nieznacznie się pogarszają. Najlepszy wynik testowy dla regresji uzyskano przy podziale 90/10. Różnice są jednak małe, co wskazuje, że przy 10 000 obserwacji wielkość próby nie jest krytycznym czynnikiem.

---

## 4. Podsumowanie i wnioski

### Zadanie klasyfikacji – przewidywanie rozwodu

Wyniki dla zadania klasyfikacji były rozczarowujące – accuracy utrzymywała się na poziomie ~90% niezależnie od konfiguracji sieci. Jest to klasyczny efekt **silnie niezbalansowanych danych**: w zbiorze około 90% obserwacji to przypadki bez rozwodu. Sieć nauczyła się przewidywać zawsze tę samą klasę (brak rozwodu), co daje wysoką dokładność bez faktycznego uczenia się wzorców prowadzących do rozwodu.

W takich przypadkach należałoby zastosować techniki wyrównywania klas (np. oversampling mniejszościowej klasy) lub używać innych metryk (np. F1-score, balanced accuracy). W niniejszym projekcie zrealizowano wymagania formalne – zbadano wpływ 8 parametrów, choć ich wpływ na wynik okazał się nieistotny ze względu na wskazany problem.

### Zadanie regresji – przewidywanie liczby lat od ślubu

Wyniki dla regresji były bardziej zróżnicowane i sensowne do analizy. Bazowy błąd MSE oscylował wokół 200 (RMSE ≈ 14,1 roku), co oznacza, że sieć myli się średnio o około 14 lat. Wynik ten jest umiarkowany i sugeruje, że cechy zawarte w zbiorze danych jedynie częściowo wyjaśniają liczbę lat od ślubu.

**Najważniejsze wnioski z regresji:**

1. **Najlepsza konfiguracja** pod względem błędu testowego: architektura [32, 16, 8], learning rate = 0,05–0,1, inicjalizacja LeCun, batch size = 16.

2. **Przetrenowanie** było widoczne przy: małym learning rate (0,001), funkcjach aktywacji Sigmoid i Tanh, głębokiej sieci z 1 warstwą o dużej liczbie neuronów.

3. **Niestabilność numeryczna** (wyniki NaN) wystąpiła przy: sieci z 4 warstwami ukrytymi (eksplodujący gradient) oraz przy Leaky ReLU z domyślnym learning rate (nadmierne wartości wag).

4. **Liczba epok** powyżej 50 pogarsza generalizację – sieć zaczyna „zapamiętywać" dane uczące zamiast uczyć się ogólnych wzorców.

5. **Funkcja ReLU** okazała się najbardziej stabilna i dała najlepsze wyniki testowe spośród czterech testowanych funkcji aktywacji.

6. **Metody inicjalizacji wag** Xavier, He i LeCun dają podobne wyniki – wszystkie są poprawne dla tej architektury. Inicjalizacja czysto losowa (small random) jest nieco gorsza i mniej przewidywalna.

7. **Podział danych** nie miał dużego wpływu na wyniki, co jest korzystną obserwacją – zbiór danych jest wystarczająco duży.

### Ogólna konkluzja

Implementacja sieci neuronowej od podstaw pozwoliła zaobserwować typowe zjawiska znane z teorii uczenia maszynowego: przetrenowanie, eksplodujący gradient oraz problem niezbalansowanych klas. Sieć sprawdziła się lepiej w zadaniu regresji niż klasyfikacji, gdzie dominujący wpływ miała nierównowaga klas. Do poprawy wyników klasyfikacji konieczne byłoby zastosowanie technik balansowania danych lub innych metryk oceny.
