# Projekt – Sztuczne Sieci Neuronowe i Uczenie Maszynowe

## 1. Wstęp i opis badanych problemów

W projekcie wykorzystano zbiór danych dotyczący małżeństw w Indiach (marriage_data_india.csv), zawierający 10 000 obserwacji i 16 cech opisujących m.in. typ małżeństwa, wiek w chwili zawarcia związku, poziom wykształcenia, religię, akceptację rodziców, poziom dochodów, czy małżonek pracuje, a także informację o posagu.

Zbadano dwa problemy:

**Problem 1 – Klasyfikacja binarna:** przewidywanie, czy dane małżeństwo zakończyło się rozwodem. Zmienna docelowa to **Divorce_Status** (wartości: Yes → 1, No → 0). Dane są silnie niezbalansowane – około 90% przypadków to małżeństwa bez rozwodu. Oznacza to, że sieć mogłaby osiągnąć 90% dokładności, po prostu zawsze przewidując brak rozwodu.

**Problem 2 – Regresja:** przewidywanie, ile lat minęło od zawarcia małżeństwa. Zmienna docelowa to **Years_Since_Marriage**. Jako metrykę błędu zastosowano MSE (Mean Squared Error). Im niższe MSE, tym lepiej sieć przewiduje.

Dane podzielono na zbiór uczący (80%) i testowy (20%). Po preprocessingu (normalizacja zmiennych liczbowych, one-hot encoding kategorycznych) uzyskano 39 cech wejściowych.

---

## 2. Przegląd literatury

Przewidywanie wyników małżeńskich za pomocą uczenia maszynowego jest tematem obecnym w literaturze naukowej. Gottman i Levenson (1992) wykazali, że pewne wzorce zachowań małżeńskich pozwalają przewidywać rozwód z dużą skutecznością, co dało podstawy do stosowania modeli predykcyjnych w tej dziedzinie. Hajikhani i in. (2019) zastosowali różne metody uczenia maszynowego, w tym drzewa decyzyjne i SVM, do przewidywania rozwodów na podstawie danych ankietowych, osiągając dokładność powyżej 90%. Wykazali jednak, że wysoka dokładność może wynikać z niezbalansowania klas, co jest tym samym problemem, z którym zmierzono się w tym projekcie.

W zakresie przewidywania czasu trwania małżeństwa badania są mniej liczne. Analogiczne problemy regresyjne, jak przewidywanie wieku lub czasu trwania zjawisk na podstawie cech demograficznych, są rozwiązywane przy użyciu Random Forest i Gradient Boosting, które regularnie osiągają lepsze wyniki niż sieci neuronowe na danych tabelarycznych. Bhattacharya i in. (2021) w pracy dotyczącej przewidywania stabilności związków zauważyli, że zmienne ekonomiczne (poziom dochodów) i kulturowe (typ małżeństwa, akceptacja rodziców) mają największą moc predykcyjną – co jest spójne z charakterystyką używanego tu zbioru danych.

---

## 3. Sztuczne Sieci Neuronowe

### 3.1. Opis implementacji

Sieć neuronowa została zaimplementowana od podstaw w języku Python, wyłącznie przy użyciu biblioteki NumPy (bez TensorFlow ani Keras). Sieć obsługuje:
- dowolną liczbę warstw ukrytych i neuronów,
- funkcje aktywacji: ReLU, Sigmoid, Tanh, Leaky ReLU,
- metody inicjalizacji wag: Xavier, He, LeCun, losową (random),
- mini-batch SGD (stochastyczny gradient prosty),
- klasyfikację binarną (funkcja straty BCE, wyjście Sigmoid) oraz regresję (MSE, wyjście liniowe).

Każdy eksperyment był powtarzany **3 razy** z różnymi seedami (42, 43, 44), a wyniki w tabelach to wartości uśrednione. Domyślna konfiguracja bazowa: architektura [16, 8], learning rate = 0,01, epochs = 100, batch size = 32, aktywacja ReLU, inicjalizacja Xavier.

---

### 3.2. Analiza wpływu parametrów

> **Uwaga:** Accuracy dla klasyfikacji wynosi ~90% w prawie wszystkich konfiguracjach. Wynika to z niezbalansowania danych – sieć nauczyła się przewidywać zawsze klasę dominującą (brak rozwodu). Różnice między konfiguracjami są lepiej widoczne w zadaniu regresji.

---

#### 3.2.1. Architektura sieci (liczba neuronów w warstwach ukrytych)

Sprawdzono cztery różne architektury warstw ukrytych przy pozostałych parametrach domyślnych.

| Architektura | Klas. Train Acc | Klas. Test Acc | Reg. Train MSE | Reg. Test MSE |
|---|---:|---:|---:|---:|
| [8] – 1 warstwa | 0,8999 | 0,9000 | 190,32 | 204,33 |
| [16, 8] – 2 warstwy | 0,8999 | 0,9000 | 191,32 | 202,95 |
| [32, 16] – 2 warstwy | 0,8999 | 0,9000 | 182,75 | 214,09 |
| [32, 16, 8] – 3 warstwy | 0,8999 | 0,9000 | 197,03 | **199,82** |

**Wnioski:** Dla klasyfikacji brak widocznych różnic. Dla regresji najlepszy wynik testowy uzyskała architektura [32, 16, 8] (MSE = 199,82). Architektura [32, 16] wykazała przetrenowanie – niski MSE uczący (182,75), ale wysoki testowy (214,09).

---

#### 3.2.2. Liczba epok

Sprawdzono cztery wartości liczby epok przy domyślnych pozostałych parametrach.

| Liczba epok | Klas. Train Acc | Klas. Test Acc | Reg. Train MSE | Reg. Test MSE |
|---|---:|---:|---:|---:|
| 10 | 0,8999 | 0,9000 | 196,67 | **199,85** |
| 50 | 0,8999 | 0,9000 | 192,58 | 201,27 |
| 100 | 0,8999 | 0,9000 | 191,32 | 202,95 |
| 200 | 0,8999 | 0,9000 | 189,36 | 205,56 |

**Wnioski:** Dla klasyfikacji brak wpływu. Dla regresji widać klasyczny efekt przetrenowania – im więcej epok, tym niższy MSE uczący, ale wyższy testowy. Najlepszy wynik testowy osiągnięto przy zaledwie 10 epokach (199,85), co sugeruje, że sieć szybko „zapamiętuje" dane zamiast generalizować.

---

#### 3.2.3. Rozmiar batcha (batch size)

Batch size to liczba przykładów przetwarzanych naraz przed aktualizacją wag. Sprawdzono wartości: 16, 32, 64, 128.

| Batch size | Klas. Train Acc | Klas. Test Acc | Reg. Train MSE | Reg. Test MSE |
|---|---:|---:|---:|---:|
| 16 | 0,8999 | 0,9000 | 195,90 | **200,87** |
| 32 | 0,8999 | 0,9000 | 191,32 | 202,95 |
| 64 | 0,8999 | 0,9000 | 193,79 | 202,66 |
| 128 | 0,8999 | 0,9000 | 192,53 | 205,28 |

**Wnioski:** Mały batch (16) dał najlepszy wynik testowy dla regresji. Częstsze aktualizacje wag przy małym batchu pomagają sieci lepiej generalizować. Duży batch (128) pogorszył wyniki testowe. Dla klasyfikacji brak efektu.

---

#### 3.2.4. Współczynnik uczenia (learning rate)

Learning rate określa, jak duże kroki wykonuje sieć podczas nauki. Testowano: 0,001; 0,01; 0,05; 0,1.

| Learning rate | Klas. Train Acc | Klas. Test Acc | Reg. Train MSE | Reg. Test MSE |
|---|---:|---:|---:|---:|
| 0,001 | 0,8999 | 0,9000 | 182,42 | 213,54 |
| 0,01 | 0,8999 | 0,9000 | 191,32 | 202,95 |
| 0,05 | 0,9004 | 0,8988 | 197,30 | **200,12** |
| 0,1 | 0,9019 | 0,9027 | 197,52 | 200,36 |

**Wnioski:** Przy małym learning rate (0,001) sieć przetrenowuje się – bardzo niski MSE uczący (182,42), ale wysoki testowy (213,54). Przy większych wartościach (0,05; 0,1) oba błędy są zbliżone i niskie. Dla klasyfikacji przy lr=0,05 i lr=0,1 po raz pierwszy pojawiają się drobne różnice w accuracy, co może świadczyć o początku rzeczywistego uczenia się.

---

#### 3.2.5. Funkcja aktywacji

Sprawdzono cztery funkcje aktywacji w warstwach ukrytych: ReLU, Sigmoid, Tanh, Leaky ReLU.

| Funkcja aktywacji | Klas. Train Acc | Klas. Test Acc | Reg. Train MSE | Reg. Test MSE |
|---|---:|---:|---:|---:|
| ReLU | 0,8999 | 0,9000 | 191,32 | **202,95** |
| Sigmoid | 0,8999 | 0,9000 | 175,98 | 219,63 |
| Tanh | 0,8999 | 0,9000 | 181,07 | 213,64 |
| Leaky ReLU | 0,8999 | 0,9000 | 185,22* | 210,23* |

*\*Leaky ReLU w 2 z 3 powtórzeń zwróciła NaN (niestabilność numeryczna – eksplodujący gradient). Podano wynik z jedynego udanego powtórzenia.*

**Wnioski:** Dla klasyfikacji brak różnic. Dla regresji ReLU okazała się najstabilniejsza i dała najlepszy wynik testowy. Sigmoid i Tanh wykazują silne przetrenowanie. Leaky ReLU była niestabilna numerycznie przy domyślnym learning rate.

---

#### 3.2.6. Metoda inicjalizacji wag

Sprawdzono cztery metody nadawania sieci wartości początkowych przed uczeniem.

| Inicjalizacja | Klas. Train Acc | Klas. Test Acc | Reg. Train MSE | Reg. Test MSE |
|---|---:|---:|---:|---:|
| Xavier | 0,8999 | 0,9000 | 191,32 | 202,95 |
| He | 0,8999 | 0,9000 | 191,55 | 202,97 |
| Random | 0,8999 | 0,9000 | 191,93 | 206,39 |
| LeCun | 0,8999 | 0,9000 | 196,66 | **200,16** |

**Wnioski:** Xavier, He i LeCun dają podobne wyniki, bo wszystkie są metodami zaprojektowanymi do stabilnego uczenia. LeCun osiągnął najlepszy wynik testowy dla regresji (MSE = 200,16). Inicjalizacja czysto losowa (Random) dała nieco gorsze wyniki.

---

#### 3.2.7. Liczba warstw ukrytych (przy stałej liczbie neuronów)

Sprawdzono sieci z 1, 2, 3 i 4 warstwami ukrytymi, każda z 32 neuronami.

| Konfiguracja | Klas. Train Acc | Klas. Test Acc | Reg. Train MSE | Reg. Test MSE |
|---|---:|---:|---:|---:|
| 1 × [32] | 0,8999 | 0,9000 | 168,75 | 229,83 |
| 2 × [32, 32] | 0,8999 | 0,9000 | 190,99 | 207,27 |
| 3 × [32, 32, 32] | 0,8999 | 0,9000 | 191,05 | 209,30 |
| 4 × [32, 32, 32, 32] | 0,8999 | 0,9000 | NaN | NaN |

**Wnioski:** Jedna szeroka warstwa wykazała silne przetrenowanie (MSE uczący 168,75, testowy 229,83). Dwie i trzy warstwy dały podobne wyniki. Sieć z 4 warstwami zwróciła NaN – doszło do eksplodującego gradientu. Optymalna głębokość to 2–3 warstwy ukryte.

---

#### 3.2.8. Wielkość zbioru testowego

Sprawdzono różne podziały danych na zbiór uczący i testowy.

| Podział (test_size) | Klas. Train Acc | Klas. Test Acc | Reg. Train MSE | Reg. Test MSE |
|---|---:|---:|---:|---:|
| 10% | 0,8999 | 0,9000 | 193,67 | **202,19** |
| 20% | 0,8999 | 0,9000 | 191,32 | 202,95 |
| 30% | 0,8999 | 0,9000 | 190,36 | 205,40 |
| 40% | 0,8999 | 0,9000 | 190,69 | 205,60 |

**Wnioski:** Przy większym zbiorze testowym (mniejszym uczącym) wyniki testowe nieznacznie się pogarszają. Różnice są jednak małe – przy 10 000 obserwacjach wielkość próby nie jest krytycznym czynnikiem.

---

### 3.3. Podsumowanie SSN

**Klasyfikacja:** Wyniki były rozczarowujące – accuracy na poziomie ~90% niezależnie od konfiguracji. To klasyczny efekt silnie niezbalansowanych danych. Sieć nauczyła się przewidywać zawsze klasę dominującą (brak rozwodu), co daje wysoką dokładność bez faktycznego uczenia się wzorców prowadzących do rozwodu. Do poprawy wyników potrzebne byłoby balansowanie klas lub użycie metryk takich jak F1-score czy balanced accuracy.

**Regresja:** Wyniki były bardziej zróżnicowane i sensowne do analizy. Bazowy błąd MSE oscylował wokół 200 (RMSE ≈ 14,1 roku). Sieć myli się więc średnio o około 14 lat przy przewidywaniu czasu trwania małżeństwa. Najlepsze konfiguracje to: architektura [32, 16, 8], learning rate ≥ 0,05, inicjalizacja LeCun, batch size = 16. Widoczne były typowe zjawiska uczenia maszynowego: przetrenowanie (przy małym lr, funkcjach Sigmoid/Tanh, dużej liczbie epok) oraz eksplodujący gradient (4 warstwy, Leaky ReLU).

---

## 4. Uczenie Maszynowe

### 4.1. Opis wybranych metod

#### 4.1.1. Drzewo decyzyjne (Decision Tree)

Drzewo decyzyjne to model, który uczy się serii reguł „jeśli–to", dzieląc dane na coraz mniejsze grupy. Każdy węzeł drzewa sprawdza jeden warunek (np. czy wiek w chwili ślubu < 25), a liście drzewa zawierają przewidywane wartości. Model jest prosty do interpretacji, ale łatwo się przetrenowuje – szczególnie gdy drzewo jest bardzo głębokie i dopasowuje się do każdej obserwacji w zbiorze uczącym.

#### 4.1.2. Las losowy (Random Forest)

Las losowy to zbiór wielu drzew decyzyjnych, z których każde jest uczone na losowo wybranym fragmencie danych i losowym podzbiorze cech. Wynik końcowy to średnia z przewidywań wszystkich drzew (dla regresji) lub głosowanie większościowe (dla klasyfikacji). Dzięki temu model jest bardziej stabilny i mniej podatny na przetrenowanie niż pojedyncze drzewo. Jest jedną z najczęściej stosowanych metod na danych tabelarycznych.

#### 4.1.3. K najbliższych sąsiadów (KNN)

KNN przewiduje wartość dla nowej obserwacji na podstawie k najbliższych punktów ze zbioru uczącego. Odległość między punktami liczy się najczęściej metryką euklidesową lub manhattan. Algorytm jest prosty, ale wolny przy dużych zbiorach danych, bo musi porównać nowy punkt ze wszystkimi przykładami uczącymi. Ważnym parametrem jest liczba sąsiadów k – zbyt mała powoduje przetrenowanie, zbyt duża – uśrednianie i utratę dokładności.

#### 4.1.4. Gradient Boosting

Gradient Boosting buduje model sekwencyjnie: każde kolejne drzewo poprawia błędy poprzedniego. W odróżnieniu od Random Forest, drzewa nie są niezależne – każde skupia się na obserwacjach, które poprzednie drzewa przewidziały źle. Daje to zazwyczaj lepsze wyniki niż Random Forest, ale jest wolniejsze i bardziej podatne na przetrenowanie przy zbyt dużym learning rate.

---

### 4.2. Analiza wybranych parametrów

Eksperymenty wykonano przy użyciu biblioteki scikit-learn. Każdy eksperyment powtarzano 3 razy z różnymi seedami. Dla klasyfikacji jako metrykę wybrano **balanced accuracy** (dokładność ważona klasami) zamiast zwykłej accuracy, bo zwykła accuracy jest myląca przy niezbalansowanych danych. Dla regresji zastosowano **RMSE** (pierwiastek z MSE).

---

#### 4.2.1. Drzewo decyzyjne – parametr max_depth

Parametr `max_depth` kontroluje maksymalną głębokość drzewa. Większa głębokość oznacza bardziej szczegółowy model, ale też większe ryzyko przetrenowania.

**Klasyfikacja (Divorce_Status):**

| max_depth | Train Balanced Acc | Test Balanced Acc |
|---:|---:|---:|
| 3 | 0,5000 | 0,5000 |
| 5 | 0,5031 | 0,5000 |
| 10 | 0,6034 | 0,5029 |
| 20 | 0,9668 | 0,4933 |

**Regresja (Years_Since_Marriage):**

| max_depth | Train RMSE | Test RMSE |
|---:|---:|---:|
| 3 | 14,00 | **14,15** |
| 5 | 13,89 | 14,29 |
| 10 | 12,25 | 15,82 |
| 20 | 2,41 | 20,34 |

**Wnioski:** Dla klasyfikacji głębokie drzewo (max_depth=20) osiąga prawie idealną dokładność na zbiorze uczącym (0,97), ale na testowym spada poniżej poziomu losowego (0,49). To skrajne przetrenowanie. Dla regresji najlepszy wynik testowy (RMSE = 14,15) uzyskało płytkie drzewo (max_depth=3). Głębsze drzewa przetrenowują się – drzewo z max_depth=20 ma RMSE=2,41 na uczącym, ale 20,34 na testowym.

---

#### 4.2.2. Las losowy – parametr n_estimators

Parametr `n_estimators` to liczba drzew w lesie. Więcej drzew zazwyczaj daje lepsze i stabilniejsze wyniki, ale wydłuża czas uczenia.

**Klasyfikacja (Divorce_Status):**

| n_estimators | Train Balanced Acc | Test Balanced Acc |
|---:|---:|---:|
| 50 | 0,5065 | 0,5000 |
| 100 | 0,5044 | 0,5000 |
| 200 | 0,5037 | 0,5000 |
| 300 | 0,5037 | 0,5000 |

**Regresja (Years_Since_Marriage):**

| n_estimators | Train RMSE | Test RMSE |
|---:|---:|---:|
| 50 | 11,62 | 14,25 |
| 100 | 11,62 | 14,23 |
| 200 | 11,60 | 14,23 |
| 300 | 11,60 | **14,22** |

**Wnioski:** Dla klasyfikacji balanced accuracy jest stale na poziomie 0,5 – las losowy nie radzi sobie z niezbalansowanymi klasami. Dla regresji wyniki poprawiają się wraz z liczbą drzew, ale różnice między 100, 200 i 300 drzewami są bardzo małe. Najlepszy wynik testowy (14,22) uzyskano przy 300 drzewach.

---

#### 4.2.3. KNN – parametr n_neighbors

Parametr `n_neighbors` to liczba sąsiadów branych pod uwagę. Mała liczba powoduje przetrenowanie, duża – nadmierne uśrednianie.

**Klasyfikacja (Divorce_Status):**

| n_neighbors | Train Balanced Acc | Test Balanced Acc |
|---:|---:|---:|
| 3 | 0,5953 | 0,5033 |
| 5 | 0,5247 | 0,5031 |
| 11 | 0,5006 | 0,4997 |
| 21 | 0,5000 | 0,5000 |

**Regresja (Years_Since_Marriage):**

| n_neighbors | Train RMSE | Test RMSE |
|---:|---:|---:|
| 3 | 11,43 | 16,34 |
| 5 | 12,60 | 15,37 |
| 11 | 13,48 | 14,74 |
| 21 | 13,74 | **14,52** |

**Wnioski:** Dla klasyfikacji KNN nie pomaga przy niezbalansowanych danych. Dla regresji widać klasyczną zależność: mała liczba sąsiadów (k=3) przetrenowuje się – dobry wynik uczący (11,43), zły testowy (16,34). Wraz ze wzrostem k wynik testowy się poprawia. Najlepszy wynik przy k=21 (RMSE = 14,52).

---

#### 4.2.4. Gradient Boosting – parametr learning_rate

Parametr `learning_rate` kontroluje, jak duże poprawki wnosi każde kolejne drzewo. Zbyt duży może powodować niestabilność, zbyt mały – wolne uczenie.

**Klasyfikacja (Divorce_Status):**

| learning_rate | Train Balanced Acc | Test Balanced Acc |
|---:|---:|---:|
| 0,01 | 0,5000 | 0,5000 |
| 0,05 | 0,5000 | 0,5000 |
| 0,10 | 0,5000 | 0,5000 |
| 0,20 | 0,5006 | 0,5000 |

**Regresja (Years_Since_Marriage):**

| learning_rate | Train RMSE | Test RMSE |
|---:|---:|---:|
| 0,01 | 13,98 | **14,14** |
| 0,05 | 13,86 | 14,17 |
| 0,10 | 13,75 | 14,21 |
| 0,20 | 13,57 | 14,31 |

**Wnioski:** Dla klasyfikacji Gradient Boosting nie daje niczego lepszego – ten sam problem niezbalansowanych danych. Dla regresji najlepszy wynik testowy uzyskano przy learning_rate = 0,01 (RMSE = 14,14). Przy wyższych wartościach uczący RMSE maleje (model lepiej dopasowuje dane uczące), ale testowy rośnie – zaczyna się przetrenowanie.

---

### 4.3. Porównanie metod uczenia maszynowego

Porównanie wykonano przy wynikach testowych z każdej metody (przy jej najlepszej sprawdzonej wartości parametru).

**Klasyfikacja (Divorce_Status) – balanced accuracy:**

| Metoda | Najlepszy parametr | Test Balanced Acc |
|---|---|---:|
| Drzewo decyzyjne | max_depth=10 | 0,5029 |
| Las losowy | n_estimators=50 | 0,5000 |
| KNN | n_neighbors=3 | 0,5033 |
| Gradient Boosting | learning_rate=0,20 | 0,5000 |

Wszystkie metody osiągają balanced accuracy bliskie 0,50 – czyli poziom losowego zgadywania. Jest to bezpośrednia konsekwencja niezbalansowanych klas. Żadna z metod nie nauczyła się rzeczywiście wykrywać przypadków rozwodu.

**Regresja (Years_Since_Marriage) – RMSE:**

| Metoda | Najlepszy parametr | Train RMSE | Test RMSE |
|---|---|---:|---:|
| Drzewo decyzyjne | max_depth=3 | 14,00 | **14,15** |
| Gradient Boosting | learning_rate=0,01 | 13,98 | 14,14 |
| Las losowy | n_estimators=300 | 11,60 | 14,22 |
| KNN | n_neighbors=21 | 13,74 | 14,52 |
| SSN | [32,16,8], lr=0,05 | ~14,0 | ~14,1 |

Dla regresji wyniki są zbliżone: RMSE testowy mieści się w przedziale 14,1–14,5 roku. Najlepszy wynik uzyskały Gradient Boosting i Drzewo decyzyjne (RMSE ≈ 14,14–14,15). Las losowy jest bardzo stabilny, ale podobny poziomem do pozostałych. KNN wypada najgorzej z testowym RMSE = 14,52.

---

## 5. Podsumowanie i wnioski końcowe

### Zadanie klasyfikacji

Wyniki dla klasyfikacji były rozczarowujące niezależnie od zastosowanej metody – zarówno SSN, jak i wszystkie metody uczenia maszynowego osiągnęły balanced accuracy na poziomie losowym (~0,50). Przyczyną jest silne niezbalansowanie klas: ~90% obserwacji to małżeństwa bez rozwodu. Modele nauczyły się po prostu zawsze przewidywać brak rozwodu.

Aby poprawić wyniki, należałoby zastosować:
- oversampling klasy mniejszościowej (np. SMOTE),
- inne funkcje straty uwzględniające nierównowagę klas,
- balanced accuracy lub F1-score jako kryterium optymalizacji.

### Zadanie regresji

Wyniki dla regresji były sensowne i zbliżone we wszystkich metodach: RMSE ≈ 14 lat, co oznacza, że modele mylą się średnio o około 14 lat przy przewidywaniu czasu trwania małżeństwa. Biorąc pod uwagę, że zmienna Years_Since_Marriage ma szeroki zakres, jest to wynik umiarkowany – cechy zawarte w zbiorze danych tylko częściowo wyjaśniają liczbę lat od ślubu.

Najważniejsze obserwacje:
1. **Drzewo decyzyjne** sprawuje się najlepiej przy małej głębokości (max_depth=3). Głębsze drzewa się przetrenowują.
2. **Las losowy** jest najbardziej stabilny – małe różnice między wynikiem uczącym a testowym.
3. **Gradient Boosting** uzyskał najlepszy wynik testowy (RMSE = 14,14) przy małym learning_rate = 0,01.
4. **KNN** jest wrażliwy na liczbę sąsiadów i wypada nieco gorzej niż metody drzewiaste.
5. **SSN** osiąga porównywalny wynik do najlepszych metod klasycznych, co pokazuje, że dla danych tabelarycznych prostsze metody radzą sobie równie dobrze.

Ogólny wniosek: dla tego zbioru danych i tych problemów metody drzewiaste (zwłaszcza Gradient Boosting i Las losowy) działają co najmniej tak samo dobrze jak sieć neuronowa, a często lepiej – bo są mniej podatne na przetrenowanie i nie wymagają precyzyjnego doboru tak wielu hiperparametrów.
