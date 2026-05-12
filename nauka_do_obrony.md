# Nauka do obrony — Sztuczne Sieci Neuronowe

---

## Czym w ogóle jest sieć neuronowa?

Wyobraź sobie, że chcesz nauczyć kogoś rozpoznawać koty. Pokazujesz mu 1000 zdjęć kotów i 1000 zdjęć nie-kotów. Stopniowo sam się uczy, co jest charakterystyczne dla kota.

Sieć neuronowa robi to samo z liczbami. Dostajesz dane wejściowe (np. wiek, religia, wykształcenie małżonków), przechodzą przez "neurony", a na końcu sieć odpowiada: "to małżeństwo się rozpadnie" albo "będą razem 12 lat".

---

## Blok 1 — Jak wygląda sieć (warstwy)

```
Dane wejściowe (39 liczb)
        ↓
  Warstwa ukryta 1 (np. 16 neuronów)
        ↓
  Warstwa ukryta 2 (np. 8 neuronów)
        ↓
  Wyjście (1 liczba: prawdopodobieństwo rozwodu LUB liczba lat)
```

W kodzie wygląda to tak — definiujesz rozmiary każdej warstwy:

```python
layer_sizes = [39, 16, 8, 1]
# 39 = liczba cech wejściowych (po przetworzeniu danych)
# 16, 8 = warstwy ukryte (możesz mieć więcej lub mniej)
# 1 = jedno wyjście
```

---

## Blok 2 — Inicjalizacja wag (początek treningu)

Przed uczeniem każde połączenie między neuronami ma jakąś losową wartość — to **waga**. Wagi to właśnie to, czego sieć się "uczy" — na początku są losowe, potem się poprawiają.

```python
# Dla każdej pary sąsiednich warstw tworzymy macierz wag W i biasy b
for i in range(len(self.layer_sizes) - 1):
    n_in = self.layer_sizes[i]    # ile neuronów wchodzi
    n_out = self.layer_sizes[i+1] # ile neuronów wychodzi

    if self.weight_init == "xavier":
        limit = np.sqrt(6.0 / (n_in + n_out))
        W = np.random.uniform(-limit, limit, (n_in, n_out))
    elif self.weight_init == "he":
        W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
    elif self.weight_init == "lecun":
        W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
    elif self.weight_init == "random":
        W = np.random.randn(n_in, n_out) * 0.01
```

**Dlaczego różne metody inicjalizacji?**
Xavier jest dobry do sigmoid/tanh, He do ReLU. Złe wartości startowe mogą sprawić, że sieć nigdy się nie nauczy — albo wartości "wybuchają" do nieskończoności (to właśnie NaN w wynikach).

---

## Blok 3 — Forward pass (przejście do przodu)

To najważniejszy blok. Dane wejściowe "przepływają" przez sieć i na końcu dostajemy odpowiedź.

```python
def _forward(self, X):
    A = X  # A to aktualny sygnał — zaczyna jako dane wejściowe

    for i in range(len(self.weights)):
        W = self.weights[i]  # wagi między warstwą i a i+1
        b = self.biases[i]   # bias (przesunięcie)

        Z = np.dot(A, W) + b
        # np.dot = mnożenie macierzowe
        # Z = "surowy sygnał" neuronu, zanim zostanie przefiltrowany
```

**Co to jest Z?**
Każdy neuron sumuje wszystkie sygnały wejściowe pomnożone przez wagi, dodaje bias. To jak ważona średnia.

```python
        # Teraz stosujemy funkcję aktywacji
        if self.activation == "relu":
            A = np.maximum(0, Z)
            # ReLU: jeśli Z < 0, zwróć 0; jeśli Z > 0, zwróć Z
            # Czyli neuron "milczy" jeśli dostał słaby sygnał
```

**Dlaczego funkcja aktywacji?**
Bez niej cała sieć to po prostu jedno duże mnożenie — nie umiałaby uczyć się skomplikowanych wzorców. Aktywacja dodaje "nieliniowość".

```
ReLU:        Sigmoid:        Tanh:
  /           ___            _
 /           /             _/
___         /           ___/
```

Na wyjściu zamiast ReLU używamy:
```python
        if self.output_activation == "sigmoid":
            A = 1.0 / (1.0 + np.exp(-Z))
            # Sigmoid "ściska" wynik do przedziału [0, 1]
            # Idealny dla klasyfikacji (prawdopodobieństwo)

        elif self.output_activation == "linear":
            A = Z
            # Dla regresji — nie ograniczamy, zwracamy surową liczbę
```

---

## Blok 4 — Funkcja straty (jak mierzymy błąd)

Po forward pass wiemy co sieć odpowiedziała. Teraz liczymy jak bardzo się myliła.

**Dla klasyfikacji — Binary Cross-Entropy (BCE):**
```python
# Jeśli odpowiedź to 1 (rozwód), a sieć powiedziała 0.1 — duży błąd
# Jeśli odpowiedź to 1, a sieć powiedziała 0.9 — mały błąd
train_l = -np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))
```

**Dla regresji — MSE (Mean Squared Error):**
```python
# Jeśli prawda to 10 lat, a sieć powiedziała 15 — błąd = (10-15)² = 25
# Średnia z takich błędów to właśnie MSE
train_l = np.mean((y_train - train_preds[-1])**2)
```

---

## Blok 5 — Backward pass (uczenie się)

To serce algorytmu. Po obliczeniu błędu cofamy się przez sieć i korygujemy wagi — te które "zawiniły" najbardziej dostają większą korektę.

```python
def _backward(self, y_true, activations, zs):
    y_pred = activations[-1]  # co sieć odpowiedziała

    # Gradient — o ile błąd zmieni się gdy zmienimy wagi wyjściowe?
    if self.problem_type == "classification":
        dZ = (y_pred - y_true) / m
        # Jeśli sieć powiedziała 0.9 a prawda to 0 → dZ = +0.9/m (trzeba zmniejszyć)
        # Jeśli sieć powiedziała 0.1 a prawda to 1 → dZ = -0.9/m (trzeba zwiększyć)

    elif self.problem_type == "regression":
        dZ = 2.0 * (y_pred - y_true) / m
        # Pochodna MSE — im większy błąd, tym większa korekta
```

Potem cofamy się warstwa po warstwie:
```python
    for i in reversed(range(len(self.weights))):
        A_prev = activations[i]  # co dostała poprzednia warstwa

        dW = np.dot(A_prev.T, dZ)  # gradient wag = wejście × błąd wyjścia
        db = np.sum(dZ, axis=0)    # gradient biasów

        # Liczymy jak błąd płynął przez aktywację (pochodna)
        if self.activation == "relu":
            dZ = dA_prev * (Z_prev > 0).astype(float)
            # Gradient przez ReLU: 1 gdzie Z > 0, 0 gdzie Z ≤ 0
            # Jeśli neuron "milczał" (ReLU go wyciął), nie dostaje gradientu
```

---

## Blok 6 — Aktualizacja wag (gradient descent)

```python
for j in range(len(self.weights)):
    self.weights[j] -= self.learning_rate * dW_list[j]
    self.biases[j]  -= self.learning_rate * db_list[j]
```

**Learning rate** to jak duży krok robimy w kierunku poprawy.

```
Za mały (0.001):  ......................cel   (dojdziemy ale bardzo wolno, może utknąć)
Dobry (0.01):     --------cel               (sprawnie)
Za duży (0.5):    ↗↘↗↘↗ nigdy nie trafia   (skacze ponad cel)
```

To tłumaczy wyniki z eksperymentów — przy `learning_rate=0.001` sieć przetrenowała się,
przy `0.05–0.1` wyniki testowe były najlepsze.

---

## Blok 7 — Pętla uczenia (cały trening)

```python
for epoch in range(self.epochs):       # ile razy przejdziemy przez CAŁE dane
    # Losowe mieszanie danych (żeby sieć nie uczyła się kolejności)
    indices = np.arange(m)
    np.random.shuffle(indices)

    for i in range(0, m, self.batch_size):   # mini-batche
        X_batch = X_shuffled[i:i+self.batch_size]
        y_batch = y_shuffled[i:i+self.batch_size]

        activations, zs = self._forward(X_batch)   # 1. oblicz odpowiedź
        dW_list, db_list = self._backward(...)      # 2. oblicz gradienty
        self.weights[j] -= self.learning_rate * dW_list[j]  # 3. popraw wagi
```

**Epoch** = jedno pełne przejście przez dane. Przy 100 epokach sieć widzi każdy przykład 100 razy.

**Batch** = kawałek danych. Zamiast liczyć gradient na 8000 przykładach naraz, liczymy go na 32 i od razu poprawiamy wagi. Szybciej i lepiej generalizuje.

---

## Pytania na obronie — gotowe odpowiedzi

**P: Co to jest forward pass?**
> To przepływ danych przez sieć — od wejścia do wyjścia. Każda warstwa mnoży dane przez wagi i stosuje funkcję aktywacji. Na końcu dostajemy przewidywaną wartość.

**P: Co to jest backward pass?**
> To algorytm wstecznej propagacji błędu. Po obliczeniu jak bardzo sieć się myliła, cofamy się przez sieć i poprawiamy wagi — im bardziej dana waga "zawiniła", tym bardziej ją korygujemy.

**P: Dlaczego klasyfikacja dała ~90% dla wszystkich konfiguracji?**
> Dane są silnie niezbalansowane — 90% przypadków to brak rozwodu. Sieć nauczyła się przewidywać zawsze klasę większościową, co daje 90% dokładności bez faktycznego uczenia się. To znany problem w uczeniu maszynowym, rozwiązywany np. przez oversampling lub inne metryki jak F1-score.

**P: Co to są NaN w wynikach?**
> To eksplodujący gradient — przy zbyt głębokiej sieci (4 warstwy) lub przy niestabilnej funkcji aktywacji (Leaky ReLU) wartości wag podczas uczenia rosły do nieskończoności. Wynik stał się liczbowo niemożliwy do reprezentacji.

**P: Dlaczego Xavier/He, nie zwykłe losowe?**
> Specjalne metody inicjalizacji zapewniają, że sygnał ma podobną "siłę" przez całą sieć. Zbyt małe wagi → sygnał zanika; zbyt duże → eksploduje. Xavier i He dobierają skalę inicjalizacji do liczby neuronów w warstwie.

**P: Co to jest przetrenowanie (overfitting)?**
> Sieć "zapamiętała" dane uczące zamiast nauczyć się ogólnych wzorców. Objawia się tym, że błąd na zbiorze uczącym jest niski, ale na testowym wysoki. Widać to np. przy 200 epokach — train MSE spada, ale test MSE rośnie.

**P: Po co dzielimy dane na uczące i testowe?**
> Zbiór testowy to dane, których sieć nigdy nie widziała podczas uczenia. Dzięki temu możemy sprawdzić, czy sieć naprawdę się nauczyła, czy tylko zapamiętała przykłady. Gdybyśmy oceniali na danych uczących, zawsze mielibyśmy fałszywie dobre wyniki.

**P: Dlaczego używasz MSE a nie czegoś innego dla regresji?**
> MSE (średni błąd kwadratowy) karze mocno duże błędy — błąd o 10 lat daje 100, a błąd o 1 rok daje 1. Dzięki temu sieć stara się unikać dużych pomyłek. Łatwo też liczyć jego pochodną, co przyspiesza uczenie.

**P: Co to jest batch size i dlaczego ma znaczenie?**
> Batch to porcja danych przetwarzana naraz przed aktualizacją wag. Mały batch (16) = częste aktualizacje, więcej losowości, lepsza generalizacja ale wolniej. Duży batch (128) = rzadsze aktualizacje, stabilniejsze uczenie ale może gorzej generalizować. W eksperymentach najlepszy wynik testowy dał batch=16.
