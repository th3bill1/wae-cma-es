# WAE 2026L - Modyfikacja algorytmu CMA-ES

Projekt porownuje standardowy algorytm CMA-ES z wariantem, w ktorym sciezka
ewolucyjna kroku `p_sigma` jest inicjalizowana losowo.

Porownywane warianty:

- `zero`: standardowe `p_sigma(0) = 0`,
- `random`: zmodyfikowane `p_sigma(0) = v`, gdzie `v` jest losowym wektorem.

Eksperymenty sprawdzaja wplyw tej modyfikacji na zbieznosc, koncowa wartosc
funkcji celu, liczbe ewaluacji oraz zachowanie parametru `sigma` w czasie.

## Wymagania

Projekt jest przygotowany jako zestaw skryptow Python. Do uruchomienia potrzebne sa:

- Linux lub zgodne srodowisko shell,
- Python 3.11 lub nowszy,
- pakiety z `requirements.txt`.

Glowne biblioteki:

- NumPy,
- SciPy,
- Pandas,
- Matplotlib,
- tqdm.

## Instalacja na Linuxie

Z katalogu glownego repozytorium:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Odtworzenie wszystkich eksperymentow

Pelne odtworzenie wynikow od zera uruchamia wszystkie eksperymenty, analizy i
wykresy:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

Skrypt wykonuje kolejno:

1. `run_experiments.py` - generuje dane surowe,
2. `reproduce_all.py` - generuje podsumowania statystyczne i wykresy.

Pelny przebieg obejmuje 1800 uruchomien CMA-ES, wiec moze trwac dluzej w
zaleznosci od sprzetu (ok. 10 minut).

## Odtworzenie wykresow z istniejacych danych

Jesli pliki w `results/raw/` sa juz dostepne, same analizy i wykresy mozna
odtworzyc poleceniem:

```bash
python reproduce_all.py
```

## Konfiguracja eksperymentow

Glowne ustawienia znajduja sie w `run_experiments.py`:

- funkcje testowe: `sphere`, `rosenbrock`, `rastrigin`, `ackley`, `gaussian_noise`,
- wymiary: `2`, `10`, `30`,
- warianty `p_sigma`: `zero`, `random`,
- generatory PRNG: `pcg64`, `mt19937`,
- seedy: `1..30`,
- limit ewaluacji: `20000`,
- cel optymalizacji: `1e-8`.

Lista seedow i generatorow jest zapisana takze w `seeds/seeds.json`.

## Generatory PRNG

Projekt uzywa dwoch generatorow:

- `pcg64`: `numpy.random.default_rng(seed)`,
- `mt19937`: `numpy.random.MT19937(seed)` opakowany w `numpy.random.Generator`.

Generator probkowania populacji jest oddzielony od generatora uzywanego do
losowej inicjalizacji `p_sigma`. Seed dla `p_sigma` jest zapisywany w wynikach
jako `p_sigma_seed`.

## Struktura wynikow

Po uruchomieniu eksperymentow tworzone sa pliki:

- `results/raw/results.csv` - metryki koncowe kazdego uruchomienia,
- `results/raw/histories.json` - historie zbieznosci, sigmy i normy `p_sigma`,
- `results/summary/summary.csv` - statystyki opisowe,
- `results/summary/wilcoxon.csv` - wyniki testow Wilcoxona,
- `results/plots/` - wykresy zbieznosci, ECDF, boxploty, trajektorie sigmy i trajektorie sredniej populacji.

## Najwazniejsze pliki

- `src/cmaes.py` - implementacja CMA-ES i wariantow inicjalizacji `p_sigma`,
- `src/rng.py` - tworzenie generatorow PRNG,
- `src/benchmarks.py` - funkcje testowe,
- `src/experiment.py` - pojedynczy eksperyment,
- `src/analysis.py` - statystyki opisowe i test Wilcoxona,
- `src/plots.py` - generowanie wykresow,
- `run_experiments.py` - pelny zestaw eksperymentow,
- `reproduce_all.py` - odtwarzanie analiz i wykresow,
- `run_all_experiments.sh` - pelna reprodukcja od zera na Linuxie.
