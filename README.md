# WAE 2026L — Modyfikacja algorytmu CMA-ES

Projekt realizowany w ramach przedmiotu **Wstęp do Algorytmów Ewolucyjnych**.

## Temat projektu

Celem projektu jest zbadanie wpływu modyfikacji algorytmu **CMA-ES**  
(*Covariance Matrix Adaptation Evolution Strategy*) polegającej na zmianie sposobu inicjalizacji ścieżki adaptacji kroku `p_sigma`.

Porównywane są dwa warianty:

1. **Wariant standardowy** — `p_sigma(0) = 0`
2. **Wariant zmodyfikowany** — `p_sigma(0) = v`, gdzie `v` jest losowym wektorem

Badamy, czy losowa inicjalizacja `p_sigma` wpływa na:

- szybkość zbieżności,
- liczbę ewaluacji funkcji celu,
- jakość końcowego rozwiązania,
- stabilność działania algorytmu.

## Wykorzystywane biblioteki

- **NumPy** — operacje numeryczne i macierzowe
- **SciPy** — testy statystyczne i wybrane narzędzia obliczeniowe
- **Matplotlib** — wizualizacja wyników
- **Pandas** — zapis i analiza wyników eksperymentów
- **tqdm** — pasek postępu
- **cma** — biblioteka referencyjna do walidacji poprawności implementacji

## Do zrobienia:

[x] pobawić się z parametrami
- zrobić coś żeby zauważalna była różnica pomiędzy bazową wersją i modyfikacją XD
- uruchomić to na większych instancjach
- sprawdzić czy nie dało by się jeszcze jakoś dodatkowo porównywać tych wyników
[x] poprawić wizualnie obecną wizualizację trajektorii
- zastanowić sie czy na pewno chcemy uzywac jupytera, moze jakos prosciej to opisac?
- zaczac dokumentacje koncowa (bo pewnie wyjdzie wtedy cos co chcielibysmy zmienic w kodzie)