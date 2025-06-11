# Dice poker ai

## Jan Gasztold, Uniwersytet Gdański, Wydział Matematyki Fizyki i Informatyki, Informatyka praktyczna rok 2, Data: 11.06.2025

## Spis treści

1. [Spis treści](#spis-treści)
2. [Wprowadzenie](#wprowadzenie)
3. [Etapy realizacji](#etapy-realizacji)
3. [Etap 1](#etap-1---przygotowanie-do-projektu)
4. [Etap 2](#etap-2)
5. [Etap 3](#etap-3)
6. [Etap 4](#etap-4)
10. [Tesotowanie](#)
11. [Funkcjonalności](#)
12. [Problemy i wyzwania](#)
13. [Podsumowanie](#)
14. [Załączniki](#załączniki)

## Wprowadzenie
### Opis projektu
...
### Cel projektu
Celem projektu jest stworzenie programu zdolnego do klasyfikacji kości do gry na podstawie obrazu, a następnie oszacowania opłacalności gry. Program powinien mieć zdolność podejmowania decyzji o ryzyku w zależności od warunków otoczenia.

## Etapy realizacji
### Etap 1 - przygotowanie do projektu
Prace rozpocząłem od wyboru technologii, w której zrealizuję projekt. Zdecydowałem się na Pythona ze względu na szeroki ekosystem bibliotek oraz moje doświadczenie z tą technologią. Projekt realizowałem w środowisku WSL2 na systemie Windows 11, w wersji Pythona 3.10.12. Główną biblioteką używaną do budowy modelu był Keras.

### Etap 2 - przygotowanie środowiska
Skorzystałem z ogólnodostępnych datasetów zdjęć kości do gry. Zebrałem około 2000 zdjęć przedstawiających różne układy kostek z różnych perspektyw [zobacz załączniki](#załączniki). Następnie przygotowałem strukturę projektu oraz odpowiednio skonfigurowałem `.gitignore`.

### Etap 3 - trenowanie rozpoznawania jednej kości
Celem tego etapu było stworzenie modelu AI, który rozpoznaje liczbę oczek (1–6) na pojedynczej kostce. Eksperymentowałem z różnymi parametrami augmentacji oraz architektury CNN. W Trialu 10 zastosowałem 2 warstwy Conv2D (32 i 64 filtry), dropout 0.3 oraz lekką augmentację. Ten wariant okazał się najskuteczniejszy – zarówno accuracy na zbiorze walidacyjnym, jak i treningowym rosło stabilnie bez oznak przeuczenia.

### Etap 4 - trenowanie rozpoznawania dwóch kości

## Problemy i wyzwania
### Transer Learning 
Pierwszym problemem okazał się transfer learning, na początku ciężko było stabilnie przenieść trening z pierwszego skryptu do drugiego, ale w końcu się udało. Mam nadzieję tylko, że pod koniec będzie to opłacalne i wytrenowanie na tylko jednym datasetcie nie będzie bardziej skuteczne.

### Zdjęcia pod kątem
Dużym problemem było nauczenie programu umiejętności rozpoznawania kości gdy zdjęcie było pod kątem. 

### Zbyt mały dataset
Pomimo połączenia 4 datasetów i posiadaniu w sumie 2000 obrazów była to zbyt mała ilość przez co ai często się myliła

## Załączniki
