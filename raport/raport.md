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
Celem tego etapu było stworzenie modelu AI, który rozpoznaje liczbę oczek (1–6) na pojedynczej kostce. Eksperymentowałem z różnymi parametrami augmentacji oraz architektury CNN. Na podstawie wcześniejszych prób, które zakończyły się błędem zarządzania plikami i utratą modeli (Trials 9 i 10) - [zobacz problem](#usunięty-zapis-modelu)- wybrałem dwa najlepsze warianty: Trial 13 i Trial 14. Wersja Trial 13 była bardziej stabilna, natomiast Trial 14 wyróżniała się bardziej agresywną augmentacją. Zakładając, że model jest na etapie wstępnego uczenia (prelearning) uznałem, że Trial 14 będzie lepszym wyborem, ponieważ trenowałem go na ograniczonym zbiorze danych, a dalsze etapy powinny stabilizować model.

W trialu 14 zastosowałem dwie warstwy Conv2D (32 i 64 filtry), dropout na poziomie 0.3 oraz lekką augmentację. Ten wariant okazał się najskuteczniejszy – zarówno accuracy na zbiorze walidacyjnym, jak i treningowym rosło znacząco bez oznak przeuczenia. Najwyższa osiągnięta dokładność wyniosła 54,2%. Po zakończeniu treningu przystąpiłem do kolejnego etapu, wykorzystując wytrenowany model jako punkt startowy.

### Etap 4 - trenowanie rozpoznawania dwóch kości
Etap 4 miał na celu kontynuację nauki modelu wytrenowanego w Etapie 3, tym razem na bardziej wymagającym zbiorze danych. Nowy zbiór zawierał 800 zdjęć przedstawiających dwie kości sfotografowane z trudnej perspektywy - pod kątem 45° z boku. Taka perspektywa powodowała, że na każdej kości widoczne były jednocześnie trzy ścianki [zobacz problem](#zdjęcia-pod-kątem). Dla 21 klas taka liczba przykładów była bardzo ograniczona co stanowiło dodatkowe wyzwanie.

W celu poprawy jakości klasyfikacji zastosowałem transfer learning z poprzedniego etapu z modelem MobileNetV2 oraz znacznie bardziej zaawansowaną augmentację, rozszerzoną m.in. o Mixup i CutMix. Najlepszy wynik osiągnąłem w Trial 21 osiągając dokładność: 15.3%, który został napisany na podstawie Trial 12 (dokładność: 14.7%), który był pierwotnym modelem, ale nie został zachowany z powodu błędu wspomnianego w 3 Etapie [zobacz problem](#usunięty-zapis-modelu). Model trenował się stabilnie przez ponad 100 epok, a najlepszy wynik uzyskałem jeszcze przed ukończeniem pełnego treningu, dzięki czemu moje obawy z hiperbolicznymi wynikami zostały zażegnane.Pomimo relatywnie niskiej dokładności, model uznałem za wystarczająco dobry dla danej bazy obrazów, by kontynuować dalsze eksperymenty.

### Etap 

## Problemy i wyzwania
### Transer Learning 
Pierwszym problemem okazał się transfer learning, na początku ciężko było stabilnie przenieść trening z pierwszego skryptu do drugiego, ale w końcu się udało. Mam nadzieję tylko, że pod koniec będzie to opłacalne i wytrenowanie na tylko jednym datasetcie nie będzie bardziej skuteczne.

### Zbyt mały dataset
Pomimo połączenia 4 datasetów i posiadaniu w sumie 2000 obrazów była to zbyt mała ilość przez co ai często się myliła.

### Zdjęcia pod kątem
Dużym wyzwaniem okazało się rozpoznawanie kostek na zdjęciach wykonywanych pod kątem. Z uwagi na znaczne obniżenie dokładności dla kątów około 45°, planuję dalsze trenowanie modelu z uwzględnieniem tych trudniejszych ujęć. W przypadku, gdy ostateczny model będzie miał poważne problemy z rozpoznawaniem kości przy kątach nawet około 22,5°, rozważę wytrenowanie osobnego modelu na danych z dodatkowymi etykietami wysokości. Nie rozpocząłem nauki z tego poziomu od razu, aby uniknąć nieprawidłowego trenowania na dominującym w zbiorze materiale wykonanym z góry. Zamiast tego, skorzystałem z możliwości augmentacji, aby zwiększyć różnorodność danych wejściowych.

### Usunięty zapis modelu
Podczas pracy z repozytorium Git doszło do przypadkowego usunięcia zapisanych modeli z etapów 3 i 4, co uniemożliwiło ich ponowne wykorzystanie. Rozpocząłem trenowanie nowego modelu na tych samych etykietach i pierwszy wynik był słabszy aż o około 30% - dokładność spadła o 4%. Mimo chęci przejścia do następnego etapu z gorszym wynikiem, zdecydowałem się ponownie wytrenować model na najlepszych dostępnych wcześniej etykietach. To okazało się sukcesem - prawdopodobnie wcześniej skorzystałem ze złego skryptu i wynik nowego modelu nie tylko nie był gorszy, ale też lepszy o 4% względem poprzedniego. Można zatem powiedzieć, że nie ma tego złego co na dobre nie wyszło.

## Załączniki
