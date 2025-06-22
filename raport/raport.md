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

<!-- ### Etap 3 - trenowanie rozpoznawania jednej kości
~~Celem tego etapu było stworzenie modelu AI, który rozpoznaje liczbę oczek (1–6) na pojedynczej kostce. Eksperymentowałem z różnymi parametrami augmentacji oraz architektury CNN. Na podstawie wcześniejszych prób, które zakończyły się błędem zarządzania plikami i utratą modeli (Trials 9 i 10) - [zobacz problem](#usunięty-zapis-modelu)- wybrałem dwa najlepsze warianty: Trial 13 i Trial 14. Wersja Trial 13 była bardziej stabilna, natomiast Trial 14 wyróżniała się bardziej agresywną augmentacją. Zakładając, że model jest na etapie wstępnego uczenia (prelearning) uznałem, że Trial 14 będzie lepszym wyborem, ponieważ trenowałem go na ograniczonym zbiorze danych, a dalsze etapy powinny stabilizować model.~~

~~W trialu 14 zastosowałem dwie warstwy Conv2D (32 i 64 filtry), dropout na poziomie 0.3 oraz lekką augmentację. Ten wariant okazał się najskuteczniejszy – zarówno accuracy na zbiorze walidacyjnym, jak i treningowym rosło znacząco bez oznak przeuczenia. Najwyższa osiągnięta dokładność wyniosła 54,2%. Po zakończeniu treningu przystąpiłem do kolejnego etapu, wykorzystując wytrenowany model jako punkt startowy.~~

### Etap 4 - trenowanie rozpoznawania dwóch kości
~~Etap 4 miał na celu kontynuację nauki modelu wytrenowanego w Etapie 3, tym razem na bardziej wymagającym zbiorze danych. Nowy zbiór zawierał 800 zdjęć przedstawiających dwie kości sfotografowane z trudnej perspektywy - pod kątem 45° z boku. Taka perspektywa powodowała, że na każdej kości widoczne były jednocześnie trzy ścianki [zobacz problem](#zdjęcia-pod-kątem). Dla 21 klas taka liczba przykładów była bardzo ograniczona co stanowiło dodatkowe wyzwanie.~~

~~W celu poprawy jakości klasyfikacji zastosowałem transfer learning z poprzedniego etapu z modelem MobileNetV2 oraz znacznie bardziej zaawansowaną augmentację, rozszerzoną m.in. o Mixup i CutMix. Najlepszy wynik osiągnąłem w Trial 21 osiągając dokładność: 15.3%, który został napisany na podstawie Trial 12 (dokładność: 14.7%), który był pierwotnym modelem, ale nie został zachowany z powodu błędu wspomnianego w 3 Etapie [zobacz problem](#usunięty-zapis-modelu). Model trenował się stabilnie przez ponad 100 epok, a najlepszy wynik uzyskałem jeszcze przed ukończeniem pełnego treningu, dzięki czemu moje obawy z hiperbolicznymi wynikami zostały zażegnane.Pomimo relatywnie niskiej dokładności, model uznałem za wystarczająco dobry dla danej bazy obrazów, by kontynuować dalsze eksperymenty.~~ -->

### Etap 3 - przemyślenie planu działania
W pierworodnym planie model miał rozpoznawać z całego zdjęcia jakie kości zostały wyrzucone, jednak w trakcie wykonywania go z każdym krokiem rodziły się nowe problemy. Ciągłe przenoszenie modelu poprzez transfer learning powodował, że model zaczynał się gubić, a ciągłe wykładnicze dokładanie liczby klas nie mogło się powieść przy tak ograniczonym datasetcie.

Po ustaleniu tego czego się nauczyłem i jakie są największe problemy mojego modelu uznałem, że muszę inaczej podejść do aktualnych przeciwności. Największym błędem było staranie się przypisania każdej konfugiracji odpowiedniej klasie - postanowiłem skorzystać z yolo, aby wycięło każdą kość ze zdjęcia po czym inny stworzony do tego model będzie klasyfikował jaka jest to kość.

### Etap 4 - trenowanie rozpoznawania jednej kości
Na początku chciałem skorzystać z prelearningu, ale zdjęcie kości z tłem utrudniało całą pracę. Dzięki temu, że jeden z moich datasetów był typu PASCAL VOC mogłem przyciąć wszystkie zdjęcia z tego datasetu (około 1200), a następnie uczyć model jaka to jest kość. Okazało się to sukcesem, ponieważ val_acc od 10 epoki oscylowało na poziomie 99-100%. Mimo tak wczesnych świetnych wyników model dalej się uczył i wartość val_loss ciągle spadała. 

Model ten został stworzony na podstawie skryptu trial-5. Wykorzystywał MobileNetV2 z częściowo odblokowanymi warstwami, do których dodana jest warstwa gęsta i dropout, augmentację typu Mixup i CutMix. Pomimo zawartego early stopping model uczył się aż do maksymalnej (100) epoki. Przy tak dobrych wynikach uznałem, że nie są potrzebne kolejne eksperymenty z augmentacją i zachowałem ten model jako najlepszy.

### Etap 5 - trenowanie rozpoznawania lokalizacji kości dzięki yolo
W tym etapie postanowiłem nauczyć model rozpoznawania lokalizacji kości za pomocą yolo. Pobrałem repozytorium yolo oraz utworzyłem parę skryptów, żeby pozwolić i ułatwić sobie pracę z nową biblioteką. Rozpącząłem naukę modelu dla odpowiedniego datasetu skryptem `yolo-train-run.pl`. Niestety przez brak karty NVIDIA musiałem korzystać z CPU i doprowadziło to do krytycznej temperatury, więc postanowiłem zakończyć naukę na 31 epoce. Pomimo wczesnego zakończenia rezultaty detekcji były bardzo zadowalające, chociaż zdarzały się błedy - najczęstszym z nich była wielokrotna detekcja jednej kości. 

Model radził sobie bardzo dobrze z innymi datasetami, a także ze zdjęciami wielu kości, chociaż spodziewałem się, że będzie to sprawiać większy problem i mam nadzieję, że następne dotrenowanie nie spowoduje utraty tej umiejętności. Po głębszym zastanowieniu doszedłem do wniosku, że model musi być dotrenowany, ale nie będę tego robić na swoim urządzeniu - skorzystałem z darmowej chmury - postawiłem na Google Colab ze względu na prostą konfigurację.

### Etap 6 - trenowanie yolo na Google Colab
Przejście na Google Colab okazało się świetną opcją - niedość, że trenowanie modelu w ten sposób nie obciążała mojego komputera, to jeszcze trening odbywał się znacznie szybciej. Niestety po zakończeniu się mojego dostępu do GPU cała nauka została utracona, więc musiałem poczekać, aż znowu uzyskam darmowy dostęp do pracy na GPU. Po kilku dniach trenowania modelu z przerwami model został ostatecznie wytrenowany. 

Wyniki były zadowalające: mAP_0.5 (średnia precyzja) na wysokim poziomie (około 98%), mAP_0.5:0.95 (bardziej rygorystyczne mAP_0.5) około 70%, precyzja ponad 95% i recall około 98%. Uznałem, że model yolo jest wytrenowany wystarczająco, ponieważ będzie wykorzystywany tylko do znajdowania obiektów, jednak miałem pewne obawy, ponieważ dataset był dosyć jednolity i bałem się, że przy bardziej wymagających zdjęciach (np. duża odległość kości od aparatu, duży kąt do podłoża) kości będą źle rozpoznawane.

### Etap 7 - trenowanie rozpoznawania jednej kości na innym datasetcie
Z gotowym modelem do rozpoznawania kości oraz yolo postanowiłem je przetestować na wstępnej aplikacji. Niestety przy użyciu trochę trudniejszych zdjęć zrobionych przeze mnie model słabo radził sobie z rozpoznawaniem kości wyciętych przez yolo. Z tego powodu postanowiłem rozpocząć trenowanie aktualnego modelu na trochę trudniejszym datasetcie, przerobionym przez mój model yolo.

Po kilku treningach wyniki val_acc były ciągle na poziomie 100%, jednak w rzeczywistym wykorzystaniu modelu jego dokładność była bardzo niska - cały czas rozpoznawał 4 i 6 jako 5. Postanowiłem przefiltrować datasety. Usunąłem obrazy które mogły powodować gorsze wyniki. Oprócz tego okazało się, że było parę błędnych obrazów - niektóre były pustymi zdjęciami, ale było też kilka błędnych kości przypisanych do odpowiedniego labelu. Ponownie nauczyłem model na dwóch skryptach treningowych. Dokładność dalej oscylowała na poziomie 100%, ale w praktyce model wciąż przypisywał 4 i 6 do 5.

### Etap 8 - sprawdzanie modelu za pomocą Grad-CAM dodanie OpenCV
Dla sprawdzenia z czego wynika błędne rozpoznawanie kości skorzystałem z Grad-CAM. Okazało się, że model patrzy tylko na część kości - najczęściej na róg. Moim zdaniem model nauczył się z poprzednich datasetów schematu, które oznacza rozpoznanie kości w pewnych miejscach, nie zwracając uwagi na najbardziej szczególny aspekt - oczka po środku. Po sprawdzeniu wielu zdjęć zdałem sobie sprawę, że gdy kości maja 4 lub 6 kości model nie patrzy na środek profilu, co powodowało główny problem błędnego rozpoznawania.

Próbowałem ograniczyć model do bezmyślnego wybierania kości z pięcioma oczkami - usunąłem większość ich zdjęć z datasetu, jednak model wciąż faworyzował tę klasę. Bałem się, że próba naprawy tego błędu będzie jak walka z wiatrakami oraz tego skutkiem ubocznym będzie utrata umiejętności modelu do rozpoznawania 5.

Postanowiłem więc wykorzystać najprostszą metodę - analiza obrazu bez nauki maszynowej. Metoda ta nie była w 100% skuteczna, więc postanowiłem wykorzystać hybrydę modelu .keras i OpenCV, które okazały się bardzo skuteczne. Wykorzystując logikę i rozpoznawanie kości przez obie metody, doszedłem do zadowalającego wyniku 80% dokładności (7 błędów na 35 kości, testy aplikacji były wykonywane już ręcznie).

Dodatkowo podczas tego etapu zauważyłem, że model yolo rozpoznaje tylko górną część kości, a problem z nakładającymi się boxami naprawiłem zwykłym sprawdzeniem punktów bloczków. Dzięki temu mogłem spokojnie przejść do następnego etapu - stworzenie nowego modelu do wygrywania w kościanego pokera.

### Etap 9 - stworzenie pierwowzoru aplikacji
Gdy miałem już stworzone modele do rozpoznawania zdjęć postanowiłem napisać szablon aplikacji. Chciałem, żeby to była prosta aplikacja konsolowa, więc nie korzystałem z żadnego frameworka. Do obliczania wyniku wykorzystałem funkcję z dawnego projektu, który miał ten sam temat przewodni. Celem aplikacji była możliwość grania przez jedną osobę z botem, którego miałem zamiar teraz wytrenować.
 
## Problemy i wyzwania
### Transer Learning 
~~Pierwszym problemem okazał się transfer learning, na początku ciężko było stabilnie przenieść trening z pierwszego skryptu do drugiego, ale w końcu się udało. Mam nadzieję tylko, że pod koniec będzie to opłacalne i wytrenowanie na tylko jednym datasetcie nie będzie bardziej skuteczne.~~

### Zbyt mały dataset
Pomimo połączenia 4 datasetów i posiadaniu w sumie 4500 obrazów była to zbyt mała ilość przez co ai często się myliła.

### Zdjęcia pod kątem
~~Dużym wyzwaniem okazało się rozpoznawanie kostek na zdjęciach wykonywanych pod kątem. Z uwagi na znaczne obniżenie dokładności dla kątów około 45°, planuję dalsze trenowanie modelu z uwzględnieniem tych trudniejszych ujęć. W przypadku, gdy ostateczny model będzie miał poważne problemy z rozpoznawaniem kości przy kątach nawet około 22,5°, rozważę wytrenowanie osobnego modelu na danych z dodatkowymi etykietami wysokości. Nie rozpocząłem nauki z tego poziomu od razu, aby uniknąć nieprawidłowego trenowania na dominującym w zbiorze materiale wykonanym z góry. Zamiast tego, skorzystałem z możliwości augmentacji, aby zwiększyć różnorodność danych wejściowych.~~

### Usunięty zapis modelu
Podczas pracy z repozytorium Git doszło do przypadkowego usunięcia zapisanych modeli z etapów 3 i 4, co uniemożliwiło ich ponowne wykorzystanie. Rozpocząłem trenowanie nowego modelu na tych samych etykietach i pierwszy wynik był słabszy aż o około 30% - dokładność spadła o 4%. Mimo chęci przejścia do następnego etapu z gorszym wynikiem, zdecydowałem się ponownie wytrenować model na najlepszych dostępnych wcześniej etykietach. To okazało się sukcesem - prawdopodobnie wcześniej skorzystałem ze złego skryptu i wynik nowego modelu nie tylko nie był gorszy, ale też lepszy o 4% względem poprzedniego. Można zatem powiedzieć, że nie ma tego złego co na dobre nie wyszło.

### Problem 5 oczek
Pomimo tego, że model osiągnął 100% val_accuracy i świetnie sobie radził na danych treningowych to rzeczywiste rozpoznawanie kości było bardzo słabe. Wszystkie kości z 4 i 6 oczkami uznawał za kości z 5 oczkami, przez co rzeczywista dokładność nie mogła przerosnąć 67%. Prawdopodobnym problemem było wyuczenie się modelu zwracania uwagi na niedecydujące elementy, a także zbyt mały dataset. Po wielu próbach i zmianach problem ten nie został rozwiązany tylko bardziej ominięty - wykorzystując metodę analizy obrazu powstała hybryda do wspólnego rozpoznawania kości z częściowo działającym modelem .keras.

## Załączniki
