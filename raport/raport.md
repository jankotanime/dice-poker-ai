## Etap 1 – Trenowanie modelu do rozpoznawania jednej kostki do gry
Celem tego etapu projektu było stworzenie modelu sztucznej inteligencji, który na podstawie zdjęcia pojedynczej kostki do gry będzie potrafił rozpoznać liczbę oczek (od 1 do 6).

Do trenowania modelu wykorzystałem zbiór danych znajdujący się w katalogu train/image/one-dice, który zawierał łącznie 133 zdjęcia kostek, podzielonych na sześć klas odpowiadających liczbie oczek. Dane zostały uporządkowane w strukturze katalogów (dice-1 do dice-6), a wszystkie obrazy zostały automatycznie przeskalowane do rozmiaru 128x128 pikseli. Zbiór danych został podzielony w proporcji 80:20 – 109 zdjęć przeznaczono na trening, a 24 na walidację.

Do zbudowania modelu wykorzystałem prostą konwolucyjną sieć neuronową (CNN) zawierającą trzy warstwy konwolucyjne z max poolingiem, warstwę Flatten oraz w pełni połączoną warstwę Dense. Na końcu model posiada sześć neuronów wyjściowych z funkcją aktywacji softmax, pozwalającą na klasyfikację do jednej z sześciu klas. Jako funkcję kosztu zastosowano categorical_crossentropy, a optymalizacja przebiegała z użyciem algorytmu Adam. Model trenowany był przez 10 epok z rozmiarem batcha równym 32. Proces treningu odbywał się z użyciem TensorFlow na CPU.

Podczas eksperymentów analizowałem wpływ różnych parametrów na skuteczność modelu, w tym liczbę warstw i parametrów konwolucyjnych. Przeprowadziłem kilka iteracji treningu, aby dobrać odpowiednie ustawienia. Obserwowałem dokładność i stratę zarówno dla danych treningowych, jak i walidacyjnych w każdej epoce.

Na podstawie wyników można zauważyć, że model stopniowo uczył się danych treningowych – dokładność treningowa wzrosła z około 13% do 52%, a strata spadła z 1.86 do 1.25. Niestety, dokładność na zbiorze walidacyjnym była niestabilna i oscylowała w przedziale 12–29%, ostatecznie spadając do poziomu około 16%. Wartości strat dla walidacji wzrastały, co sugeruje występowanie przeuczenia (overfittingu).

Na tym etapie wyciągnąłem następujące wnioski:

Model jest zbyt dopasowany do danych treningowych i nie generalizuje dobrze do nowych danych.

Głównym ograniczeniem jest zbyt mały zbiór danych oraz brak augmentacji obrazów.

W kolejnych krokach należy wprowadzić augmentację danych (np. obrót, skalowanie, zmiana jasności) oraz rozważyć zastosowanie bardziej złożonej architektury modelu lub transfer learningu (np. z użyciem MobileNetV2).

W dalszej części projektu planuję rozszerzyć zbiór danych poprzez augmentację oraz przeprowadzić trening z użyciem tych danych, aby sprawdzić wpływ sztucznego zwiększenia zbioru uczącego na zdolność modelu do generalizacji.
