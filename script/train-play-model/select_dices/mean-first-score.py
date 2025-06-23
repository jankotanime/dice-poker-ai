import random
from functions import pointCount

total_score = 0
num_hands = 1000000

print("Obliczam średnie wyniki...")

for _ in range(num_hands):
    random_hand = [random.randint(1, 6) for _ in range(5)]
    total_score += pointCount(random_hand)

average_score = total_score / num_hands
print(f"Średni wynik z {num_hands} losowych rąk: {average_score:.4f}")

# Wyniki w przedziale 9.54-9.56