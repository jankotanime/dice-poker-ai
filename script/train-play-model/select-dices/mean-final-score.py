import random
from functions import pointCount, best_mask_and_score

total_score = 0
trials = 10000
procent = trials/100

for i in range(trials):
    if (i % procent == 0):
        print(f"{i/procent}% do końca...")
    hand = [random.randint(1, 6) for _ in range(5)]
    
    mask, _ = best_mask_and_score(hand)
    
    new_hand = [
        random.randint(1, 6) if m else d
        for d, m in zip(hand, mask)
    ]
    
    total_score += pointCount(new_hand)

average_score = total_score / trials
print(f"Średni wynik po {trials} grach: {average_score:.2f}")

# Średni wynik po drugim rzucie - około 12, średnio zwiększa się o 2.5 od pierwszego rzutu