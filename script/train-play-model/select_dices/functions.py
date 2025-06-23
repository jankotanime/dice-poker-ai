from itertools import product
from collections import Counter
from math import factorial

def pointCount(dices):
    if None in dices:
        return 0
    actPoints = 0
    if {1, 2, 3, 4, 5}.issubset(dices) or {2, 3, 4, 5, 6}.issubset(dices):
        actPoints += 13 + sum(dices) / 100
    else:
        dice = Counter(dices)
        for i in dice.keys():
            actPoints += int(i)**(1/10)*(dice[i]**2)
    return actPoints

def expected_score_for_mask(dices, mask):
    reroll_idxs = [i for i, m in enumerate(mask) if m]
    if not reroll_idxs:
        return pointCount(dices)
    
    total_score = 0
    count = 0
    for reroll_values in product(range(1,7), repeat=len(reroll_idxs)):
        new_hand = list(dices)
        for idx, val in zip(reroll_idxs, reroll_values):
            new_hand[idx] = val
        total_score += pointCount(new_hand)
        count += 1
    return total_score / count

def best_mask_and_score(dices):
    straights = [{1,2,3,4,5}, {2,3,4,5,6}]
    for target in straights:
        common = target & set(dices)
        if len(common) == 4:
            for i, die in enumerate(dices):
                if die not in common:
                    mask = [0]*5
                    mask[i] = 1
                    return mask, expected_score_for_mask(dices, mask)

    best_mask = None
    best_score = -1
    for mask_int in range(32):
        mask = [(mask_int >> i) & 1 for i in range(5)]
        score = expected_score_for_mask(dices, mask)
        if score > best_score:
            best_score = score
            best_mask = mask
    return best_mask, best_score


def hand_probability(hand):
    count = Counter(hand)
    denom = 1
    for v in count.values():
        denom *= factorial(v)
    total_permutations = factorial(len(hand)) // denom
    prob = total_permutations / (6 ** len(hand))
    return prob

def evaluate_hand_optimal(hand):
    best_mask, best_score = best_mask_and_score(hand)
    prob = hand_probability(hand)
    print(f"Ręka: {hand}")
    print(f"Maska (opt):   {best_mask}")
    print(f"Punktacja (opt):   {best_score:.2f}")
    print(f"Szansa na taki wynik: {prob*100:.4f}%")

def best_score_with_min_probability(dices, min_prob=0.5):
    best_mask = None
    best_score_threshold = None

    for mask_int in range(32):
        mask = [(mask_int >> i) & 1 for i in range(5)]
        reroll_idxs = [i for i, m in enumerate(mask) if m]
        if not reroll_idxs:
            score = pointCount(dices)
            if best_score_threshold is None or score > best_score_threshold:
                best_mask = mask
                best_score_threshold = score
            continue
        
        scores = []
        for reroll_values in product(range(1,7), repeat=len(reroll_idxs)):
            new_hand = list(dices)
            for idx, val in zip(reroll_idxs, reroll_values):
                new_hand[idx] = val
            scores.append(pointCount(new_hand))
        
        scores.sort(reverse=True)
        
        p = 1 / len(scores)
        
        cumulative_p = 0
        for score_val in scores:
            cumulative_p += p
            if cumulative_p >= min_prob:
                if best_score_threshold is None or score_val > best_score_threshold:
                    best_score_threshold = score_val
                    best_mask = mask
                break

    return best_mask, best_score_threshold