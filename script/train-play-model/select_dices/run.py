from functions import *

def print_results(best_mask_opt, best_score_opt, best_mask_thresh, best_score_thresh, prob, hand):
    min_prob=1/6
    print(f"Ręka: {hand}")
    print(f"Maska (optymalna oczekiwana): {best_mask_opt}")
    print(f"Najlepszy oczekiwany wynik: {best_score_opt:.2f}")
    print(f"Maska (wynik z prawdopodobieństwem >= {min_prob*100:.1f}%): {best_mask_thresh}")
    print(f"Minimalny wynik punktacji z prawdopodobieństwem >= {min_prob*100:.1f}%: {best_score_thresh:.2f}")
    print(f"Szansa na taki wynik: {prob*100:.4f}%")

def evaluate_hand_with_threshold(hand, min_prob=1/6):
    best_mask_opt, best_score_opt = best_mask_and_score(hand)
    best_mask_thresh, best_score_thresh = best_score_with_min_probability(hand, min_prob)
    prob = hand_probability(hand)

    return best_mask_opt, best_score_opt, best_mask_thresh, best_score_thresh, prob

# print_results(*evaluate_hand_with_threshold([2,3,4,4,6]), [2,3,4,4,6])
# print_results(*evaluate_hand_with_threshold([1,3,4,5,6]), [1,3,4,5,6])
# print_results(*evaluate_hand_with_threshold([2,3,4,5,6]), [2,3,4,5,6])
# print_results(*evaluate_hand_with_threshold([2,3,4,5,1]), [2,3,4,5,1])
# print_results(*evaluate_hand_with_threshold([1,2,5,5,1]), [1,2,5,5,1])
# print_results(*evaluate_hand_with_threshold([2,2,4,4,1]), [2,2,4,4,1])
# print_results(*evaluate_hand_with_threshold([1,1,1,1,1]), [1,1,1,1,1])
# print_results(*evaluate_hand_with_threshold([6,6,6,6,6]), [6,6,6,6,6])
