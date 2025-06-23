import numpy as np
import pickle
from run import evaluate_hand_with_threshold

TRIAL = 1
MODEL_PATH = f"model/play-model/bet/trial-{TRIAL}.pkl"

def losuj_reke():
    return list(np.random.randint(1, 7, size=5))

samples = 10000

def generate_dataset(n_samples):
    procent = n_samples/100
    X = []
    y = []

    for i in range(n_samples):
        if (i%procent == 0):
            print(f"{i/procent}% do końca...")
        hand = losuj_reke()
        best_mask_opt, best_score_opt, best_mask_thresh, best_score_thresh, prob = evaluate_hand_with_threshold(hand)

        features = [
            best_score_opt,
            best_score_thresh,
            prob,
            best_score_opt - best_score_thresh,
            len(set(hand)),
            np.mean(hand),
            np.var(hand)
        ]

        label = 1 if best_score_thresh >= 25 and prob > 0.12 else 0

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = generate_dataset(samples)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((X, y), f)
    print("Zapisano dane treningowe")
