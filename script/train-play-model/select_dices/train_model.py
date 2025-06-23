import pickle
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import os

TRIAL = 1
MODEL_PATH = f"model/play-model/bet/model-1.pkl"
DATASET_PATH = f"model/play-model/bet/trial-1.pkl"
RAPORT_PATH = "raport/train-play-model"
os.makedirs(RAPORT_PATH, exist_ok=True)

with open(DATASET_PATH, "rb") as f:
    X_all, _ = pickle.load(f)

print(f"Przykład 0 ma shape: {X_all.shape}, typ elementu 0: {type(X_all[0])}, wartość elementu 0: {X_all[0]}")

y_all = np.array([1 if features[0] >= 9.5 else 0 for features in X_all])
print("Rozkład etykiet y_all:", Counter(y_all))


def generate_money():
    return np.random.randint(100, 1001)


def generate_max_score(features):
    """Szacujemy maksymalną punktację na podstawie cech."""
    return features[0] + np.random.uniform(0, 2)


def pair_data(X_all, y_all, n_samples=5000):
    paired_X = []
    paired_y = []

    for _ in range(n_samples):
        idx_ai = np.random.randint(0, len(X_all))
        idx_opp = np.random.randint(0, len(X_all))

        ai_features = X_all[idx_ai]
        opp_features = X_all[idx_opp]
        ai_label = y_all[idx_ai]

        ai_money = generate_money()
        opp_money = generate_money()

        max_bid = min(ai_money, opp_money)
        current_bid = np.random.randint(10, max_bid // 2 + 1)

        ai_score_est = generate_max_score(ai_features)
        opp_score_est = generate_max_score(opp_features)

        features = [
            ai_money,
            opp_money,
            current_bid,
            ai_score_est,
            opp_score_est,
        ]

        paired_X.append(features)
        paired_y.append(ai_label)

    return np.array(paired_X), np.array(paired_y)

X, y = pair_data(X_all, y_all)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

acc_train = clf.score(X_train, y_train)
acc_test = clf.score(X_test, y_test)
print(f"Train accuracy: {acc_train:.2f}")
print(f"Test accuracy: {acc_test:.2f}")

feature_names = [
    "ai_total_money", "opp_total_money", "current_bid",
    "ai_max_score", "opp_max_score"
]

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=feature_names, class_names=["Pass", "Raise"])
plt.title("Drzewo decyzyjne: Czy AI powinno podbić stawkę?")
plt.savefig(f"{RAPORT_PATH}/decision_tree.png")
plt.close()

importances = clf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

print("Feature importance (od największego wpływu):")
for idx in sorted_indices:
    print(f"{feature_names[idx]}: {importances[idx]:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances[sorted_indices])
plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_indices])
plt.xlabel("Ważność cechy")
plt.title("Wpływ cech na decyzje drzewa")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{RAPORT_PATH}/feature_importance.png")
plt.close()

with open(f"{RAPORT_PATH}/report.txt", "w") as f:
    f.write("=== Decision Tree Report ===\n")
    f.write(f"Train accuracy: {acc_train:.2f}\n")
    f.write(f"Test accuracy: {acc_test:.2f}\n")
    f.write("Drzewo zapisane jako 'decision_tree.png'\n")


def predict_decision():
    idx_ai = np.random.randint(0, len(X_all))
    idx_opp = np.random.randint(0, len(X_all))

    ai_features = X_all[idx_ai]
    opp_features = X_all[idx_opp]

    ai_money = generate_money()
    opp_money = generate_money()
    max_bid = min(ai_money, opp_money)
    current_bid = np.random.randint(10, max_bid // 2 + 1)

    ai_score_est = generate_max_score(ai_features)
    opp_score_est = generate_max_score(opp_features)

    input_vec = [ai_money, opp_money, current_bid, ai_score_est, opp_score_est]
    decision = clf.predict([input_vec])[0]

    if decision == 1:
        bid = min(max_bid, current_bid * 2)
        print(f"AI decyduje się grać. Podbija do: {bid} zł")
    else:
        print(f"AI pasuje. Traci {current_bid} zł.")

predict_decision()

with open(MODEL_PATH, "wb") as f_model:
    pickle.dump(clf, f_model)
print(f"Zapisano model drzewa decyzyjnego do {MODEL_PATH}")
