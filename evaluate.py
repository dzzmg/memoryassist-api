"""Offline evaluation of the recommender (quality, not speed).

Run:  python evaluate.py
5-fold cross-validation over users: the model (item similarities, item means, gaze model)
is built only from the training users, and for every test user each item is predicted
from that user's OTHER items. The real next-day recall is the ground truth.

Methods compared:
  Random       - random scores
  Popularity   - the item's forget rate (same list for everybody)
  Gaze only    - logistic regression on the user's gaze for the item
  Item-CF      - item-based collaborative filtering (the main algorithm)
  Hybrid a=... - a * Item-CF + (1 - a) * Gaze, for a = 0.25, 0.5, 0.75
Metrics: ROC AUC, Precision@3 and Recall@3 (are the top-3 reminders really forgotten?).
"""
import csv
import os

import numpy as np

from app.database import db
from app import recommender as rec

FOLDS, TOP = 5, 3
rng = np.random.default_rng(42)


def auc(scores, labels):
    """Probability that a random forgotten item is ranked above a random remembered one."""
    s, y = np.asarray(scores), np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def main():
    user_ids, item_ids, R = rec.load_recall_matrix(db)
    feats, _ = rec.placement_dataset(db)
    n_users = len(user_ids)
    order = rng.permutation(n_users)
    folds = np.array_split(order, FOLDS)
    methods = ["Random", "Popularity", "Gaze only", "Item-CF", "Hybrid a=0.25", "Hybrid a=0.5", "Hybrid a=0.75"]
    pooled = {m: ([], []) for m in methods}
    p_at, r_at = {m: [] for m in methods}, {m: [] for m in methods}

    for test in folds:
        train = np.setdiff1d(order, test)
        S, means, _ = rec.item_similarity(R[train])
        X, y = [], []
        for u in train:
            for i in range(len(item_ids)):
                f = feats.get((user_ids[u], item_ids[i]))
                if f is not None and not np.isnan(R[u, i]):
                    X.append(f); y.append(R[u, i])
        model = rec.fit_logreg(X, y)
        w, b, mu, sd = np.array(model["w"]), model["b"], np.array(model["mu"]), np.array(model["sd"])

        for u in test:
            row = R[u]
            known = ~np.isnan(row)
            cf = rec.predict_cf(S, means, row)  # item i is predicted without using row[i]
            gz = np.array([
                1 / (1 + np.exp(-(((feats[(user_ids[u], item_ids[i])] - mu) / sd) @ w + b)))
                if (user_ids[u], item_ids[i]) in feats else means[i]
                for i in range(len(item_ids))
            ])
            scores = {
                "Random": rng.random(len(item_ids)),
                "Popularity": means,
                "Gaze only": gz,
                "Item-CF": cf,
                "Hybrid a=0.25": 0.25 * cf + 0.75 * gz,
                "Hybrid a=0.5": 0.5 * cf + 0.5 * gz,
                "Hybrid a=0.75": 0.75 * cf + 0.25 * gz,
            }
            truth = row[known]
            for m in methods:
                sc = scores[m][known]
                pooled[m][0].extend(sc); pooled[m][1].extend(truth)
                top = np.argsort(-sc)[:TOP]
                hits = truth[top].sum()
                p_at[m].append(hits / TOP)
                if truth.sum() > 0:
                    r_at[m].append(hits / truth.sum())

    rows = []
    print(f"{'Method':<14} {'AUC':>6} {'P@3':>6} {'R@3':>6}")
    for m in methods:
        a = auc(*pooled[m])
        p, r = float(np.mean(p_at[m])), float(np.mean(r_at[m]))
        rows.append({"method": m, "auc": round(a, 3), "precision_at_3": round(p, 3), "recall_at_3": round(r, 3)})
        print(f"{m:<14} {a:6.3f} {p:6.3f} {r:6.3f}")
    base = float(np.nanmean(R))
    print(f"\nUsers: {n_users}, items: {len(item_ids)}, overall forget rate: {base:.3f}")

    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader(); wr.writerows(rows)
    print("Saved results/evaluation.csv")


if __name__ == "__main__":
    main()
