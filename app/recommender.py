"""Recommendation engine: which items is a user most likely to forget?

The "rating" in this project is forgetting: for every (user, item) pair that has a
next-day recall result, r = 1 if the user forgot where the item was, 0 if they remembered.

1. Item-based collaborative filtering (the main algorithm)
   - Build the user x item matrix R from the "recalled" interactions.
   - Centre each item column by its mean (the item's forget rate).
   - Similarity of two items = cosine of their centred columns over the users who
     recalled both (adjusted cosine), shrunk towards 0 when few users overlap.
   - Keep the top-K most similar items of every item ("neighbours") and store them
     in the collection item_similarity, so requests do not recompute them.
   - Prediction for user u and item i:
         r(u,i) = mean_i + sum_j sim(i,j) * (r(u,j) - mean_j) / sum_j |sim(i,j)|
     over the neighbours j of i that user u has recalled (j != i).
     A user who forgot more than average on similar items gets a higher score.
   - New users without recall history get mean_i (the item's forget rate).

2. Gaze risk (content part of the hybrid)
   Logistic regression on the gaze recorded when the item was placed
   (log dwell time, fixations, revisits) and on item salience.

Final score = ALPHA * CF + (1 - ALPHA) * gaze risk. The items with the highest score
are recommended as reminders.
"""
import math
import time
from datetime import datetime, timezone

import numpy as np

ALPHA = 0.25         # weight of CF in the hybrid score (chosen by cross-validation, see evaluate.py)
NEIGHBOURS = 5       # K most similar items kept per item
SHRINK = 10          # similarity shrinkage: sim * n / (n + SHRINK)
CACHE_SECONDS = 600  # lifetime of a cached recommendation list


# ---------------------------------------------------------------- data loading
def load_recall_matrix(db):
    """Return (user_ids, item_ids, R) where R[u, i] is 1 (forgot), 0 (remembered) or nan."""
    cursor = db.interactions.find(
        {"type": "recalled"},
        {"_id": 0, "userId": 1, "itemId": 1, "recall.correct": 1},
    )
    rows = [(d["userId"], d["itemId"], 0.0 if d["recall"]["correct"] else 1.0) for d in cursor]
    item_ids = [d["_id"] for d in db.items.find({}, {"_id": 1}).sort("_id", 1)]
    user_ids = sorted({u for u, _, _ in rows})
    u_index = {u: k for k, u in enumerate(user_ids)}
    i_index = {i: k for k, i in enumerate(item_ids)}
    R = np.full((len(user_ids), len(item_ids)), np.nan)
    for u, i, r in rows:  # later rows overwrite earlier ones (latest recall wins)
        if i in i_index:
            R[u_index[u], i_index[i]] = r
    return user_ids, item_ids, R


def item_similarity(R):
    """Adjusted-cosine similarity between item columns of R (nan = unknown)."""
    known = ~np.isnan(R)
    counts = known.sum(axis=0)
    means = np.where(counts > 0, np.nansum(R, axis=0) / np.maximum(counts, 1), 0.0)
    C = np.where(known, R - means, 0.0)                 # centred, unknown -> 0
    dot = C.T @ C
    sq = (C ** 2).T @ known.astype(float)               # sum of c_ui^2 over users who also know j
    denom = np.sqrt(sq * sq.T)
    S = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    overlap = known.T.astype(float) @ known.astype(float)
    S = S * overlap / (overlap + SHRINK)                # shrink when few users overlap
    np.fill_diagonal(S, 0.0)
    return S, means, counts


def predict_cf(S, means, user_row, neighbours=NEIGHBOURS):
    """Predicted forgetting for every item for one user (row of R, nan = unknown)."""
    preds = means.copy()
    known = ~np.isnan(user_row)
    dev = np.where(known, user_row - means, 0.0)
    for i in range(len(means)):
        sims = S[i].copy()
        sims[~known] = 0.0
        sims[i] = 0.0
        top = np.argsort(-np.abs(sims))[:neighbours]
        top = top[sims[top] != 0]
        if len(top):
            preds[i] = means[i] + (sims[top] @ dev[top]) / np.abs(sims[top]).sum()
    return np.clip(preds, 0.0, 1.0)


# ---------------------------------------------------------------- gaze model
def gaze_features(gaze, salience):
    return np.array([
        math.log1p(gaze.get("dwellMs", 0)),
        gaze.get("fixations", 0),
        gaze.get("revisits", 0),
        salience or 3,
    ], dtype=float)


def fit_logreg(X, y, epochs=800, lr=0.1):
    """Logistic regression with plain gradient descent on standardised features."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= lr * Z.T @ (p - y) / len(y)
        b -= lr * float(np.mean(p - y))
    return {"w": w.tolist(), "b": b, "mu": mu.tolist(), "sd": sd.tolist(),
            "features": ["log_dwell", "fixations", "revisits", "salience"]}


def placement_dataset(db):
    """Return {(userId, itemId): features} for the latest placement and {(u, i): forgot} for recalls."""
    salience = {d["_id"]: d.get("attributes", {}).get("salience", 3) for d in db.items.find()}
    feats = {}
    for d in db.interactions.find({"type": "placed"}, {"userId": 1, "itemId": 1, "gaze": 1}).sort("timestamp", 1):
        if d.get("gaze"):
            feats[(d["userId"], d["itemId"])] = gaze_features(d["gaze"], salience.get(d["itemId"]))
    labels = {}
    for d in db.interactions.find({"type": "recalled"}, {"userId": 1, "itemId": 1, "recall.correct": 1}).sort("timestamp", 1):
        labels[(d["userId"], d["itemId"])] = 0.0 if d["recall"]["correct"] else 1.0
    return feats, labels


def train_gaze_model(db):
    """Logistic regression: features of a placement -> probability of forgetting it."""
    feats, labels = placement_dataset(db)
    keys = [k for k in feats if k in labels]
    if len(keys) < 10:
        return None
    return fit_logreg([feats[k] for k in keys], [labels[k] for k in keys])


def gaze_risk(model, gaze, salience):
    if not model or not gaze:
        return None
    z = (gaze_features(gaze, salience) - np.array(model["mu"])) / np.array(model["sd"])
    return float(1 / (1 + np.exp(-(z @ np.array(model["w"]) + model["b"]))))


# ---------------------------------------------------------------- model build
def build_model(db):
    """Recompute item similarities and the gaze model and store them in MongoDB."""
    t0 = time.perf_counter()
    user_ids, item_ids, R = load_recall_matrix(db)
    S, means, counts = item_similarity(R)
    now = datetime.now(timezone.utc)
    docs = []
    for k, item_id in enumerate(item_ids):
        order = np.argsort(-S[k])[:NEIGHBOURS]
        docs.append({
            "_id": item_id,
            "forgetRate": round(float(means[k]), 4),
            "recallCount": int(counts[k]),
            "neighbours": [{"itemId": item_ids[j], "sim": round(float(S[k, j]), 4)}
                           for j in order if round(float(S[k, j]), 4) > 0],
            "updatedAt": now,
        })
    db.item_similarity.delete_many({})
    if docs:
        db.item_similarity.insert_many(docs)
    model = train_gaze_model(db)
    db.models.replace_one({"_id": "gaze_lr"}, {"_id": "gaze_lr", **(model or {}), "updatedAt": now}, upsert=True)
    db.recommendations.delete_many({})  # old cached lists are no longer valid
    return {"users": len(user_ids), "items": len(item_ids),
            "recalls": int((~np.isnan(R)).sum()), "seconds": round(time.perf_counter() - t0, 3)}


def ensure_model(db):
    if db.item_similarity.estimated_document_count() == 0:
        build_model(db)


# ---------------------------------------------------------------- recommendations
def recommend_for_user(db, user_id, k=5):
    """Ranked reminders for one user, using the stored model (fast path)."""
    ensure_model(db)
    sim_docs = list(db.item_similarity.find())
    items = {d["_id"]: d for d in db.items.find({}, {"name": 1, "category": 1, "attributes": 1})}
    model = db.models.find_one({"_id": "gaze_lr"})
    model = model if model and model.get("w") else None

    # this user's own recall results (index {userId, type, timestamp})
    own = {}
    for d in db.interactions.find({"userId": user_id, "type": "recalled"},
                                  {"itemId": 1, "recall.correct": 1}).sort("timestamp", 1):
        own[d["itemId"]] = 0.0 if d["recall"]["correct"] else 1.0

    # the latest placement of every item by this user (what the glasses saw)
    placed = {}
    for d in db.interactions.find({"userId": user_id, "type": "placed"},
                                  {"itemId": 1, "gaze": 1, "locationId": 1, "timestamp": 1}).sort("timestamp", 1):
        placed[d["itemId"]] = d

    means = {d["_id"]: d["forgetRate"] for d in sim_docs}
    results = []
    for d in sim_docs:
        i = d["_id"]
        if i not in items:
            continue
        num = den = 0.0
        because = []
        for nb in d["neighbours"]:
            j = nb["itemId"]
            if j == i or j not in own:
                continue
            num += nb["sim"] * (own[j] - means.get(j, 0.0))
            den += abs(nb["sim"])
            if own[j] == 1.0:
                because.append(items[j]["name"] if j in items else str(j))
        cf = means.get(i, 0.0) + (num / den if den else 0.0)
        cf = min(max(cf, 0.0), 1.0)
        p = placed.get(i)
        g = gaze_risk(model, p.get("gaze") if p else None, items[i].get("attributes", {}).get("salience"))
        score = ALPHA * cf + (1 - ALPHA) * g if g is not None else cf
        results.append({
            "itemId": str(i),
            "item": items[i]["name"],
            "category": items[i].get("category"),
            "score": round(score, 3),
            "cfScore": round(cf, 3),
            "gazeRisk": None if g is None else round(g, 3),
            "itemForgetRate": round(means.get(i, 0.0), 3),
            "becauseYouForgot": because[:3],
            "lastPlacedAt": p["timestamp"] if p else None,
            "locationId": str(p["locationId"]) if p and p.get("locationId") else None,
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def recommend_naive(db, user_id, k=5):
    """Baseline without precomputation: rebuild the whole matrix on every request."""
    user_ids, item_ids, R = load_recall_matrix(db)
    S, means, _ = item_similarity(R)
    if user_id in user_ids:
        row = R[user_ids.index(user_id)]
    else:
        row = np.full(len(item_ids), np.nan)
    preds = predict_cf(S, means, row)
    order = np.argsort(-preds)[:k]
    return [{"itemId": str(item_ids[j]), "cfScore": round(float(preds[j]), 3)} for j in order]


def cached_recommendations(db, user_id, k=5):
    """Return (list, cached?, milliseconds). Cache lives in a TTL collection."""
    t0 = time.perf_counter()
    hit = db.recommendations.find_one({"_id": user_id})
    if hit and len(hit.get("items", [])) >= k:
        return hit["items"][:k], True, round((time.perf_counter() - t0) * 1000, 2)
    recs = recommend_for_user(db, user_id, k=max(k, 10))
    db.recommendations.replace_one(
        {"_id": user_id},
        {"_id": user_id, "items": recs, "createdAt": datetime.now(timezone.utc)},
        upsert=True,
    )
    return recs[:k], False, round((time.perf_counter() - t0) * 1000, 2)


def invalidate(db, user_id):
    db.recommendations.delete_one({"_id": user_id})
