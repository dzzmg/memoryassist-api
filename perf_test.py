"""Performance test: how the recommender behaves as the database grows.

Run:  python perf_test.py            (about 2-5 minutes)
      python perf_test.py --quick    (smaller sizes, about 30 seconds)

It uses a SEPARATE database, memoryassist_perf, so the main data is not touched.
For every size (number of users) it generates synthetic data like seed.py
(each user places 12 of 50 items and recalls them the next day) and measures:

  build_s          time to build the model (item similarities + gaze model)
  naive_ms         one recommendation computed from scratch (no precomputed model)
  noindex_ms       recommendation with the precomputed model, WITHOUT the new indexes
  indexed_ms       recommendation with the precomputed model and the indexes
  cached_ms        recommendation served from the TTL cache collection
  docs_noindex     documents MongoDB examined for one user's query without the index
  docs_indexed     documents examined with the index (explain executionStats)

Results: results/performance.csv and charts in results/*.png
"""
import csv
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone

from pymongo import MongoClient

from app import recommender as rec
from app.database import MONGO_URI, create_indexes

QUICK = "--quick" in sys.argv
SIZES = [100, 500, 1000] if QUICK else [100, 1000, 5000, 10000, 20000]
N_ITEMS, N_LOCS, PER_USER = 50, 20, 12
REPEAT_FAST, REPEAT_NAIVE = 20, 5

client = MongoClient(MONGO_URI)
pdb = client["memoryassist_perf"]
random.seed(7)


def ms(fn, repeat, users):
    times = []
    for u in random.sample(users, min(repeat, len(users))):
        t0 = time.perf_counter()
        fn(pdb, u)
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 2)


def generate(n_users):
    for name in pdb.list_collection_names():
        pdb.drop_collection(name)
    items = [{"name": f"Item {k:02d}", "description": "synthetic item", "category": f"cat{k % 5}",
              "attributes": {"color": "grey", "size": "small", "salience": 1 + k % 5}} for k in range(N_ITEMS)]
    item_ids = pdb.items.insert_many(items).inserted_ids
    sal = {i: d["attributes"]["salience"] for i, d in zip(item_ids, items)}
    loc_ids = pdb.locations.insert_many(
        [{"code": f"L{k:02d}", "name": f"Place {k}", "zone": "z", "position": [k * 2.5, 1.0]} for k in range(N_LOCS)]
    ).inserted_ids
    user_ids = pdb.users.insert_many(
        [{"email": f"u{k}@perf.test", "passwordHash": "x", "role": "participant",
          "profile": {"memorySelfRating": random.randint(2, 5)}} for k in range(n_users)]
    ).inserted_ids
    ratings = {u: d for u, d in zip(user_ids, [random.randint(2, 5) for _ in user_ids])}
    t0 = datetime(2026, 11, 2, tzinfo=timezone.utc)
    batch = []
    for u in user_ids:
        for k, i in enumerate(random.sample(item_ids, PER_USER)):
            dwell = int(min(6000, max(200, random.lognormvariate(7.1, 0.6))))
            rev = min(4, int(random.expovariate(1.4)))
            ts = t0 + timedelta(minutes=k)
            batch.append({"userId": u, "itemId": i, "type": "placed", "locationId": random.choice(loc_ids),
                          "timestamp": ts, "gaze": {"dwellMs": dwell, "fixations": max(1, dwell // 280), "revisits": rev}})
            p = 1 / (1 + math.exp(-(-0.4 + 0.0009 * (dwell - 1200) + 0.7 * rev + 0.3 * (sal[i] - 3) + 0.2 * (ratings[u] - 3))))
            ok = random.random() < p
            batch.append({"userId": u, "itemId": i, "type": "recalled", "timestamp": ts + timedelta(days=1),
                          "recall": {"correct": ok, "errorMeters": 1.0 if ok else 12.0, "confidence": 4 if ok else 2}})
        if len(batch) >= 20000:
            pdb.interactions.insert_many(batch, ordered=False); batch = []
    if batch:
        pdb.interactions.insert_many(batch, ordered=False)
    return list(user_ids)


def docs_examined(user_id):
    plan = pdb.command("explain", {"find": "interactions", "filter": {"userId": user_id, "type": "recalled"}},
                       verbosity="executionStats")
    return plan["executionStats"]["totalDocsExamined"]


def run():
    rows = []
    for n in SIZES:
        t0 = time.perf_counter()
        users = generate(n)
        gen_s = time.perf_counter() - t0
        count = pdb.interactions.estimated_document_count()
        print(f"\n== {n} users, {count} interactions (generated in {gen_s:.1f} s)")

        # 1) no extra indexes (only _id)
        for name in pdb.interactions.index_information():
            if name != "_id_":
                pdb.interactions.drop_index(name)
        naive = ms(rec.recommend_naive, REPEAT_NAIVE, users)
        t1 = time.perf_counter(); rec.build_model(pdb); build_noidx = time.perf_counter() - t1
        noindex = ms(rec.recommend_for_user, REPEAT_FAST, users)
        d_no = docs_examined(users[0])

        # 2) with the indexes
        create_indexes(pdb)
        t1 = time.perf_counter(); rec.build_model(pdb); build_idx = time.perf_counter() - t1
        indexed = ms(rec.recommend_for_user, REPEAT_FAST, users)
        d_idx = docs_examined(users[0])

        # 3) cache: first call fills the cache, the measured calls read it
        sample = random.sample(users, min(REPEAT_FAST, len(users)))
        for u in sample:
            rec.cached_recommendations(pdb, u, 5)
        cached = statistics.median([rec.cached_recommendations(pdb, u, 5)[2] for u in sample])

        row = {"users": n, "interactions": count, "build_s_noindex": round(build_noidx, 3),
               "build_s": round(build_idx, 3), "naive_ms": naive, "noindex_ms": noindex,
               "indexed_ms": indexed, "cached_ms": round(cached, 2),
               "docs_noindex": d_no, "docs_indexed": d_idx}
        rows.append(row)
        print("  " + ", ".join(f"{k}={v}" for k, v in row.items()))

    os.makedirs("results", exist_ok=True)
    with open("results/performance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("\nSaved results/performance.csv")
    charts(rows)
    client.drop_database("memoryassist_perf")


def charts(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["users"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, label, mk in [("naive_ms", "Naive (recompute everything)", "o"),
                           ("noindex_ms", "Precomputed model, no index", "s"),
                           ("indexed_ms", "Precomputed model + index", "^"),
                           ("cached_ms", "Cached list (TTL collection)", "D")]:
        ax.plot(x, [r[key] for r in rows], marker=mk, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of users"); ax.set_ylabel("Median latency per request, ms (log)")
    ax.set_title("Recommendation latency as the database grows")
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("results/latency.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, [r["build_s_noindex"] for r in rows], marker="o", label="without indexes")
    ax.plot(x, [r["build_s"] for r in rows], marker="^", label="with indexes")
    ax.set_xlabel("Number of users"); ax.set_ylabel("Seconds")
    ax.set_title("Model build time (item similarities + gaze model)")
    ax.grid(True, alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("results/build_time.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.38
    pos = range(len(rows))
    ax.bar([p - width / 2 for p in pos], [r["docs_noindex"] for r in rows], width, label="without index (COLLSCAN)")
    ax.bar([p + width / 2 for p in pos], [r["docs_indexed"] for r in rows], width, label="with index (IXSCAN)")
    ax.set_xticks(list(pos)); ax.set_xticklabels([str(v) for v in x])
    ax.set_yscale("log"); ax.set_xlabel("Number of users"); ax.set_ylabel("Documents examined (log)")
    ax.set_title("Documents examined for one user's recall query")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig("results/docs_examined.png", dpi=160); plt.close(fig)
    print("Saved results/latency.png, results/build_time.png, results/docs_examined.png")


if __name__ == "__main__":
    run()
