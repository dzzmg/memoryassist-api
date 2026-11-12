"""Fill the database with synthetic test data.

Run:  python seed.py
WARNING: it deletes the existing data in the four collections first.

The data imitates the planned Aria glasses experiment: 30 participants walk along
the Block 3 corridor, leave 12 small items, and one day later try to recall where
each item is. Gaze values are generated so that looking longer and looking back
at an item make remembering it more likely. No real participant data is used.
"""
import math
import random
from datetime import datetime, timedelta, timezone

from app.database import create_indexes, db, interactions, items, locations, users
from app.recommender import build_model
from app.security import hash_password

random.seed(42)  # the same data every time

LOCATIONS = [
    ("L01", "Bench near the main entrance", "entrance", [2.0, 1.0]),
    ("L02", "Notice board", "entrance", [6.5, 0.2]),
    ("L03", "Window sill by the stairs", "stairs", [12.0, 2.8]),
    ("L04", "Water cooler", "stairs", [15.5, 0.3]),
    ("L05", "Printer corner", "middle", [22.0, 2.7]),
    ("L06", "Sofa area", "middle", [27.5, 1.5]),
    ("L07", "Recycling bins", "middle", [33.0, 0.2]),
    ("L08", "Fire extinguisher cabinet", "far end", [39.0, 2.8]),
    ("L09", "Elevator hall", "far end", [45.0, 1.2]),
    ("L10", "Plant pot by the last room", "far end", [51.0, 2.6]),
]

ITEMS = [
    ("Key", "Small metal key on a red ring", "personal belongings", "silver", "small", 3),
    ("Pen", "Blue ballpoint pen", "stationery", "blue", "small", 2),
    ("USB flash drive", "16 GB USB stick with a cap", "electronics", "black", "small", 2),
    ("Student ID card", "Plastic university ID card", "personal belongings", "white", "small", 3),
    ("Earphones case", "Round case for wireless earphones", "electronics", "white", "small", 3),
    ("Coin", "100 tenge coin", "personal belongings", "gold", "small", 1),
    ("Sticky notes", "Pad of yellow sticky notes", "stationery", "yellow", "medium", 5),
    ("Eraser", "Pink rubber eraser", "stationery", "pink", "small", 2),
    ("Lip balm", "Small tube of lip balm", "accessories", "green", "small", 3),
    ("Hair clip", "Plastic hair clip", "accessories", "purple", "small", 2),
    ("Paper clip box", "Small box of metal paper clips", "stationery", "orange", "medium", 4),
    ("Mini notebook", "Pocket notebook with a hard cover", "stationery", "red", "medium", 5),
]

AGE_GROUPS = ["18-21", "22-25", "26-30"]
FIELDS = ["Computer Science", "Engineering", "Biology", "Economics", "Physics", "Medicine"]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def main():
    for col in (users, items, locations, interactions):
        col.delete_many({})
    for name in ("item_similarity", "recommendations", "models"):
        db[name].delete_many({})
    create_indexes()

    # ---- locations ----
    loc_ids = locations.insert_many(
        [{"code": c, "name": n, "zone": z, "position": p} for c, n, z, p in LOCATIONS]
    ).inserted_ids

    # ---- items ----
    item_docs = [
        {"name": n, "description": d, "category": cat,
         "attributes": {"color": col, "size": size, "salience": sal},
         "createdAt": datetime.now(timezone.utc)}
        for n, d, cat, col, size, sal in ITEMS
    ]
    item_ids = items.insert_many(item_docs).inserted_ids

    # ---- users: one researcher (admin) + 30 participants ----
    users.insert_one({
        "email": "researcher@memoryassist.test", "passwordHash": hash_password("admin1234"),
        "role": "admin", "profile": {"displayName": "Researcher"},
        "createdAt": datetime.now(timezone.utc),
    })
    participant_hash = hash_password("test1234")  # same test password for all participants
    participants = []
    for i in range(1, 31):
        participants.append({
            "email": f"p{i:02d}@memoryassist.test",
            "passwordHash": participant_hash,
            "role": "participant",
            "profile": {
                "displayName": f"Participant P{i:02d}",
                "ageGroup": random.choice(AGE_GROUPS),
                "fieldOfStudy": random.choice(FIELDS),
                "wearsGlasses": random.random() < 0.35,
                "memorySelfRating": random.randint(2, 5),
            },
            "createdAt": datetime.now(timezone.utc),
        })
    user_ids = users.insert_many(participants).inserted_ids

    # ---- interactions ----
    docs = []
    day1 = datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)
    for u_index, (uid, user) in enumerate(zip(user_ids, participants)):
        session = f"S{u_index + 1:02d}"
        start = day1 + timedelta(minutes=40 * u_index)
        rating = user["profile"]["memorySelfRating"]
        order = list(range(len(ITEMS)))
        random.shuffle(order)
        for step, k in enumerate(order):
            salience = ITEMS[k][5]
            dwell = int(min(6000, max(200, random.lognormvariate(7.1, 0.6))))  # ms, median ~1200
            fixations = max(1, int(dwell / 280 + random.gauss(0, 1.5)))
            revisits = min(4, int(random.expovariate(1.4)))
            placed_at = start + timedelta(minutes=step * 1.5)
            docs.append({
                "userId": uid, "itemId": item_ids[k], "type": "placed", "sessionId": session,
                "locationId": random.choice(loc_ids), "timestamp": placed_at,
                "gaze": {"dwellMs": dwell, "fixations": fixations, "revisits": revisits},
            })

            # next day: recall without the glasses
            p_remember = sigmoid(-0.4 + 0.0009 * (dwell - 1200) + 0.7 * revisits
                                 + 0.3 * (salience - 3) + 0.2 * (rating - 3))
            correct = random.random() < p_remember
            error = round(random.uniform(0, 2) if correct else random.uniform(3, 30), 1)
            confidence = random.randint(3, 5) if correct else random.randint(1, 3)
            docs.append({
                "userId": uid, "itemId": item_ids[k], "type": "recalled", "sessionId": session,
                "timestamp": placed_at + timedelta(days=1),
                "recall": {"correct": correct, "errorMeters": error, "confidence": confidence},
            })

        # a few "where is it?" questions
        for k in random.sample(range(len(ITEMS)), 2):
            docs.append({
                "userId": uid, "itemId": item_ids[k], "type": "queried", "sessionId": session,
                "timestamp": start + timedelta(hours=3),
            })

    interactions.insert_many(docs)

    print("Database:", db.name)
    for col in (users, items, locations, interactions):
        print(f"  {col.name}: {col.count_documents({})} documents")
    by_type = {d["_id"]: d["n"] for d in interactions.aggregate(
        [{"$group": {"_id": "$type", "n": {"$sum": 1}}}])}
    print("  interactions by type:", by_type)
    stats = build_model(db)
    print("Recommendation model built:", stats)
    print("Logins: researcher@memoryassist.test / admin1234,  p01..p30@memoryassist.test / test1234")


if __name__ == "__main__":
    main()
