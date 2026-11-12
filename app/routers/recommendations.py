"""Recommendations API (Final Project)."""
from fastapi import APIRouter, Depends, HTTPException, Query

from .. import recommender
from ..database import db, items, users
from ..schemas import clean, to_object_id
from ..security import get_current_user, require_admin

router = APIRouter(tags=["Recommendations"])


@router.get("/recommendations/me")
def my_recommendations(k: int = Query(5, ge=1, le=20), user: dict = Depends(get_current_user)):
    """Items the logged-in user is most likely to forget (reminders), best first.

    score = 0.25 * collaborative filtering + 0.75 * gaze risk (weight chosen by cross-validation).
    The list is cached for 10 minutes; `cached` and `computeMs` show whether it came from the cache.
    """
    recs, cached, ms = recommender.cached_recommendations(db, user["_id"], k)
    return {"userId": str(user["_id"]), "cached": cached, "computeMs": ms, "recommendations": recs}


@router.get("/recommendations/user/{user_id}")
def user_recommendations(user_id: str, k: int = Query(5, ge=1, le=20), admin: dict = Depends(require_admin)):
    """Recommendations for any user (researcher only)."""
    oid = to_object_id(user_id)
    if users.count_documents({"_id": oid}, limit=1) == 0:
        raise HTTPException(status_code=404, detail="User not found")
    recs, cached, ms = recommender.cached_recommendations(db, oid, k)
    return {"userId": user_id, "cached": cached, "computeMs": ms, "recommendations": recs}


@router.get("/items/{item_id}/similar")
def similar_items(item_id: str):
    """Items that tend to be forgotten by the same people (item-item similarity)."""
    recommender.ensure_model(db)
    doc = db.item_similarity.find_one({"_id": to_object_id(item_id)})
    if doc is None:
        raise HTTPException(status_code=404, detail="Item not found")
    names = {d["_id"]: d["name"] for d in items.find({}, {"name": 1})}
    return {
        "item": names.get(doc["_id"]),
        "forgetRate": doc["forgetRate"],
        "similar": [{"itemId": str(n["itemId"]), "item": names.get(n["itemId"]), "similarity": n["sim"]}
                    for n in doc["neighbours"]],
    }


@router.post("/recommendations/rebuild")
def rebuild(admin: dict = Depends(require_admin)):
    """Recompute the item similarities and the gaze model, and clear the cache (researcher only)."""
    return recommender.build_model(db)


@router.get("/recommendations/model")
def model_info(admin: dict = Depends(require_admin)):
    """The trained gaze model and when the model was last built."""
    m = db.models.find_one({"_id": "gaze_lr"})
    sim = db.item_similarity.find_one(sort=[("updatedAt", -1)])
    return {
        "gazeModel": clean(m) if m else None,
        "similarityUpdatedAt": sim["updatedAt"] if sim else None,
        "items": db.item_similarity.count_documents({}),
        "cachedLists": db.recommendations.count_documents({}),
        "alpha": recommender.ALPHA,
        "neighbours": recommender.NEIGHBOURS,
    }
