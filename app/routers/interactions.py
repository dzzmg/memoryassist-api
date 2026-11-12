"""User history: placing items, recalling them, asking where they are."""
import re
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from .. import recommender
from ..database import db, interactions, items, locations
from ..schemas import InteractionIn, clean, to_object_id
from ..security import get_current_user, require_admin

router = APIRouter()


@router.post("/interactions", status_code=201, tags=["Interactions"])
def record_interaction(data: InteractionIn, user: dict = Depends(get_current_user)):
    """Record one interaction of the logged-in user with an item.

    placed   - needs locationId and gaze (dwellMs, fixations, revisits)
    recalled - needs recall (correct, errorMeters, confidence)
    queried  - the user asked where the item is
    """
    item_oid = to_object_id(data.itemId)
    if items.count_documents({"_id": item_oid}, limit=1) == 0:
        raise HTTPException(status_code=404, detail="Item not found")

    doc = {
        "userId": user["_id"],
        "itemId": item_oid,
        "type": data.type,
        "timestamp": data.timestamp or datetime.now(timezone.utc),
    }
    if data.sessionId:
        doc["sessionId"] = data.sessionId

    if data.type == "placed":
        if not data.locationId or not data.gaze:
            raise HTTPException(status_code=422, detail="'placed' needs locationId and gaze")
        loc_oid = to_object_id(data.locationId)
        if locations.count_documents({"_id": loc_oid}, limit=1) == 0:
            raise HTTPException(status_code=404, detail="Location not found")
        doc["locationId"] = loc_oid
        doc["gaze"] = data.gaze.model_dump()
    elif data.type == "recalled":
        if not data.recall:
            raise HTTPException(status_code=422, detail="'recalled' needs recall")
        doc["recall"] = data.recall.model_dump()

    result = interactions.insert_one(doc)
    doc["_id"] = result.inserted_id
    if data.type in ("placed", "recalled"):
        recommender.invalidate(db, user["_id"])  # this user's cached recommendations are now outdated
    return clean(doc)


@router.get("/users/me/history", tags=["Interactions"])
def my_history(
    type: Optional[Literal["placed", "recalled", "queried"]] = None,
    limit: int = Query(50, ge=1, le=500),
    user: dict = Depends(get_current_user),
):
    """The logged-in user's interactions, newest first, with item names."""
    match = {"userId": user["_id"]}
    if type:
        match["type"] = type
    pipeline = [
        {"$match": match},
        {"$sort": {"timestamp": -1}},
        {"$limit": limit},
        {"$lookup": {"from": "items", "localField": "itemId", "foreignField": "_id", "as": "item"}},
        {"$unwind": "$item"},
        {"$addFields": {"itemName": "$item.name"}},
        {"$project": {"item": 0}},
    ]
    return [clean(d) for d in interactions.aggregate(pipeline)]


@router.get("/memory/where", tags=["Search"])
def where_is(item: str = Query(..., description="item name, e.g. 'keys'"), user: dict = Depends(get_current_user)):
    """'Where did I leave my ...?' - returns the last place where this user put the item.

    The question itself is saved as a 'queried' interaction.
    """
    item_doc = items.find_one({"name": {"$regex": f"^{re.escape(item)}$", "$options": "i"}}) or \
        items.find_one({"$text": {"$search": item}})
    if item_doc is None:
        raise HTTPException(status_code=404, detail="No such item")

    last = interactions.find_one(
        {"userId": user["_id"], "itemId": item_doc["_id"], "type": "placed"},
        sort=[("timestamp", -1)],
    )
    interactions.insert_one({
        "userId": user["_id"], "itemId": item_doc["_id"], "type": "queried",
        "timestamp": datetime.now(timezone.utc),
    })
    if last is None:
        return {"item": item_doc["name"], "found": False, "message": "You have not placed this item yet"}

    loc = locations.find_one({"_id": last["locationId"]})
    return {
        "item": item_doc["name"],
        "found": True,
        "location": clean(loc),
        "placedAt": last["timestamp"],
        "gaze": last.get("gaze"),
    }


@router.get("/analytics/forgetting", tags=["Analytics"])
def forgetting_by_item(admin: dict = Depends(require_admin)):
    """Share of recall attempts that failed, per item (basis for the Final Project recommender)."""
    pipeline = [
        {"$match": {"type": "recalled"}},
        {"$group": {
            "_id": "$itemId",
            "attempts": {"$sum": 1},
            "forgotten": {"$sum": {"$cond": ["$recall.correct", 0, 1]}},
            "avgErrorMeters": {"$avg": "$recall.errorMeters"},
        }},
        {"$lookup": {"from": "items", "localField": "_id", "foreignField": "_id", "as": "item"}},
        {"$unwind": "$item"},
        {"$project": {
            "_id": 0,
            "item": "$item.name",
            "attempts": 1,
            "forgotten": 1,
            "forgetRate": {"$round": [{"$divide": ["$forgotten", "$attempts"]}, 3]},
            "avgErrorMeters": {"$round": ["$avgErrorMeters", 2]},
        }},
        {"$sort": {"forgetRate": -1}},
    ]
    return list(interactions.aggregate(pipeline))
