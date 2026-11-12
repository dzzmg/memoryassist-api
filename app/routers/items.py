"""Item catalog: create, read, update, delete and search; corridor locations."""
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.errors import DuplicateKeyError

from ..database import interactions, items, locations
from ..schemas import ItemIn, ItemUpdate, LocationIn, clean, to_object_id
from ..security import require_admin

router = APIRouter()


# ---------- search (declared before /items/{item_id} so "search" is not read as an id) ----------
@router.get("/items/search", tags=["Search"])
def search_items(
    q: Optional[str] = Query(None, description="words to find in name or description"),
    category: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
):
    """Find items by text, category, colour or size. Any combination of filters works."""
    filt = {}
    if q:
        filt["$text"] = {"$search": q}
    if category:
        filt["category"] = category
    if color:
        filt["attributes.color"] = color
    if size:
        filt["attributes.size"] = size

    if q:  # sort by text relevance
        cursor = items.find(filt, {"score": {"$meta": "textScore"}}).sort([("score", {"$meta": "textScore"})])
    else:
        cursor = items.find(filt).sort("name", 1)
    return [clean(d) for d in cursor.limit(limit)]


# ---------- items CRUD ----------
@router.get("/items", tags=["Items"])
def list_items(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    return [clean(d) for d in items.find().sort("name", 1).skip(skip).limit(limit)]


@router.get("/items/{item_id}", tags=["Items"])
def get_item(item_id: str):
    doc = items.find_one({"_id": to_object_id(item_id)})
    if doc is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return clean(doc)


@router.post("/items", status_code=201, tags=["Items"])
def create_item(data: ItemIn, admin: dict = Depends(require_admin)):
    doc = data.model_dump(exclude_none=True)
    doc["createdAt"] = datetime.now(timezone.utc)
    try:
        result = items.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="An item with this name already exists")
    doc["_id"] = result.inserted_id
    return clean(doc)


@router.put("/items/{item_id}", tags=["Items"])
def update_item(item_id: str, data: ItemUpdate, admin: dict = Depends(require_admin)):
    changes = {}
    for key, value in data.model_dump(exclude_none=True).items():
        if key == "attributes":  # update only the attributes that were sent
            for attr, attr_value in value.items():
                changes[f"attributes.{attr}"] = attr_value
        else:
            changes[key] = value
    if not changes:
        raise HTTPException(status_code=400, detail="Nothing to update")
    result = items.update_one({"_id": to_object_id(item_id)}, {"$set": changes})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return clean(items.find_one({"_id": to_object_id(item_id)}))


@router.delete("/items/{item_id}", tags=["Items"])
def delete_item(item_id: str, admin: dict = Depends(require_admin)):
    oid = to_object_id(item_id)
    if interactions.count_documents({"itemId": oid}, limit=1):
        raise HTTPException(status_code=409, detail="Item has recorded interactions and cannot be deleted")
    result = items.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"deleted": item_id}


# ---------- locations ----------
@router.get("/locations", tags=["Locations"])
def list_locations():
    return [clean(d) for d in locations.find().sort("code", 1)]


@router.post("/locations", status_code=201, tags=["Locations"])
def create_location(data: LocationIn, admin: dict = Depends(require_admin)):
    doc = data.model_dump()
    try:
        result = locations.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="A location with this code already exists")
    doc["_id"] = result.inserted_id
    return clean(doc)


@router.get("/locations/near", tags=["Locations"])
def locations_near(x: float, y: float, limit: int = Query(3, ge=1, le=20)):
    """Nearest corridor locations to a point (uses the 2d index)."""
    return [clean(d) for d in locations.find({"position": {"$near": [x, y]}}).limit(limit)]
