"""Shapes of the data the API receives (validated by Pydantic)."""
from datetime import datetime
from typing import Literal, Optional

from bson import ObjectId
from fastapi import HTTPException
from pydantic import BaseModel, EmailStr, Field


# ---------- helpers ----------
def to_object_id(value: str) -> ObjectId:
    if not ObjectId.is_valid(value):
        raise HTTPException(status_code=400, detail=f"Invalid id: {value}")
    return ObjectId(value)


def clean(doc: Optional[dict]) -> Optional[dict]:
    """Turn a MongoDB document into JSON: ObjectId -> str, never return the password hash."""
    if doc is None:
        return None
    out = {}
    for key, value in doc.items():
        if key == "passwordHash":
            continue
        if key == "_id":
            out["id"] = str(value)
        elif isinstance(value, ObjectId):
            out[key] = str(value)
        elif isinstance(value, dict):
            out[key] = clean(value)
        else:
            out[key] = value
    return out


# ---------- users ----------
class Profile(BaseModel):
    displayName: Optional[str] = None
    ageGroup: Optional[Literal["18-21", "22-25", "26-30", "31+"]] = None
    fieldOfStudy: Optional[str] = None
    wearsGlasses: Optional[bool] = None
    memorySelfRating: Optional[int] = Field(None, ge=1, le=5, description="1 = poor, 5 = excellent")


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    profile: Profile = Profile()


class ProfileUpdate(BaseModel):
    profile: Profile


# ---------- items ----------
class ItemAttributes(BaseModel):
    color: Optional[str] = None
    size: Optional[Literal["small", "medium", "large"]] = None
    salience: Optional[int] = Field(None, ge=1, le=5, description="how noticeable the item is")


class ItemIn(BaseModel):
    name: str
    description: str
    category: str
    attributes: ItemAttributes = ItemAttributes()


class ItemUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    attributes: Optional[ItemAttributes] = None


# ---------- locations ----------
class LocationIn(BaseModel):
    code: str
    name: str
    zone: str
    position: list[float] = Field(min_length=2, max_length=2, description="[x, y] in metres on the corridor map")


# ---------- interactions ----------
class Gaze(BaseModel):
    dwellMs: int = Field(ge=0, description="total time looking at the item, ms")
    fixations: int = Field(ge=0, description="number of fixations on the item")
    revisits: int = Field(ge=0, description="how many times the person looked back at it later")


class Recall(BaseModel):
    correct: bool
    errorMeters: float = Field(ge=0, description="distance between marked and real place")
    confidence: int = Field(ge=1, le=5)


class InteractionIn(BaseModel):
    type: Literal["placed", "recalled", "queried"]
    itemId: str
    locationId: Optional[str] = None      # required for "placed"
    gaze: Optional[Gaze] = None           # required for "placed"
    recall: Optional[Recall] = None       # required for "recalled"
    sessionId: Optional[str] = None
    timestamp: Optional[datetime] = None  # default: now
