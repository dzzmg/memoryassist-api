"""Registration, login and the user's own profile."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pymongo.errors import DuplicateKeyError

from ..database import users
from ..schemas import ProfileUpdate, RegisterIn, clean
from ..security import create_token, get_current_user, hash_password, verify_password

router = APIRouter(tags=["Users and authentication"])


@router.post("/auth/register", status_code=201)
def register(data: RegisterIn):
    """Create a participant account. The password is stored only as a bcrypt hash."""
    doc = {
        "email": data.email.lower(),
        "passwordHash": hash_password(data.password),
        "role": "participant",
        "profile": data.profile.model_dump(exclude_none=True),
        "createdAt": datetime.now(timezone.utc),
    }
    try:
        result = users.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="This e-mail is already registered")
    doc["_id"] = result.inserted_id
    return clean(doc)


@router.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    """Log in with e-mail (in the username field) and password. Returns a Bearer token."""
    user = users.find_one({"email": form.username.lower()})
    if user is None or not verify_password(form.password, user["passwordHash"]):
        raise HTTPException(status_code=401, detail="Wrong e-mail or password")
    token = create_token(str(user["_id"]), user["role"])
    return {"access_token": token, "token_type": "bearer"}


@router.get("/users/me")
def read_me(user: dict = Depends(get_current_user)):
    return clean(user)


@router.put("/users/me")
def update_me(data: ProfileUpdate, user: dict = Depends(get_current_user)):
    """Update only the profile fields that were sent."""
    changes = {f"profile.{k}": v for k, v in data.profile.model_dump(exclude_none=True).items()}
    if changes:
        users.update_one({"_id": user["_id"]}, {"$set": changes})
    return clean(users.find_one({"_id": user["_id"]}))
