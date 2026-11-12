# MemoryAssist API documentation

Base URL: `http://127.0.0.1:8000`. Interactive version: `/docs` (Swagger UI) and `/redoc`.

All request and response bodies are JSON, except `/auth/login`, which takes a form.
Protected endpoints need the header `Authorization: Bearer <token>`, where the token comes from `/auth/login`.
Access levels: **public** (no token), **user** (any logged-in user), **admin** (role `admin`, the researcher).

## Users and authentication

### POST /auth/register (public)
Creates a participant account. The password is stored as a bcrypt hash.

Request:
```json
{
  "email": "dilnaz@test.kz",
  "password": "secret123",
  "profile": {"displayName": "Dilnaz", "ageGroup": "22-25", "fieldOfStudy": "Computer Science",
              "wearsGlasses": false, "memorySelfRating": 4}
}
```
`password` needs at least 8 characters. All profile fields are optional. `ageGroup` is one of
`18-21`, `22-25`, `26-30`, `31+`; `memorySelfRating` is 1-5.

Responses: `201` the new user (without the password), `409` e-mail already registered, `422` invalid data.

### POST /auth/login (public)
Form fields (`application/x-www-form-urlencoded`): `username` = e-mail, `password`.

Response `200`:
```json
{"access_token": "eyJhbGciOi...", "token_type": "bearer"}
```
`401` wrong e-mail or password. The token is valid for 120 minutes.

### GET /users/me (user)
Returns the logged-in user's account and profile.

### PUT /users/me (user)
Updates only the profile fields that are sent.
```json
{"profile": {"fieldOfStudy": "Robotics", "memorySelfRating": 3}}
```

## Items (catalog)

Item fields: `name` (unique), `description`, `category`, `attributes.color`,
`attributes.size` (`small`, `medium`, `large`), `attributes.salience` (1-5, how noticeable the item is).

### GET /items (public)
List of items sorted by name. Query parameters: `skip` (default 0), `limit` (default 50, max 200).

### GET /items/{id} (public)
One item. `404` if not found, `400` if the id is not a valid ObjectId.

### POST /items (admin)
```json
{"name": "Glasses case", "description": "Hard case for glasses", "category": "accessories",
 "attributes": {"color": "black", "size": "medium", "salience": 4}}
```
Responses: `201` the new item, `403` not an admin, `409` name already exists.

### PUT /items/{id} (admin)
Updates only the fields that are sent; inside `attributes`, only the attributes that are sent.
```json
{"attributes": {"color": "brown"}}
```

### DELETE /items/{id} (admin)
`200` `{"deleted": "<id>"}`. `409` if the item already has interactions (the history is kept), `404` if not found.

## Search

### GET /items/search (public)
Query parameters, all optional and combinable:
`q` (words in name or description, full-text index, results sorted by relevance),
`category`, `color`, `size`, `limit` (default 20).

Examples: `/items/search?q=card`, `/items/search?category=stationery&color=red`.

### GET /memory/where?item=Key (user)
"Where did I leave my ...?" Finds the item by name (not case-sensitive) and returns the last place
where the logged-in user placed it. The question is saved as a `queried` interaction.
```json
{"item": "Key", "found": true,
 "location": {"code": "L03", "name": "Window sill by the stairs", "zone": "stairs", "position": [12.0, 2.8]},
 "placedAt": "2026-11-02T10:12:00", "gaze": {"dwellMs": 1650, "fixations": 6, "revisits": 1}}
```

## Interactions (user history)

### POST /interactions (user)
Records an interaction of the logged-in user. `type` is `placed`, `recalled` or `queried`.

Placed (needs `locationId` and `gaze`):
```json
{"type": "placed", "itemId": "<item id>", "locationId": "<location id>",
 "gaze": {"dwellMs": 1500, "fixations": 5, "revisits": 1}}
```
Recalled (needs `recall`):
```json
{"type": "recalled", "itemId": "<item id>",
 "recall": {"correct": false, "errorMeters": 7.5, "confidence": 2}}
```
Optional for all: `sessionId`, `timestamp` (default: now).
Responses: `201` the saved interaction, `404` item or location not found, `422` missing fields.

### GET /users/me/history (user)
The user's interactions, newest first, with the item name. Query parameters: `type`, `limit` (default 50).

## Locations

### GET /locations (public)
The 10 places along the Block 3 corridor with `[x, y]` positions in metres.

### POST /locations (admin)
```json
{"code": "L11", "name": "Bench by room 3.120", "zone": "far end", "position": [55.0, 1.0]}
```

### GET /locations/near?x=20&y=1 (public)
The nearest locations to a point (uses the 2d index). `limit` default 3.

## Analytics

### GET /analytics/forgetting (admin)
For every item: number of recall attempts, number forgotten, forget rate and average map error,
sorted from the most forgotten. This is the starting point for the recommender in the Final Project.

## Recommendations (Final Project)

The recommender ranks the items a user is most likely to forget.
score = 0.25 x item-based collaborative filtering + 0.75 x gaze risk (logistic regression).

### GET /recommendations/me?k=5 (user)
```json
{"userId": "...", "cached": false, "computeMs": 9.2,
 "recommendations": [
   {"itemId": "...", "item": "USB flash drive", "category": "electronics", "score": 0.93,
    "cfScore": 0.96, "gazeRisk": 0.92, "itemForgetRate": 0.667,
    "becauseYouForgot": ["Pen", "Paper clip box"], "lastPlacedAt": "...", "locationId": "..."}]}
```
`k` is 1-20 (default 5). The list is cached for 10 minutes in the `recommendations` collection
(TTL index); `cached` shows whether it came from the cache. Recording a `placed` or `recalled`
interaction clears the user's cached list.

### GET /recommendations/user/{user_id}?k=5 (admin)
The same for any user.

### GET /items/{item_id}/similar (public)
Items that are forgotten by the same people (item-item similarity, top 5).

### POST /recommendations/rebuild (admin)
Recomputes the item similarities and the gaze model and clears the cache.
Returns `{"users": 30, "items": 12, "recalls": 360, "seconds": 0.09}`.

### GET /recommendations/model (admin)
The gaze model weights, when the similarities were built, the number of cached lists, alpha and K.

## Web interface

`/app/` - login and registration, reminders (recommendations), "where is my item", logging a
placement, catalog search with similar items, history, and a research dashboard for the admin.

## Error format
Errors return a status code and `{"detail": "message"}`.
