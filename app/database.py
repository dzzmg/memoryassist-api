"""Connection to MongoDB and creation of the indexes."""
import os

from pymongo import ASCENDING, DESCENDING, TEXT, GEO2D, MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "memoryassist")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# the four collections of the project
users = db["users"]
items = db["items"]
locations = db["locations"]
interactions = db["interactions"]
item_similarity = db["item_similarity"]   # precomputed neighbours of every item (Final Project)
recommendations = db["recommendations"]   # cached recommendation lists (TTL)
models = db["models"]                     # stored gaze model


def create_indexes(database=None):
    """Create every index the queries of the API need.

    create_index does nothing if the index already exists, so it is safe
    to call this every time the server starts.
    """
    database = database if database is not None else db
    users, items, locations, interactions = (database.users, database.items,
                                             database.locations, database.interactions)
    # users: login looks a user up by e-mail, and an e-mail must be unique
    users.create_index([("email", ASCENDING)], unique=True, name="email_unique")

    # items: full-text search on name and description (name counts more),
    # plus filters by category and colour
    items.create_index(
        [("name", TEXT), ("description", TEXT)],
        weights={"name": 3, "description": 1},
        name="item_text",
    )
    items.create_index([("category", ASCENDING)], name="item_category")
    items.create_index([("attributes.color", ASCENDING)], name="item_color")
    items.create_index([("name", ASCENDING)], unique=True, name="item_name_unique")

    # locations: x/y position on the corridor map (flat 2d coordinates)
    locations.create_index([("position", GEO2D)], name="location_position_2d")
    locations.create_index([("code", ASCENDING)], unique=True, name="location_code_unique")

    # interactions: the history of one user (newest first),
    # "where did I leave item X" (user + item + type, newest first),
    # and statistics per item (item + type)
    interactions.create_index(
        [("userId", ASCENDING), ("timestamp", DESCENDING)], name="user_history"
    )
    interactions.create_index(
        [("userId", ASCENDING), ("itemId", ASCENDING), ("type", ASCENDING), ("timestamp", DESCENDING)],
        name="user_item_type",
    )
    interactions.create_index([("itemId", ASCENDING), ("type", ASCENDING)], name="item_type")

    # ---- Final Project: indexes for the recommender ----
    # the recall results / placements of one user (recommendation request)
    interactions.create_index(
        [("userId", ASCENDING), ("type", ASCENDING), ("timestamp", ASCENDING)], name="user_type_time"
    )
    # building the user x item matrix reads only these fields -> covered query
    interactions.create_index(
        [("type", ASCENDING), ("userId", ASCENDING), ("itemId", ASCENDING), ("recall.correct", ASCENDING)],
        name="type_user_item_correct",
    )
    # cached recommendation lists expire automatically after 10 minutes
    database.recommendations.create_index([("createdAt", ASCENDING)], expireAfterSeconds=600, name="cache_ttl")
