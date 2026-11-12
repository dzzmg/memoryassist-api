# MemoryAssist API

Memory recommendation platform for AR glasses (Assignment #6 and Final Project, Advanced Databases).
Users (glasses wearers) place everyday items, the glasses record gaze, and a next-day recall test
shows which items were forgotten. The data is stored in MongoDB and served by a FastAPI REST API.
The Final Project adds an item-based collaborative filtering recommender (which items will this user
forget?), a web interface, an offline evaluation and a performance test.

## Requirements

- Python 3.10 or newer (`python3 --version`)
- MongoDB running locally on port 27017 (MongoDB Community Server; Compass is useful for viewing the data)
- Postman (optional, for testing)

## Project structure

```
memoryassist-api/
  app/
    main.py              FastAPI app, creates the indexes on start
    database.py          MongoDB connection, collections and indexes
    security.py          password hashing (bcrypt) and JWT tokens
    schemas.py           validation models for requests
    routers/
      auth.py            register, login, profile
      items.py           item catalog CRUD, search, locations
      interactions.py    user history, "where is my item", analytics
      recommendations.py recommendations API (Final Project)
    recommender.py       collaborative filtering + gaze model, caching (Final Project)
    static/index.html    the web interface (Final Project)
  seed.py                fills the database with synthetic test data and builds the model
  evaluate.py            offline evaluation of recommendation quality (Final Project)
  perf_test.py           performance test as the database grows (Final Project)
  requirements.txt
  API.md                 description of every endpoint
  docs/Project_Report.pdf  project report (design, indexes, evaluation, performance)
```

## Setup (macOS)

```bash
git clone https://github.com/<your-username>/memoryassist-api.git
cd memoryassist-api
python3 -m venv venv              # create a virtual environment (only the first time)
source venv/bin/activate          # activate it (every new terminal)
pip install --isolated -r requirements.txt   # install the libraries (again after updating the project)
```

Make sure MongoDB is running. If you installed it with Homebrew: `brew services start mongodb-community`.

## Load the test data

```bash
python seed.py
```

This deletes the old data of the project and creates 31 users, 12 items, 10 corridor locations
and 780 interactions (360 placed, 360 recalled, 60 queried), and builds the recommendation model.

Test logins:

| Role | E-mail | Password |
|------|--------|----------|
| researcher (admin) | researcher@memoryassist.test | admin1234 |
| participants | p01@memoryassist.test ... p30@memoryassist.test | test1234 |

## Run the server

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/app/ for the web interface and http://127.0.0.1:8000/docs for the
interactive API documentation (Swagger UI).
Click **Authorize**, enter an e-mail as username and the password, and the protected endpoints can be tested.

## Evaluation and performance test

```bash
python evaluate.py      # recommendation quality: AUC, Precision@3, Recall@3 -> results/evaluation.csv
python perf_test.py     # speed as the database grows -> results/performance.csv and charts
```

`perf_test.py` uses a separate database, `memoryassist_perf`, and deletes it at the end.
Use `python perf_test.py --quick` for a short run.

## Settings

The defaults work for a local MongoDB. To change them, set environment variables before starting
(see `.env.example`): `MONGO_URI`, `DB_NAME`, `JWT_SECRET`, `TOKEN_MINUTES`.
Change `JWT_SECRET` to a long random string if the API is used outside a local test.
