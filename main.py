import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response

# ===================== NLP Suggestion Engine (NLTK) =====================

from functools import lru_cache

class SuggestRequest(BaseModel):
    text: str
    language: str = "en"
    limit: int = 5

class Suggestion(BaseModel):
    text: str
    score: float

class SuggestResponse(BaseModel):
    suggestions: List[Suggestion]

@lru_cache(maxsize=1)
def _init_nltk() -> Dict[str, Any]:
    import nltk
    # Try to load, else download quietly
    try:
        from nltk.corpus import words as nltk_words
    except LookupError:  # pragma: no cover
        nltk.download('words')
        from nltk.corpus import words as nltk_words
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:  # pragma: no cover
        nltk.download('punkt')
    vocab = set(w.lower() for w in nltk_words.words())
    return {"vocab": vocab}

# Simple Damerau-Levenshtein approximation using NLTK's edit_distance

def _score_candidates(token: str, vocab: List[str], limit: int) -> List[Dict[str, Any]]:
    from nltk.metrics.distance import edit_distance
    t = token.lower()
    if not t:
        return []
    # Heuristic prefilter to keep it fast
    pref = t[0]
    candidates = [w for w in vocab if abs(len(w) - len(t)) <= 2 and w[0] == pref]
    # If too few, relax prefix filter
    if len(candidates) < 200:
        candidates = list({*candidates, *[w for w in vocab if abs(len(w) - len(t)) <= 1]})
    scored = []
    for w in candidates[:5000]:
        d = edit_distance(t, w, substitution_cost=1)
        # Convert distance to similarity score in [0,1]
        m = max(len(t), len(w)) or 1
        score = 1 - (d / m)
        if score > 0.4:  # threshold to remove very dissimilar words
            scored.append({"text": w, "score": float(score)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for s in scored:
        if s["text"] not in seen and s["text"] != t:
            seen.add(s["text"])
            uniq.append(s)
        if len(uniq) >= limit:
            break
    return uniq

@app.post("/api/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    data = _init_nltk()
    vocab = data["vocab"]
    # Use last token of the text for suggestions
    import re
    tokens = re.split(r"\s+", req.text.strip()) if req.text else []
    last = tokens[-1] if tokens else ""
    suggestions = _score_candidates(last, vocab, req.limit)
    return {"suggestions": suggestions}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
