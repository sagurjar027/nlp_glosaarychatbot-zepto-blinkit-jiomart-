from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

# --- Chatbot Logic Class ---
class ReviewChatbot:
    def __init__(self, csv_file):
        print("🤖 Initializing Chatbot Model...")
        self.data = pd.read_csv(csv_file)
        self.data = self.data.dropna(subset=['review', 'sentiment'])
        self.data['clean_review'] = self.data['review'].apply(self._preprocess)
        
        # Train Sentiment Model with richer features
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=20000,
            min_df=3
        )
        classifier = LogisticRegression(
            max_iter=200,
            class_weight='balanced',
            solver="lbfgs"
        )
        self.sentiment_model: Pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", classifier),
        ])
        self.sentiment_model.fit(self.data['clean_review'], self.data['sentiment'])
        
        # Prepare Search Index
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=20000,
            min_df=2
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['clean_review'])
        print("✅ Models trained and ready.")

    def _preprocess(self, text: str) -> str:
        """Normalize review text for better model quality."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def analyze(self, text):
        cleaned_text = self._preprocess(text)
        if not cleaned_text:
            raise ValueError("Empty message after cleaning.")

        # 1. Predict Sentiment
        prediction = self.sentiment_model.predict([cleaned_text])[0]
        proba = np.max(self.sentiment_model.predict_proba([cleaned_text]))
        
        # 2. Find Similar Reviews
        query_vec = self.vectorizer.transform([cleaned_text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_indices = similarities.argsort()[-2:][::-1] # Get top 2
        
        matches = []
        for idx in related_indices:
            if similarities[idx] > 0.1: # Relevance threshold
                matches.append({
                    'text': self.data.iloc[idx]['review'],
                    'sentiment': self.data.iloc[idx]['sentiment'],
                    'platform': self.data.iloc[idx]['platform'],
                    'score': float(similarities[idx])
                })
                
        return {
            "sentiment": prediction,
            "confidence": float(proba),
            "matches": matches
        }

# --- FastAPI App Setup ---
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "reviews (2).csv"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()

# Enable CORS (allows frontend to talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (CSS/JS)
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

# Global chatbot instance
chatbot = None

@app.on_event("startup")
def startup_event():
    global chatbot
    try:
        # Make sure reviews (2).csv is in the same folder
        chatbot = ReviewChatbot(DATA_FILE)
    except Exception as e:
        print(f"Error loading model: {e}")

class UserInput(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(input_data: UserInput):
    if not chatbot:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        result = chatbot.analyze(input_data.message)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

# Serve the index.html file at the root URL
@app.get("/")
async def read_index():
    return FileResponse(BASE_DIR / "index.html")