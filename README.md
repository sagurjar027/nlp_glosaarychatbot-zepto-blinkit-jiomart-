# nlp_glosaarychatbot-zepto-blinkit-jiomart-
🌟 AI Review Insights

A lightweight FastAPI + JavaScript web application that analyzes customer reviews for platforms like Zepto, Blinkit, and JioMart.

The system can:

Predict sentiment (positive / neutral / negative)

Extract top keywords

Find similar reviews from the dataset

Display confidence scores

Provide platform-wise insights

Offer an interactive UI (HTML + CSS + JS)

This project is ideal for NLP learning, customer support automation, and AI-powered review analysis.

🚀 Features
✅ 1. FastAPI-powered machine learning backend

Loads a preprocessed dataset

Uses TF-IDF + Naive Bayes / Logistic Regression

Offers /analyze endpoint

Returns predictions in under 300ms

✅ 2. Beautiful front-end (HTML + JS + CSS)

Modern dark UI

Real-time review analysis

Shows sentiment, keywords, similar reviews

✅ 3. Colab Training Notebook

The entire ML workflow is trained in Colab:

Read + clean dataset

Build TF-IDF vectorizer

Train Naive Bayes sentiment model

Compute embeddings for similarity search

Export pickle files for FastAPI

⭐ You can easily switch to BERT / HuggingFace models later!
📁 Project Structure
/
├── app.py                # FastAPI backend
├── static/
│   ├── index.html        # Frontend UI
│   ├── app.js            # Calls the API + renders results
│   ├── styles.css        # UI styling
│
├── reviews.csv           # Dataset used by the API
├── requirements.txt      # All dependencies
└── README.md             # Documentation

🧠 How It Works
🔧 Backend (FastAPI)

Accepts raw text from the frontend

Cleans + vectorizes the text

Predicts sentiment using trained ML model
Finds similar reviews using cosine similarity
Sends JSON response to UI
🎨 Front-end
Sends request using fetch() (AJAX)
Displays:
Sentiment label
Confidence
Similar reviews list
Platform-wise keywords

📦 Installation & Setup
1️⃣ Clone the project
git clone https://github.com/sagurjar027/review-insights.git
cd review-insights

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Start FastAPI server
uvicorn app:app --reload

4️⃣ Open frontend
Go to:
http://127.0.0.1:8000/static/index.html

📝 API Endpoint
POST /analyze
Request body:
{
  "text": "worst they have worst payment gateway ever"
}

Response:
{
  "sentiment": "negative",
  "confidence": 0.879,
  "similar_reviews": [
    {
      "text": "worst payment gateway…",
      "score": 0.48,
      "platform": "zepto"
    }
  ],
  "keywords": ["worst", "payment", "order", "refund"]
}

🎓 Training Notebook (Colab Workflow)
The Colab notebook includes:
Dataset loading
Text preprocessing
Sentiment model training
TF-IDF vectorizer training
Cosine similarity index creation

🎯 Future Enhancements
Upgrade sentiment model → DistilBERT, BERT, or RoBERTa
Add topic modeling (BERTopic)
Add multi-platform comparison dashboard
Deploy to cloud (Railway, Vercel, Render, HuggingFace Spaces)
🧑‍💻 Technologies Used
Backend
Python
FastAPI
Scikit-learn (Naive Bayes, TF-IDF)
Sentence Transformers (optional)
Frontend
HTML
CSS
Vanilla JS
