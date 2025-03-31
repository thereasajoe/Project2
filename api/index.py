from fastapi import UploadFile, File, Form
from api.others import function_map  # ✅ not api.data — it's in others.py
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

import os
import pickle
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# from api.others import function_map, mount_student_api
import api.others


# --- Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# --- Create FastAPI app
app = FastAPI()

api.others.main_app = app  # Register main app reference

# --- Enable CORS (optional for frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Import your function map (✅ Correct this line)

# --- Load TF-IDF vectorizer
with open("api/data/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# --- Load question vectors
question_vectors = np.load("api/data/question_vectors.npy")

# --- Load function names and resolve to actual functions
with open("api/data/function_names.json", "r") as f:
    function_names = json.load(f)

question_functions = [function_map[name] for name in function_names]


# --- Main API endpoint

@app.post("/api/")
async def solve_question(
    question: str = Form(...),
    file: UploadFile = File(None)
):
    """
    Matches user's question to stored questions using TF-IDF + Cosine Similarity
    and executes the associated function.
    """
    input_vector = vectorizer.transform([question])
    similarities = cosine_similarity(input_vector, question_vectors)

    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[0, best_match_idx]

    if best_match_score < 0.5:
        return {"error": "No good match found. Please refine your question."}

    matched_function = question_functions[best_match_idx]
    print(
        f"✅ Matched to: {function_names[best_match_idx]} with score {best_match_score:.2f}")

    return matched_function(question, file)
