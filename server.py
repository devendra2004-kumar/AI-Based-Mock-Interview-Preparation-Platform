# =============================================================================
#  server.py  —  PrepAI Backend
#  Run:  uvicorn server:app --reload --port 8000
#  Deps: pip install fastapi uvicorn google-generativeai
# =============================================================================

import os
import json
import random
import re
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ── Gemini setup ──────────────────────────────────────────────────────────────
# Set your Gemini API key here OR as env var GEMINI_API_KEY
# Get a free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyChKEsWKQ1NOyEORCLVCskUFNdcHIuvRLc")

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini = genai.GenerativeModel("gemini-1.5-flash")
    AI_AVAILABLE = GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE"
except ImportError:
    gemini = None
    AI_AVAILABLE = False
    print("⚠  google-generativeai not installed. Running in fallback mode.")
    print("   pip install google-generativeai")

# ── Load questions.json ───────────────────────────────────────────────────────
QUESTIONS_FILE = Path(__file__).parent / "questions.json"
if QUESTIONS_FILE.exists():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        QUESTION_BANK: dict = json.load(f)
else:
    QUESTION_BANK = {}
    print("⚠  questions.json not found. Place it in the same folder as server.py")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="PrepAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow the HTML file opened directly in browser
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
#  PYDANTIC SCHEMAS  (exactly what the frontend sends / expects)
# =============================================================================

class JDRequest(BaseModel):
    text: str                          # raw job description pasted by user

class JDResponse(BaseModel):
    role: str                          # detected role name e.g. "Software Development Engineer"
    role_id: str                       # mapped key e.g. "sde"
    seniority: str                     # "Entry Level" / "Mid Level" / "Senior"
    confidence: float                  # 0.0 – 1.0
    skills: List[str]                  # ["Python", "REST APIs", ...]

# ---------------------------------------------------------------------------- #

class QuestionsRequest(BaseModel):
    role: str                          # role display name
    role_id: str                       # role key (matches questions.json keys)
    count: int = 5                     # 5 / 10 / 15
    difficulty: str = "entry"          # "entry" / "mid" / "senior"
    types: List[str] = ["technical", "behavioral", "situational"]

class Question(BaseModel):
    question: str
    type: str                          # "technical" / "behavioral" / "situational"
    difficulty: str
    key_points: List[str]              # used as hints + evaluation reference

class QuestionsResponse(BaseModel):
    questions: List[Question]

# ---------------------------------------------------------------------------- #

class EvalRequest(BaseModel):
    question: str                      # the question text
    answer: str                        # user's answer
    key_points: List[str]              # expected key points from questions.json
    type: str                          # question type

class EvalResponse(BaseModel):
    score: int                         # 0 – 100
    feedback: str                      # 1-2 sentence qualitative feedback
    ideal_answer: str                  # brief model answer
    missing_points: List[str]          # key points the user missed

# =============================================================================
#  ROLE MAPPING  —  maps Gemini-detected role names → questions.json keys
# =============================================================================

ROLE_MAP = {
    # tech
    "sde": ["sde", "software", "developer", "engineer", "programming", "coding"],
    "frontend": ["frontend", "front-end", "front end", "react", "vue", "angular", "ui developer"],
    "backend": ["backend", "back-end", "back end", "api", "server", "node", "django", "flask"],
    "devops": ["devops", "dev ops", "cloud", "aws", "gcp", "azure", "infrastructure", "sre", "platform"],
    "cybersec": ["security", "cyber", "infosec", "penetration", "soc analyst", "network security"],
    "ml": ["machine learning", "ml engineer", "ai engineer", "deep learning", "data science engineer", "nlp"],
    "mobile": ["mobile", "ios", "android", "flutter", "react native"],
    # data
    "da": ["data analyst", "analytics", "business intelligence", "bi analyst", "tableau", "power bi"],
    "ds": ["data scientist", "scientist", "statistician", "research scientist"],
    "de": ["data engineer", "etl", "pipeline", "spark", "kafka", "airflow"],
    # business
    "ba": ["business analyst", "requirements", "process analyst", "functional analyst"],
    "hr": ["hr", "human resources", "people", "talent", "recruiter", "hrbp"],
    "pm": ["product manager", "product owner", "po", "program manager"],
    "marketing": ["marketing", "growth", "seo", "content marketing", "digital marketing", "brand"],
    "sales": ["sales", "account executive", "business development", "bdm", "account manager"],
    "finance": ["finance", "financial analyst", "fp&a", "investment", "accounting"],
    # design
    "ux": ["ux", "ui", "designer", "user experience", "user interface", "figma", "product design"],
    "content": ["content writer", "copywriter", "technical writer", "documentation", "content creator"],
}

def map_role_to_id(role_text: str) -> str:
    """Map a free-text role name to the questions.json key."""
    r = role_text.lower()
    for role_id, keywords in ROLE_MAP.items():
        if any(kw in r for kw in keywords):
            return role_id
    return "sde"   # sensible fallback

# =============================================================================
#  HELPER — call Gemini safely
# =============================================================================

def ask_gemini(prompt: str, fallback: dict) -> dict:
    """
    Call Gemini and parse the JSON response.
    Returns fallback dict if AI is unavailable or response is malformed.
    """
    if not AI_AVAILABLE or not gemini:
        return fallback

    try:
        response = gemini.generate_content(
            prompt,
            generation_config={"temperature": 0.4, "max_output_tokens": 800}
        )
        raw = response.text.strip()

        # Strip markdown code fences if Gemini wraps the JSON
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()

        return json.loads(raw)

    except json.JSONDecodeError:
        print("⚠  Gemini returned non-JSON. Using fallback.")
        return fallback
    except Exception as e:
        print(f"⚠  Gemini error: {e}. Using fallback.")
        return fallback

# =============================================================================
#  ROUTE 1  —  POST /parse-jd
#  Frontend sends: { text: "..." }
#  Returns: JDResponse
# =============================================================================

@app.post("/parse-jd", response_model=JDResponse)
async def parse_jd(req: JDRequest):
    if len(req.text.strip()) < 30:
        raise HTTPException(status_code=400, detail="Job description too short (min 30 chars)")

    # ── Gemini prompt ─────────────────────────────────────────────────────────
    prompt = f"""
You are a job description analyser. Read the job description below and return ONLY a valid JSON object.
No explanation, no markdown, no extra text — ONLY the raw JSON.

Return this exact structure:
{{
  "role": "<job title as a clean string, e.g. Software Development Engineer>",
  "seniority": "<one of: Entry Level, Mid Level, Senior>",
  "confidence": <float between 0.7 and 1.0>,
  "skills": ["skill1", "skill2", "skill3", "skill4", "skill5"]
}}

Job Description:
{req.text[:2000]}
"""

    fallback = {
        "role": "Software Development Engineer",
        "seniority": "Mid Level",
        "confidence": 0.80,
        "skills": ["Python", "Problem Solving", "Data Structures", "Algorithms", "System Design"]
    }

    result = ask_gemini(prompt, fallback)

    # Sanitise + fill gaps
    role_name  = str(result.get("role", fallback["role"]))
    seniority  = str(result.get("seniority", fallback["seniority"]))
    confidence = float(result.get("confidence", 0.80))
    skills     = result.get("skills", fallback["skills"])

    if not isinstance(skills, list):
        skills = fallback["skills"]

    role_id = map_role_to_id(role_name)

    return JDResponse(
        role=role_name,
        role_id=role_id,
        seniority=seniority,
        confidence=min(max(confidence, 0.0), 1.0),
        skills=[str(s) for s in skills[:8]]
    )

# =============================================================================
#  ROUTE 2  —  POST /questions
#  Frontend sends: { role, role_id, count, difficulty, types }
#  Returns: { questions: [ {question, type, difficulty, key_points}, ... ] }
# =============================================================================

@app.post("/questions", response_model=QuestionsResponse)
async def get_questions(req: QuestionsRequest):
    count      = max(1, min(req.count, 15))
    role_id    = req.role_id.lower().strip()
    difficulty = req.difficulty.lower().strip()
    types      = [t.lower() for t in req.types] if req.types else ["technical"]

    # ── Try loading from questions.json first ─────────────────────────────────
    bank = QUESTION_BANK.get(role_id, {})

    pool: list = []
    if bank:
        for qtype in types:
            qs = bank.get(qtype, [])
            # Filter by difficulty if possible, else take all
            diff_filtered = [q for q in qs if q.get("difficulty") == difficulty]
            source = diff_filtered if diff_filtered else qs
            pool.extend(source)

        random.shuffle(pool)
        selected = pool[:count]

        if selected:
            return QuestionsResponse(
                questions=[
                    Question(
                        question   = q["question"],
                        type       = q.get("type", "technical"),
                        difficulty = q.get("difficulty", difficulty),
                        key_points = q.get("key_points", [])
                    )
                    for q in selected
                ]
            )

    # ── Fallback: ask Gemini to generate questions ────────────────────────────
    type_str = ", ".join(types)
    prompt = f"""
You are an expert technical interviewer. Generate {count} interview questions for the role: {req.role}.
Difficulty level: {difficulty}. Question types to include: {type_str}.

Return ONLY a valid JSON array. No markdown, no explanation. Each item must have this structure:
{{
  "question": "<the interview question>",
  "type": "<technical|behavioral|situational>",
  "difficulty": "{difficulty}",
  "key_points": ["<key point 1>", "<key point 2>", "<key point 3>"]
}}

Mix the types proportionally across {count} questions.
"""

    fallback_questions = [
        {
            "question": f"Tell me about your experience relevant to the {req.role} role.",
            "type": "behavioral",
            "difficulty": difficulty,
            "key_points": ["Specific examples", "Quantifiable impact", "Relevance to role"]
        },
        {
            "question": f"What are the core technical skills required for a {req.role}?",
            "type": "technical",
            "difficulty": difficulty,
            "key_points": ["Domain knowledge", "Tools and technologies", "Best practices"]
        },
        {
            "question": "Describe a challenging project and how you overcame obstacles.",
            "type": "situational",
            "difficulty": difficulty,
            "key_points": ["Problem identification", "Action taken", "Outcome and learnings"]
        },
    ]

    result = ask_gemini(prompt, fallback_questions)

    # Gemini should return a list directly for this prompt
    if isinstance(result, list):
        questions_raw = result
    elif isinstance(result, dict) and "questions" in result:
        questions_raw = result["questions"]
    else:
        questions_raw = fallback_questions

    # Sanitise each question object
    cleaned = []
    for q in questions_raw[:count]:
        if not isinstance(q, dict):
            continue
        cleaned.append(Question(
            question   = str(q.get("question", "Tell me about yourself.")),
            type       = str(q.get("type", "behavioral")),
            difficulty = str(q.get("difficulty", difficulty)),
            key_points = q.get("key_points", []) if isinstance(q.get("key_points"), list) else []
        ))

    if not cleaned:
        cleaned = [Question(**q) for q in fallback_questions]

    return QuestionsResponse(questions=cleaned)

# =============================================================================
#  ROUTE 3  —  POST /evaluate
#  Frontend sends: { question, answer, key_points, type }
#  Returns: EvalResponse
# =============================================================================

@app.post("/evaluate", response_model=EvalResponse)
async def evaluate_answer(req: EvalRequest):
    if not req.answer or req.answer.strip() == "":
        raise HTTPException(status_code=400, detail="Answer cannot be empty")

    key_points_str = "\n".join(f"- {p}" for p in req.key_points) if req.key_points else "- No specific key points provided"

    # ── Gemini prompt ─────────────────────────────────────────────────────────
    prompt = f"""
You are a senior interviewer evaluating a candidate's answer. Be fair, constructive, and specific.

Question ({req.type}):
{req.question}

Expected Key Points:
{key_points_str}

Candidate's Answer:
{req.answer[:1500]}

Evaluate the answer and return ONLY a valid JSON object. No markdown, no extra text.

{{
  "score": <integer 0-100>,
  "feedback": "<1-2 sentences of honest, specific, constructive feedback>",
  "ideal_answer": "<a concise 2-3 sentence model answer>",
  "missing_points": ["<point the candidate missed>", "<another missed point>"]
}}

Scoring guide:
- 85-100: Excellent — covers key points with depth and clear communication
- 70-84:  Good — covers most points, minor gaps
- 50-69:  Fair — some relevant content but missing important aspects  
- 25-49:  Weak — vague or mostly off-track
- 0-24:   Poor — irrelevant or essentially no answer
"""

    # Fallback: simple keyword-based scoring (no AI)
    def keyword_fallback() -> dict:
        answer_lower = req.answer.lower()
        hits = sum(
            1 for kp in req.key_points
            if any(word in answer_lower for word in kp.lower().split())
        ) if req.key_points else 1

        total_kp = len(req.key_points) if req.key_points else 1
        ratio = hits / total_kp
        word_count = len(req.answer.split())

        # Score based on keyword coverage + answer length
        base = int(ratio * 70)
        length_bonus = min(20, word_count // 5)
        score = min(100, base + length_bonus)

        missed = [kp for kp in req.key_points if not any(
            word in answer_lower for word in kp.lower().split()
        )]

        return {
            "score": score,
            "feedback": (
                "Good attempt — your answer covers some relevant points."
                if score >= 50
                else "Your answer needs more depth. Try to address the key technical aspects directly."
            ),
            "ideal_answer": f"A strong answer would address: {', '.join(req.key_points[:3])}." if req.key_points else "Provide a specific, structured answer with examples.",
            "missing_points": missed[:3]
        }

    result = ask_gemini(prompt, keyword_fallback())

    # Sanitise
    score = int(result.get("score", 50))
    score = max(0, min(100, score))

    missing = result.get("missing_points", [])
    if not isinstance(missing, list):
        missing = []

    return EvalResponse(
        score        = score,
        feedback     = str(result.get("feedback", "Answer received.")),
        ideal_answer = str(result.get("ideal_answer", "")),
        missing_points = [str(p) for p in missing[:5]]
    )

# =============================================================================
#  ROUTE — GET /health
#  Simple health check — open http://localhost:8000/health in browser to verify
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ai_available": AI_AVAILABLE,
        "questions_loaded": len(QUESTION_BANK),
        "roles_available": list(QUESTION_BANK.keys())
    }

# =============================================================================
#  ENTRY POINT
#  Run directly: python server.py
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*55)
    print("  PrepAI — Backend Server")
    print("="*55)
    print(f"  AI (Gemini):     {'✓ Ready' if AI_AVAILABLE else '✗ No key — fallback mode'}")
    print(f"  Questions file:  {'✓ Loaded (' + str(len(QUESTION_BANK)) + ' roles)' if QUESTION_BANK else '✗ Not found'}")
    print(f"  Docs:            http://localhost:8000/docs")
    print(f"  Health:          http://localhost:8000/health")
    print("="*55 + "\n")

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)