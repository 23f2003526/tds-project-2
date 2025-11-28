import os
import time
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, EmailStr, HttpUrl, ValidationError
from dotenv import load_dotenv
from quiz_solver import solve_quiz_task

load_dotenv()

EXPECTED_SECRET = os.getenv("EXPECTED_SECRET")

app = FastAPI(title="Quiz Solver Agent")

class QuizRequest(BaseModel):
    """Schema for the JSON payload received by our endpoint."""
    email: EmailStr = Field(..., description="Student email ID")
    secret: str = Field(..., description="Student-provided secret")
    url: HttpUrl = Field(..., description="A unique task URL (must be a valid http/https URL)")

# --- Background Task Wrapper ---
def run_quiz_solver_in_background(email: str, secret: str, url: str, start_time: float):
    """
    Wrapper function for the actual quiz solver. 
    Runs as a background task.
    """
    # The actual solving function is recursive and will handle the iteration.
    try:
        # Pass necessary data to the solver
        solve_quiz_task(email, secret, url, start_time)
    except Exception as e:
        # Important: Log any errors here as the user won't see them directly.
        print(f"CRITICAL ERROR in Quiz Solver for URL {url}: {e}")
        # Optional: You could implement a system to notify you of the failure

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is running 🚀"}

@app.post("/quiz-task", status_code=200)
async def handle_quiz_request(request: Request, background_tasks: BackgroundTasks):
    """Receives the quiz request, validates the secret, and starts the solver."""
    
    # 1. Attempt to parse raw JSON
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON structure.")
    
    # 2. Validate against Pydantic model
    try:
        quiz_data = QuizRequest(**payload)
    except ValidationError as e:
        # Schema is wrong, but JSON is syntactically valid
        raise HTTPException(status_code=400, detail=e.errors())
    
    # 3. Secret verification
    if quiz_data.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret provided.")
    
    # 4. Sanity checks (redundant but safe)
    if not quiz_data.email or not quiz_data.url:
        raise HTTPException(status_code=400, detail="Invalid request payload.")
    
    # 5. Start Timer and Task
    start_time = time.time()
    
    # Add the solver function as a background task. 
    # This allows FastAPI to immediately return a 200 response.
    background_tasks.add_task(
        run_quiz_solver_in_background,
        str(quiz_data.email),
        str(quiz_data.secret),
        str(quiz_data.url),
        start_time
    )

    return {"message": "Quiz processing started successfully in the background."}