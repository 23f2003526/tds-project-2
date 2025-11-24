# quiz_solver.py

import asyncio
import time
import requests
from playwright.async_api import async_playwright
from typing import Dict, Any
from agent_tools import get_langchain_agent

# --- Constants ---
QUIZ_TIMEOUT_SECONDS = 180  # 3 minutes


# --- LangChain Agent/Tool Imports (Conceptual - you'll build these later) ---
# from agent_tools import get_langchain_agent, PlaywrightTool, PDFTool 
# Note: We will substitute the LangChain logic for a simple placeholder first.

# --- Core Scraper Function (Playwright) ---
async def scrape_quiz_content(url: str) -> Dict[str, Any]:
    """
    Launches Playwright, navigates to the URL, and extracts the quiz question 
    and submission endpoint.
    """
    url = str(url)
    print(f"-> Starting Playwright for URL: {url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            # Step 1. Navigate to the quiz URL
            await page.goto(url, wait_until="networkidle") 
            
            # Step 2. Extract visible text of the whole page
            page_text = await page.inner_text("body")
            html = await page.content()

            await browser.close()

        except Exception as e:
            await browser.close()
            raise Exception(f"Playwright Scraping Failed: {e}")
        
    # Step 3. Use LangChain agent to interpret content
    try:
        agent = get_langchain_agent()
        result = agent.invoke({
        "input": f"Scrape the question and submission URL from this quiz page:\n{page_text[:4000]}"
        })
        
        print("Agent result:", result)
        
        # Try to extract submission URL via regex fallback if agent missed it
        # submit_match = re.search(r'https?://[^\s"\'<>]+/submit[^\s"\'<>]*', page_text)
        # submit_url = submit_match.group(0) if submit_match else SUBMISSION_ENDPOINT_FALLBACK

        # Compose output
        return {
            # "question_text": result if isinstance(result, str) else page_text.strip(),
            "question_text": result,
            "submit_url": submit_url
        }

    except Exception as e:
        raise Exception(f"LangChain Agent Parsing Failed: {e}")

# --- Quiz Resolver (The main recursive function) ---
def solve_quiz_task(
    email: str, 
    secret: str, 
    url: str, 
    start_time: float
):
    """
    The main, recursive function that solves a quiz task.
    This function will run in a background thread provided by FastAPI.
    """
    current_url = str(url)
    
    while current_url and (time.time() - start_time) < QUIZ_TIMEOUT_SECONDS:
        print(f"\n--- Solving new Quiz: {current_url} (Time Left: {QUIZ_TIMEOUT_SECONDS - (time.time() - start_time):.1f}s) ---")
        
        try:
            # 1. SCRAPE: Execute the async Playwright function
            # Since this is a sync function (called by BackgroundTasks), we must run the async Playwright function using asyncio.run()
            # Note: A proper production solution uses a task queue (like Celery) to manage this threading/async complexity.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            quiz_details = loop.run_until_complete(scrape_quiz_content(str(current_url)))

            question_text = quiz_details['question_text']
            submit_url = quiz_details['submit_url']

            print(f"Question Text (Snippet):\n{question_text[:150]}...")
            
            # 2. SOLVE: Placeholder for LangChain Agent logic
            # Here is where the LangChain agent takes the question, uses its tools 
            # (including Playwright/PDF parser/Pandas) to find the answer.
            
            # agent = get_langchain_agent() 
            # final_answer = agent.run(question_text)
            
            # --- Hardcoded Placeholder Answer for Demo ---
            # Replace this with the actual result from your LangChain Agent
            calculated_answer = 12345 
            
            # 3. SUBMIT
            submission_payload = {
                "email": email,
                "secret": secret,
                "url": current_url,
                "answer": calculated_answer 
            }
            
            print(f"Submitting answer: {calculated_answer}")
            response = requests.post(submit_url, json=submission_payload, timeout=5)
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            
            submit_result: Dict[str, Any] = response.json()
            
            is_correct = submit_result.get("correct", False)
            next_url = submit_result.get("url")
            
            print(f"Submission Result: Correct={is_correct}. Next URL={next_url}")
            
            # 4. ITERATE / RECURSE
            if is_correct or next_url:
                current_url = next_url
            else:
                # Answer was wrong, no new URL, so we can stop or re-try the current one
                print("Quiz failed and no new URL provided. Stopping sequence.")
                break 

        except requests.exceptions.RequestException as e:
            print(f"HTTP Submission Error: {e}")
            break
        except Exception as e:
            print(f"General Error during solving: {e}")
            break

    if not current_url:
        print("\n--- Quiz Sequence Complete: No new URL received. ---")
    elif (time.time() - start_time) >= QUIZ_TIMEOUT_SECONDS:
        print("\n--- Quiz Sequence TIMED OUT. ---")