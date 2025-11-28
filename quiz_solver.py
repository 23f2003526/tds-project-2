# quiz_solver.py
import asyncio
import json
import os
import time
import re # We'll need this for parsing the Agent's final response
from dotenv import load_dotenv
import requests
from playwright.async_api import async_playwright
from typing import Dict, Any
from urllib.parse import urljoin
from agent_tools import get_langchain_agent, send_post_request as send_post_tool, scrape_data as scrape_data_tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from urllib.parse import urljoin

load_dotenv()
QUIZ_TIMEOUT_SECONDS = 3600

QUESTION_TIMEOUT_SECONDS = 150

url_extractor_llm = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1/",
    temperature=0.0,
    max_tokens=128,
)

# Initialize Agent once outside the function
agent_executor = get_langchain_agent()

SUBMIT_URL_REGEXES = [
    r"Post your answer to\s+(https?://\S+)",          # common pattern in instructions
    r"submit to\s+(https?://\S+)",
    r'"(https?://[^"\s]+/submit[^"\s]*)"',            # anything with /submit inside quotes
    r"(https?://[^\s]+/submit[^\s]*)",                # fallback: any /submit URL
]

def extract_submit_url_via_llm(page_url: str) -> str | None:
    """
    Uses scrape_data on page_url, then asks a small LLM to extract the submit URL.

    Returns:
        Absolute submit URL as a string, or None if it couldn't be found.
    """
    try:
        # 1. Get HTML for the page using your existing tool
        raw_content = scrape_data_tool.func(url=page_url, selector="body")

        if not isinstance(raw_content, str):
            print("-> [SUBMIT URL LLM] scrape_data did not return a string.")
            return None

        # 2. Strip the "HTML Content from ..." prefix if present
        #    It usually looks like: "HTML Content from {url}:\n<html>..."
        if raw_content.startswith("HTML Content from"):
            # split once on first newline
            parts = raw_content.split("\n", 1)
            if len(parts) == 2:
                raw_html = parts[1]
            else:
                raw_html = raw_content
        else:
            raw_html = raw_content

        # 3. Ask the LLM to identify the submit URL
        system_msg = SystemMessage(
            content=(
                "You are a precise HTML analyst. "
                "You will be given the full HTML content of a quiz page. "
                "In that HTML or the visible text, there will be an instruction like:\n"
                "  'Post your answer to https://example.com/submit/123 with this JSON payload ...'\n\n"
                "Your job is to extract the SINGLE submit URL used for POSTing the answer.\n"
                "Rules:\n"
                "- Return ONLY the URL, with no extra text, no quotes, no HTML tags.\n"
                "- The URL must be a POST endpoint where the answer is submitted.\n"
                f"- If the URL is relative (e.g. '/submit/16'), return it with full absolute path. (find the base URL from the current page URL {page_url})\n"
                "- If there is no such URL, return the single word: NONE"
            )
        )

        human_msg = HumanMessage(
            content=(
                f"The current page URL is: {page_url}\n\n"
                "Here is the full HTML content of the page:\n\n"
                f"{raw_html}"
            )
        )

        resp = url_extractor_llm.invoke([system_msg, human_msg])
        extracted = (resp.content or "").strip()

        print(f"-> [SUBMIT URL LLM] Raw model output: {extracted!r}")

        if extracted.upper() == "NONE":
            return None

        print(f"-> [SUBMIT URL LLM] Resolved submit URL: {extracted}")
        return extracted

    except Exception as e:
        print(f"-> [SUBMIT URL LLM] Error while extracting submit URL: {e}")
        return None


# --- Playwright Scraping Function (Kept as is) ---
async def scrape_quiz_content(url: str) -> Dict[str, Any]:
    """
    Launches Playwright, navigates to the URL, and extracts the raw page content.
    """
    url = str(url)
    print(f"\n-> Starting Playwright for URL: {url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(url, wait_until="networkidle") 
            content_to_parse = await page.inner_text("body")

            print("-> Content to parse:", content_to_parse)
            
            if len(content_to_parse) < 50:
                print("-> Text content is empty, switching to raw HTML parsing...")
                content_to_parse = await page.content()

            await browser.close()
            
            return {
                "page_url": url,
                "page_content": content_to_parse
            }

        except Exception as e:
            await browser.close()
            raise Exception(f"Playwright Scraping Failed: {e}")

# --- Quiz Resolver (The main function) ---
def solve_quiz_task(email: str, secret: str, url: str, start_time: float):
    """
    The main, recursive function that solves a quiz task with retry logic.
    """
    current_url = str(url)
    agent = agent_executor 
    last_next_url = None
    
    # Final submission JSON result structure
    submit_result: Dict[str, Any] = {} 

    total_tasks = 0
    successful_tasks = 0

    while current_url and (time.time() - start_time) < QUIZ_TIMEOUT_SECONDS:

        total_tasks += 1

        question_start_time = time.time()

        print(f"\n--- Solving new Quiz: {current_url} (Time Left: {QUIZ_TIMEOUT_SECONDS - (time.time() - start_time):.1f}s) ---")

        max_retries = 2
        success = False

        for attempt in range(1, max_retries + 1):

            time_spent_on_question = time.time() - question_start_time
            time_left_on_question = QUESTION_TIMEOUT_SECONDS - time_spent_on_question

            # --- Check Question Timeout ---
            if time_left_on_question <= 0:
                print(f"\n[TIMEOUT] Question timeout of {QUESTION_TIMEOUT_SECONDS}s reached. Moving to next question/URL.")
                break 
            
            # Check Global Timeout
            if (time.time() - start_time) >= QUIZ_TIMEOUT_SECONDS:
                print("\n[GLOBAL TIMEOUT] Quiz sequence global timeout reached.")
                break

            print(f"\nAttempt {attempt}/{max_retries} to solve the quiz at {current_url} (Time Left: {time_left_on_question:.1f}s)")
            
            try:
                # 1. SCRAPE (unchanged)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                quiz_data = loop.run_until_complete(scrape_quiz_content(current_url))

                page_content = quiz_data['page_content']
                page_url = quiz_data['page_url']
                
                # 2. SOLVE: Pass ALL relevant data to the agent (unchanged prompt)
                print("-> Asking Agent to solve based on raw content...")

                agent_prompt = (
                    f"You are currently on the page: {page_url}\n"
                    f"Your submission email is: {email}\n"
                    f"Your submission secret is: {secret}\n"
                    # f"The final submission URL is: {page_url}\n\n"
                    
                    "INSTRUCTIONS:\n"
                    "ANALYZE THE CONTENT: First, examine the 'PAGE CONTENT' below. You MUST identify the quiz question/task and the **submission URL** (the URL where the answer must be POSTed).\n"
                    "1. MATH/HASHING: For any calculation, hashing (SHA1, SHA256), or complex string manipulation, you MUST use the `run_code` tool.\n"
                    "2. ANALYZE: If there is an audio file, use `transcribe_audio` to get the instructions.\n"
                    "3. DATA: If you need to analyze a CSV with conditions (e.g., 'sum numbers > 500'), use `analyze_csv_data`. You can write Pandas code like: \"df[df['value'] > 500]['value'].sum()\".\n"
                    "4.SUBMIT: You MUST use the `send_post_request` tool for final submission.\n"
                    "5. The `send_post_request` tool takes two parameters: `url` and `json_payload`.\n"
                    "6. The `json_payload` parameter MUST be a single Python dictionary containing the keys: 'email', 'secret', 'url', and 'answer'. The url is the current page URL in most cases. Always make sure that the url is not a relative path but full absolute path.\n"
                    "7. Construct the full dictionary first, then pass it as the single `json_payload` argument to the tool call.\n"
                    "8. **CRITICAL:**Your FINAL ANSWER MUST ONLY CONTAIN THE RAW OUTPUT from the `send_post_request` tool.\n"
                    "9. If the task involves scraping, resolve relative paths against the current page URL before using tools."
                    "10. **URL RESOLUTION:** If a link is relative (e.g., 'file.opus') and the current URL does NOT end in a slash (e.g., '.../page'), standard URL resolution replaces the last segment. \n"
                    "   - CORRECT: '.../page' + 'file.opus' -> '.../file.opus'\n"
                    "   - WRONG: '.../page' + 'file.opus' -> '.../page/file.opus'\n"
                    "   Always check the generated URL before calling tools."
                    "11. When using run_code, you MUST use the print() function to output any value you need to see, as the tool only returns the contents of stdout"

                    f"PAGE CONTENT:\n{page_content}"
                )

                result = agent.invoke({"input": agent_prompt})
                # print(result)

                # Try to get the output from the send_post_request tool call
                agent_output = None

                intermediate_steps = result.get("intermediate_steps", [])

                send_post_observations = [
                    observation
                    for action, observation in intermediate_steps
                    if getattr(action, "tool", "") == "send_post_request"
                ]

                if send_post_observations:
                    # The last call to send_post_request is what we care about
                    agent_output = send_post_observations[-1]
                    print(f"-> Agent Tool Output (send_post_request): {agent_output}")
                else:
                    # Fallback: use the final output from the agent (may be just text)
                    agent_output = result.get("output")
                    print(f"-> Agent Final Output (no tool obs found): {agent_output}")

                # --- Process the tool result ---
                if isinstance(agent_output, dict):
                    submit_result = agent_output
                    print("-> Submission Successful!")
                else:
                    # agent_output is likely an error string from the tool
                    print(f"-> Submission Failed. Reason: {agent_output}")
                    submit_result = {}  # reset to empty so .get(...) below is safe


                # 4. PROCESS RESULT
                last_next_url = submit_result.get("url")
                
                if submit_result.get("correct", False):
                    successful_tasks += 1
                    print(f"Progress: {successful_tasks}/{total_tasks} tasks completed successfully.")
                    current_url = last_next_url
                    success = True
                    break # Success! Break the retry loop.
                else:
                    print(f"-> Answer incorrect. Retrying same URL (if attempts left)...")
                    # Loop continues to next attempt

            except Exception as e:
                print(f"General Error on attempt {attempt}: {e}")
                # Loop continues to next attempt

        if success:
            continue 

        if last_next_url:
            print(f"Moving to the next URL provided by the API: {last_next_url}")
            current_url = last_next_url
            continue
        
        # ❗ NEW: Skip strategy using default empty-answer submission
        print("\n[SKIP STRATEGY] Max retries/timeout reached and no next URL from agent.")
        print("Attempting to parse the submit URL from the current page and send an empty answer...")

        # 1. Try to get the submit URL by scraping the current question page again
        submit_url = extract_submit_url_via_llm(current_url)

        if not submit_url:
            print("-> [SKIP STRATEGY] Could not determine submit URL from the page. Exiting.")
            break

        # 2. Build default empty-answer payload
        default_payload = {
            "email": email,
            "secret": secret,
            # IMPORTANT: In the official spec, this 'url' field should be the QUIZ URL, not the submit URL.
            "url": current_url,
            "answer": ""
        }

        try:
            # 3. Send POST to the submit URL using the tool's underlying function
            default_response = send_post_tool.func(url=submit_url, json_payload=default_payload)
            print(f"-> [SKIP STRATEGY] Default submission response from {submit_url}: {default_response}")

            if isinstance(default_response, dict):
                # Try to extract next URL from this response
                last_next_url = default_response.get("url")
            else:
                print("-> [SKIP STRATEGY] Default submission did not return a JSON dict; cannot extract next URL.")
                
        except Exception as e:
            print(f"-> [SKIP STRATEGY] Error during default submission: {e}")
            last_next_url = None

        # 4. Move to next URL if provided
        if last_next_url:
            print(f"Moving to the next URL provided after default submission: {last_next_url}")
            current_url = last_next_url
            continue

        print("Failed to solve quiz and no new URL provided even after default submission. Exiting.")
        break

    # --- Final Conclusion ---
    if not current_url:
        print("\n--- Quiz Sequence Complete: No new URL received. ---")
    elif (time.time() - start_time) >= QUIZ_TIMEOUT_SECONDS:
        print("\n--- Quiz Sequence TIMED OUT. ---")

    # --- NEW: Final stats ---
    print(f"\n=== RUN SUMMARY ===")
    print(f"Total tasks encountered : {total_tasks}")
    print(f"Tasks solved correctly  : {successful_tasks}")
    if total_tasks > 0:
        accuracy = (successful_tasks / total_tasks) * 100.0
        print(f"Accuracy               : {accuracy:.1f}%")
    else:
        print("No tasks were attempted.")
    print("===================")