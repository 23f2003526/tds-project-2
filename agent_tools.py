# agent_tools.py
import base64
import os
import tempfile
from urllib.parse import urlparse
import uuid
import requests
import pandas as pd
import json
import subprocess
import io
import matplotlib.pyplot as plt
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_classic import hub
from openai import OpenAI
import pdfplumber
import zipfile
import tarfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from faster_whisper import WhisperModel

from dotenv import load_dotenv
load_dotenv()

local_model = WhisperModel("base", device="cpu", compute_type="int8")

# --- Code Execution Helper (NON-TOOL FUNCTION) ---
def strip_code_fences(code: str) -> str:
    """Removes leading/trailing code fences (```python or ```) from a string."""
    code = code.strip()
    # Remove ```python ... ``` or ``` ... ```
    if code.startswith("```"):
        # remove first line (```python or ```)
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("\n", 1)[0]
    return code.strip()

# --- New Tool 1: Dependency Management ---
@tool
def add_dependencies(dependencies: List[str]) -> str:
    """
    Install the given Python packages into the environment using uv add.
    
    Parameters:
        dependencies (List[str]):
            A list of Python package names to install (e.g., ['requests', 'numpy']).
            
    Returns:
        str:
            A message indicating success or failure.
    """
    print(f"-> Installing dependencies: {', '.join(dependencies)}")
    try:
        # Use uv add to install packages
        result = subprocess.run(
            ["uv", "add"] + dependencies,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True # check=True raises CalledProcessError on non-zero exit code
        )
        return "Successfully installed dependencies: " + ", ".join(dependencies)
    
    except subprocess.CalledProcessError as e:
        return (
            "Dependency installation failed.\n"
            f"Exit code: {e.returncode}\n"
            f"Error: {e.stderr or 'No error output.'}"
        )
    
    except Exception as e:
        return f"Unexpected error while installing dependencies: {e}"

# --- New Tool 2: Code Execution ---
@tool
def run_code(code: str) -> dict:
    """
    Executes Python code in an isolated environment and returns the output.
    """
    try: 
        cleaned_code = strip_code_fences(code)
        
        # Ensure directory exists
        os.makedirs("LLMFiles", exist_ok=True)
        
        # We write the file to LLMFiles/runner.py
        filename = "LLMFiles/runner.py"
        
        with open(filename, "w") as f:
            f.write(cleaned_code)

        print(f"-> Executing code in {filename}")
        
        # --- FIX HERE: Run from ROOT directory, pointing to the script ---
        # This allows the script to access 'sales_data.csv' in the root folder.
        proc = subprocess.Popen(
            ["uv", "run", filename], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="." # Run in current working directory (project root)
        )
        
        stdout, stderr = proc.communicate(timeout=30)

        return {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": proc.returncode
        }
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return {
            "stdout": "",
            "stderr": "Execution timed out after 30 seconds.",
            "return_code": -2
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }

# --- New Tool: Send POST Request ---
@tool
# The tool now returns a Dict (on success) or a String (on failure)
def send_post_request(url: str, json_payload: Dict[str, Any]) -> Dict[str, Any] | str:
    """
    Sends an HTTP POST request with `json_payload` to `url`.

    On success with a valid JSON response containing a boolean 'correct' field,
    this tool returns the FULL parsed response dict from the API
    (including keys like 'correct', 'url', 'reason', 'delay', ...).

    On failure (HTTP error or invalid JSON), it returns a human-readable error string.

    The agent MUST treat the tool's RETURN VALUE (observation) as the *only* source
    of truth about whether the answer is correct and what the next URL is.
    """
    try:
        response = requests.post(url, json=json_payload, timeout=10)
        response.raise_for_status() 
        
        # Parse the JSON response
        try:
            api_response = response.json()
        except json.JSONDecodeError:
            return f"POST Success, but failed to parse API response as JSON. Raw text: {response.text}"

        return api_response

    except requests.exceptions.RequestException as e:
        # Return a string error if the HTTP request itself failed (e.g., 400 or timeout)
        return f"POST Failed. Error: {e}"

# --- Tool 1: Playwright Scraper (ROBUST VERSION) ---
@tool
def scrape_data(url: str, selector: str = 'body') -> str:
    """
    Scrapes a web page using a headless browser (Playwright).
    Handles JavaScript rendering and retrieves 'hidden' content.
    
    Args:
        url: The URL to visit.
        selector: The CSS selector to scrape (default is 'body').
    
    Returns:
        The raw HTML (outerHTML) of the selected element. 
        This includes all child tags, visible or hidden.
    """
    try:
        # We use sync_playwright here to ensure the tool blocks until it has the data.
        # This prevents async loop conflicts inside the AgentExecutor.
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            # Create a context to allow standard browser behavior
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            page = context.new_page()
            
            # Navigate and wait for the network to settle (Wait for JS to finish)
            page.goto(url, wait_until="networkidle", timeout=60000)
            
            # Wait for the selector to exist in the DOM
            try:
                page.wait_for_selector(selector, timeout=5000)
            except:
                return f"Error: Selector '{selector}' not found on page {url}"

            # CRITICAL: We get outerHTML, not innerText. 
            # innerText ignores <div style="display:none">. 
            # outerHTML captures the full structure, allowing the LLM to parse hidden nodes.
            html_content = page.locator(selector).first.evaluate("el => el.outerHTML")
            
            browser.close()
            
            # Clean up excessive whitespace to save tokens, but keep structure
            return f"HTML Content from {url}:\n{html_content[:30000]}" # Safety truncate

    except Exception as e:
        return f"Scraping failed for {url}: {e}"

@tool
def download_file(url: str, local_filename: str) -> str:
    """Downloads a file (e.g., CSV, PDF) to a local path within the LLMFiles folder."""
    
    # 1. Define the target directory
    SAVE_DIR = Path("LLMFiles")
    
    try:
        # 2. Ensure the directory exists
        # parents=True creates parent directories if needed
        # exist_ok=True prevents an error if the directory already exists
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 3. Construct the full path for the file
        # The / operator is used by pathlib for cross-platform path joining
        full_local_path = SAVE_DIR / local_filename
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 4. Open the file using the full path
        with open(full_local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return f"File saved to: {full_local_path}"
        
    except Exception as e:
        return f"Error downloading {url}: {e}"
    
# --- Tool 2: Audio Transcriber (NEW) ---
@tool
def transcribe_audio(audio_url: str) -> str:
    """
    Downloads an audio file (mp3, wav, opus, etc.) from a URL and transcribes 
    it to text using OpenAI Whisper. Use this to understand spoken instructions.
    """
    try:
        # 1. Download audio to a temporary file
        response = requests.get(audio_url, stream=True)
        if response.status_code == 404:
            return f"Error: 404 Not Found. The URL '{audio_url}' does not exist. Please check your URL resolution logic. Did you append the filename to a page that didn't end in a slash?"
        
        response.raise_for_status()
        
        parsed_url = urlparse(audio_url)
        path_without_query = parsed_url.path
        ext = os.path.splitext(path_without_query)[1]
        if not ext: 
            ext = ".opus" # Fallback
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
            for chunk in response.iter_content(chunk_size=8192):
                temp_audio.write(chunk)
            temp_path = temp_audio.name

        segments, info = local_model.transcribe(temp_path, beam_size=5)

        full_text = " ".join([segment.text for segment in segments])
        
        # Cleanup
        os.remove(temp_path)
        
        return f"Audio Transcription: {full_text}"
        
    except Exception as e:
        return f"Error transcribing audio: {e}"

@tool
def analyze_csv_data(file_path: str, query_expression: str) -> str:
    """
    Analyzes a CSV file using Python code.
    
    Args:
        file_path: Local path to the CSV file.
        query_expression: Python code to execute. The dataframe is available as 'df'.
                          IMPORTANT: Assign your final answer to a variable named 'result'.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check for headless CSV (heuristic)
        try:
            float(df.columns[0])
            df = pd.read_csv(file_path, header=None)
        except ValueError:
            pass

        local_scope = {"df": df, "pd": pd}
        
        # --- FIX HERE: Pass global builtins so int(), round(), print() work ---
        global_scope = {"__builtins__": __builtins__}
        
        exec(query_expression, global_scope, local_scope)
        
        if 'result' in local_scope:
            return f"Analysis Result: {local_scope['result']}"
        else:
            return "Error: You must assign your answer to a variable named 'result'."
        
    except Exception as e:
        return f"Error analyzing CSV with query '{query_expression}': {e}"

# --- New Tool for Authenticated GET Requests ---
@tool
def fetch_api_data(url: str, headers_json: str) -> str:
    """
    Fetches data from an API endpoint using an HTTP GET request, including
    required custom headers for authentication or content type.
    
    The headers_json parameter MUST be a valid JSON string representing the
    HTTP headers as a dictionary (e.g., '{"X-API-Key": "your-key-value", "Accept": "application/json"}').
    
    Returns the JSON response data as a string, or an error message.
    """
    try:
        # Convert the JSON string of headers into a Python dictionary
        import json
        headers_dict = json.loads(headers_json)
        
        # Make the GET request
        response = requests.get(url, headers=headers_dict, timeout=15)
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        # Return the JSON content as a formatted string
        return f"API Response (Status {response.status_code}): {response.json()}"
        
    except requests.exceptions.RequestException as e:
        return f"API Request Error: Failed to fetch {url}. Error: {e}"
    except json.JSONDecodeError:
        return f"API Request Error: The headers_json input '{headers_json}' is not valid JSON."
    except Exception as e:
        return f"An unexpected error occurred during API fetch: {e}"

@tool
def extract_pdf_text(file_path: str, pages: str = "all", max_chars: int = 30000) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file_path: Local path to the PDF (use download_file first if needed).
        pages: "all", "1", "1-3", or "1,3,5" (1-based indices).
        max_chars: Truncate the returned text to this many characters for safety.

    Returns:
        Extracted text (possibly truncated) as a single string.
    """
    try:
        page_indices = None
        if pages != "all":
            requested = []
            for part in pages.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = part.split("-")
                    requested.extend(range(int(start), int(end) + 1))
                else:
                    requested.append(int(part))
            # Convert to 0-based indices
            page_indices = [p - 1 for p in requested]

        texts = []
        with pdfplumber.open(file_path) as pdf:
            if page_indices is None:
                iterable = pdf.pages
            else:
                iterable = [pdf.pages[i] for i in page_indices if 0 <= i < len(pdf.pages)]
            for page in iterable:
                texts.append(page.extract_text() or "")

        full_text = "\n\n".join(texts)
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n...[TRUNCATED]"

        return f"PDF Text from {file_path} (pages={pages}):\n{full_text}"
    except Exception as e:
        return f"Error extracting text from PDF {file_path}: {e}"

@tool
def analyze_tabular_file(file_path: str, query_expression: str) -> str:
    """
    Load a tabular file (CSV, TSV, XLSX, JSON, Parquet, JSONL) into a pandas DataFrame 'df'
    and execute Python code against it.

    The final answer MUST be assigned to a variable named 'result' in query_expression.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".csv"]:
            df = pd.read_csv(file_path)
        elif ext in [".tsv"]:
            df = pd.read_csv(file_path, sep="\t")
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif ext in [".json"]:
            df = pd.read_json(file_path)
        elif ext in [".jsonl", ".ndjson"]:
            # JSON Lines: one JSON object per line
            df = pd.read_json(file_path, lines=True)
        elif ext in [".parquet"]:
            df = pd.read_parquet(file_path)
        else:
            return f"Unsupported file extension '{ext}' for {file_path}"

        local_scope = {"df": df, "pd": pd}
        exec(query_expression, {"__builtins__": {}}, local_scope)

        if "result" not in local_scope:
            return "Error: You must assign your final answer to a variable named 'result'."

        return f"Analysis Result: {local_scope['result']}"
    except Exception as e:
        return f"Error analyzing file '{file_path}' with query '{query_expression}': {e}"
    
@tool
def plot_with_matplotlib(file_path: str, code: str) -> str:
    """
    Create a plot using matplotlib and optional pandas, given a local file.

    - Loads the file into `df` (using the same logic as analyze_tabular_file).
    - Executes `code` where `df`, `pd`, and `plt` are available.
    - Assumes `code` creates exactly one figure (using matplotlib).
    - Returns a base64-encoded PNG data URI.

    Example code:
        "plt.plot(df['x'], df['y']); plt.title('My Plot')"
    """
    import os
    import pandas as pd

    try:
        # 1. Load df like analyze_tabular_file
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".csv"]:
            df = pd.read_csv(file_path)
        elif ext in [".tsv"]:
            df = pd.read_csv(file_path, sep="\t")
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif ext in [".json"]:
            df = pd.read_json(file_path)
        elif ext in [".parquet"]:
            df = pd.read_parquet(file_path)
        else:
            return f"Unsupported file extension '{ext}' for {file_path}"

        # 2. Execute plotting code
        plt.clf()
        local_scope = {"df": df, "pd": pd, "plt": plt}
        exec(code, {"__builtins__": {}}, local_scope)

        # 3. Save figure to in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        png_bytes = buf.read()
        buf.close()

        # 4. Encode as base64 data URI
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64}"
        return f"Plot Data URI: {data_uri}"
    except Exception as e:
        return f"Error generating plot from {file_path}: {e}"
    
@tool
def ocr_image(file_path: str, lang: str = "eng") -> str:
    """
    Perform OCR on an image file and return the extracted text.
    Includes basic preprocessing and Tesseract config tuning.
    Also tries to detect an ALL-CAPS/underscore 'code' like VISION_MASTER.
    """
    try:
        import pytesseract
        from PIL import Image, ImageOps, ImageFilter, ImageEnhance
        import re
        import os

        if not os.path.exists(file_path):
            return f"Error: file does not exist: {file_path}"

        # 1. Load and preprocess
        img = Image.open(file_path)

        # Convert to grayscale
        img = img.convert("L")

        # Increase contrast / normalize
        img = ImageOps.autocontrast(img)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)

        # Upscale small images (Tesseract likes bigger text)
        w, h = img.size
        max_side = max(w, h)
        if max_side < 800:
            scale = max(2, int(800 / max_side))
            img = img.resize((w * scale, h * scale), Image.LANCZOS)

        # Slight denoising
        img = img.filter(ImageFilter.MedianFilter(size=3))

        # 2. Run Tesseract with a more appropriate config
        # psm 6: assume a block of text
        config_1 = "--oem 3 --psm 6"
        text1 = pytesseract.image_to_string(img, lang=lang, config=config_1)

        # If we got almost nothing, try a different PSM (single block / single line)
        text1_clean = text1.strip()
        if len(text1_clean) < 10:
            config_2 = "--oem 3 --psm 7"  # single text line
            text2 = pytesseract.image_to_string(img, lang=lang, config=config_2)
            if len(text2.strip()) > len(text1_clean):
                text1 = text2

        full_text = text1.strip()

        result_parts = [f"OCR Text from {file_path}:\n{full_text}"]

        return "\n".join(result_parts)

    except Exception as e:
        return f"Error performing OCR on {file_path}: {e}"
    
@tool
def extract_archive(archive_path: str, extract_to: str = "LLMFiles/extracted") -> str:
    """
    Extract a .zip or .tar(.gz) archive to a directory.

    Returns the extraction directory path AND a list of extracted files.
    """
    try:
        os.makedirs(extract_to, exist_ok=True)
        if archive_path.lower().endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_to)
        elif archive_path.lower().endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(extract_to)
        else:
            return f"Unsupported archive format for {archive_path}"

        # NEW: list all files so the agent can see them
        file_list = []
        for root, dirs, files in os.walk(extract_to):
            for name in files:
                rel_path = os.path.join(root, name)
                file_list.append(rel_path)

        files_str = "\n".join(sorted(file_list))
        return f"Archive extracted to: {extract_to}\nFiles:\n{files_str}"
    except Exception as e:
        return f"Error extracting archive {archive_path}: {e}"
    
@tool
def list_files(directory: str) -> str:
    """
    List all files under a directory (recursively).
    """
    if not os.path.exists(directory):
        return f"Directory does not exist: {directory}"

    paths = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            paths.append(os.path.join(root, name))

    if not paths:
        return f"No files found under {directory}"
    
    return "Files:\n" + "\n".join(sorted(paths))

@tool
def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes a local image file into a Base64 string.
    Useful when you need to send an image to an API or Vision model.
    
    Args:
        image_path: The local file path to the image (e.g., 'downloaded_image.png').
        
    Returns:
        str: The raw Base64 encoded string of the image content.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        return f"Error: The file '{image_path}' was not found."
    except Exception as e:
        return f"Error encoding image: {e}"

# --- Agent Factory (Tools List Updated) ---
def get_langchain_agent() -> Any:
    """Creates a standard AgentExecutor that runs the loop:
    Thought -> Tool Call -> Observation -> Final Answer."""
    # INCLUDE THE NEW POST TOOL
    tools = [
        scrape_data, 
        download_file, 
        transcribe_audio, 
        analyze_csv_data, 
        analyze_tabular_file,
        send_post_request, 
        fetch_api_data,
        add_dependencies,
        run_code,
        extract_pdf_text,
        plot_with_matplotlib,
        ocr_image,
        extract_archive,
        encode_image_to_base64,
        list_files
        ] 

    prompt = hub.pull("hwchase17/openai-tools-agent")

    llm = ChatOpenAI(
        model="openai/gpt-4.1-mini",
        api_key=os.getenv("AIPIPE_TOKEN"), 
        base_url="https://aipipe.org/openrouter/v1/", 
        temperature=0.0    
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, 
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    
    return agent_executor