---
title: LLM Analysis Quiz Solver
emoji: 🏃
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# 🧠 Quiz Solver Agent — TDS Project 2

## 📘 Overview

The **Quiz Solver Agent** is an automated FastAPI-based system designed to solve sequential quiz tasks hosted on the TDS evaluation platform (`tds-llm-analysis.s-anand.net`).  
It leverages **LangChain**, **OpenAI (via AIPipe)**, and **Playwright** to dynamically interpret quiz instructions, download required data, perform analysis, and submit answers automatically.

---

## 🚀 Key Features

- ⚡ **FastAPI Backend** — Provides endpoints to trigger quiz solving asynchronously.
- 🧩 **LangChain Agent** — Core reasoning engine that interprets each task and invokes the correct tools automatically.
- 🕵️ **Dynamic Web Scraping** — Uses Playwright to scrape or render quiz pages and extract task content.
- 🧰 **Custom Tools Library** — Provides modular tools for data analysis, file handling, web requests, OCR, audio transcription, and visualization.
- 🔁 **Recursive Quiz Solving** — Automatically follows next URLs provided by the grader until the quiz sequence completes.
- 🧮 **Automatic CSV/JSON Normalization** — Detects file types, standardizes date formats, handles JSONL, and performs calculations autonomously.
- 💾 **Resilient Retry & Timeout System** — Handles transient failures, timeouts, and ensures background processing reliability.

---

## 🏗️ Project Architecture

```
📦 project-root/
├── main.py                  # FastAPI app & background task handler
├── quiz_solver.py           # Core quiz solving loop and orchestration logic
├── agent_tools.py           # Modular LangChain-compatible tools
├── requirements.txt         # Python dependencies
├── .env                     # Secrets (EXPECTED_SECRET, AIPIPE_TOKEN, etc.)
├── pyproject.toml
├── Dockerfile
└── LLMFiles/                # Runtime-generated files & artifacts


```

---

## ⚙️ Tech Stack

| Component                  | Technology Used                                |
| -------------------------- | ---------------------------------------------- |
| **Framework**              | FastAPI                                        |
| **LLM Engine**             | LangChain + OpenAI GPT (via AIPipe/OpenRouter) |
| **Browser Automation**     | Playwright (Headless Chromium)                 |
| **Data Processing**        | Pandas, NumPy                                  |
| **Visualization**          | Matplotlib                                     |
| **Audio Transcription**    | Faster Whisper                                 |
| **OCR**                    | Tesseract                                      |
| **Environment Management** | dotenv, uv                                     |
| **Background Tasks**       | FastAPI BackgroundTasks                        |

---

## 🧠 How It Works

1. **Incoming Request**

   - A POST request is sent to `/quiz-task` with `{ email, secret, url }`.
   - The secret is validated against `.env` (`EXPECTED_SECRET`).

2. **Background Execution**

   - A background task is spawned to execute `solve_quiz_task()`.

3. **Quiz Solving Loop**

   - The agent scrapes the quiz page.
   - It interprets the text to determine required operations (e.g., “Download CSV”, “Compute hash”, “Normalize JSON”).
   - Based on the task, it calls the right tool (`download_file`, `analyze_tabular_file`, `transcribe_audio`, etc.).
   - Results are submitted using `send_post_request()`.

4. **Adaptive Logic**
   - If a submission fails, the system retries or uses a “skip” strategy to continue solving further quizzes.
   - Automatically handles multiple linked tasks in a single sequence.

---

## 🧰 Tools Implemented

| Tool Name                | Purpose                                                                  |
| ------------------------ | ------------------------------------------------------------------------ |
| `scrape_data`            | Render and extract HTML from web pages                                   |
| `download_file`          | Download remote files (CSV, JSON, ZIP, PDF, etc.)                        |
| `extract_archive`        | Extract ZIP/TAR files and list contents                                  |
| `analyze_csv_data`       | Execute Python queries on CSVs (with builtins enabled)                   |
| `analyze_tabular_file`   | Analyze multi-format tables (.csv, .tsv, .xlsx, .json, .jsonl, .parquet) |
| `transcribe_audio`       | Convert audio to text via Whisper                                        |
| `ocr_image`              | Perform OCR on images                                                    |
| `fetch_api_data`         | Make authenticated GET requests                                          |
| `extract_pdf_text`       | Extract text from PDFs                                                   |
| `plot_with_matplotlib`   | Generate visualizations and return Base64 data URI                       |
| `run_code`               | Safely execute arbitrary Python code snippets                            |
| `add_dependencies`       | Install runtime dependencies dynamically                                 |
| `encode_image_to_base64` | Encode local images for API/ML use                                       |

---

## ⚡ Example Flow

**POST /quiz-task**

```bash
curl -X POST https://your-server-url/quiz-task \
  -H "Content-Type: application/json" \
  -d '{
    "email": "23f2003526@ds.study.iitm.ac.in",
    "secret": "fruitberry",
    "url": "https://tds-llm-analysis.s-anand.net/project2-csv"
  }'
```

**Response**

```json
{
  "message": "Quiz processing started successfully in the background."
}
```

---

## 🧩 Advanced Handling

- **.jsonl Detection:** Automatically reads JSON Lines files (`pd.read_json(..., lines=True)`).
- **Archive Extraction:** Lists all extracted file paths to help the agent locate data files.
- **Auto-Retry:** Handles timeouts or partial data gracefully using retry loops.

---

## 🔒 Environment Variables

| Variable                   | Description                                   |
| -------------------------- | --------------------------------------------- |
| `EXPECTED_SECRET`          | Token required for authorized quiz requests   |
| `AIPIPE_TOKEN`             | API key for AIPipe/OpenRouter (LLM access)    |
| `PLAYWRIGHT_BROWSERS_PATH` | Path for headless browser binaries (optional) |

---

## 🧪 Running the Server

```bash
# 1. Install dependencies
uv pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# then edit .env with your own keys

# 3. Run FastAPI app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Health Check**

```bash
curl http://localhost:8000/health
# {"status": "ok", "message": "Server is running 🚀"}
```

---

## 🧠 Future Enhancements

- ✅ Add automatic file discovery after extraction (done via improved `extract_archive`)
- 🔜 Add intelligent caching for repeated file downloads
- 🔜 Integrate logging dashboard for background tasks
- 🔜 Add richer prompt-engineering templates for different task types

---

## 👨‍💻 Author

**Namit Gupta**  
Project developed as part of _Tools for Data Science (TDS) Project 2_ — Indian Institute of Technology Madras.

---

## 📄 License

This project is released under the MIT License.
