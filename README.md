---
title: LLM Analysis Quiz Solver
emoji: 🏃
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# LLM Analysis - Autonomous Quiz Solver Agent

A fully automated FastAPI-based backend service that accepts quiz challenge URLs, extracts the relevant content, processes tasks (including math, CSV analysis, OCR, audio transcription, scraping, hashing, code execution, etc.) using a LangChain agent with extended tool support — and submits the answers automatically.

The service is designed to solve multi-step quizzes recursively until completion or timeout.


## 📋 Table of Contents

- [🔍 Overview](#-overview)  
- [📌 Features](#-features)  
- [🏗️ Architecture](#-architecture)  
- [🧠 How It Works](#-how-it-works)  
- [📝 Key Design Decisions](#-key-design-decisions)  
- [⚙️ Installation & Setup](#-installation--setup)  
- [🔐 Environment Variables](#-environment-variables)  
- [▶️ Running the Server](#-running-the-server)  
- [📡 API Endpoints](#-api-endpoints)  
- [🛠 Tools Available to the Agent](#-tools-available-to-the-agent)  
- [⏱ Timeouts & Limits](#-timeouts--limits)  
- [🧪 Logging & Debugging](#-logging--debugging)  
- [📄 License](#-license)  
- [🙌 Credits](#-credits)  


## 🔍 Overview

This project was developed for the TDS (Tools in Data Science) course project, where the objective is to build an application that can autonomously solve multi-step quiz tasks involving:

- **Data sourcing**: Scraping websites, calling APIs, downloading files
- **Data preparation**: Cleaning text, PDFs, and various data formats
- **Data analysis**: Filtering, aggregating, statistical analysis, ML models
- **Data visualization**: Generating charts, narratives, and presentations

The system receives quiz URLs via a REST API, navigates through multiple quiz pages, solves each task using LLM-powered reasoning and specialized tools, and submits answers back to the evaluation server.

## 📌 Features

* ✔️ REST API with input validation using **Pydantic**
* ✔️ Secure request verification via a **secret key**
* ✔️ Background processing via FastAPI `BackgroundTasks`
* ✔️ Automatic scraping using **Playwright** (JavaScript-enabled pages supported)
* ✔️ Intelligent solving powered by **LangChain + GPT + Tools**
* ✔️ Supports:
  * Web scraping
  * POST submission
  * File downloads
  * Audio transcription using Whisper
  * OCR on images
  * CSV/XLSX/JSON/Parquet analysis with Pandas
  * Code execution in isolated sandbox
  * PDF text extraction
  * Plotting with matplotlib
  * API fetches with custom headers
  * Archive extraction and file handling
* ✔️ Retry logic, fallback strategies, and timeouts

## 🏗️ Architecture

```
┌─────────────┐
│   FastAPI   │  ← Receives POST requests with quiz URLs
│   Server    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Agent     │  ← Langchain orchestrator with GPT 4.1 Mini
│   (LLM)     │
└──────┬──────┘
       │
       ├───────────────┐────────────────┐──────────────────┐─────────────────┐─────────────────────┐──────────────────┐─────────────────┐──────────────┐──────────────┐─────────────────────┐────────────────┐──────────────┐
       ▼               ▼                ▼                  ▼                  ▼                     ▼                  ▼                 ▼              ▼              ▼                     ▼                ▼              ▼
 [scrape_data] [download_file] [transcribe_audio] [analyze_csv_data] [analyze_tabular_file] [send_post_request] [fetch_api_data] [add_dependencies] [run_code] [extract_pdf_text] [plot_with_matplotlib] [ocr_image] [extract_archive]


```

## 🧠 How It Works

1. Scrapes quiz page content.
2. LLM analyzes instructions using tools only when necessary.
3. Tools execute operations like scraping, OCR, file parsing, audio transcription, code evaluation, etc.
4. Sends a POST request with computed answer.
5. Extracts next quiz URL.
6. Repeats until no next URL is provided.

With fallback strategies including detecting submission endpoints automatically if the agent fails.

```
┌─────────────────────────────────────────┐
│ 1. LLM analyzes current state           │
│    - Reads quiz page instructions       │
│    - Plans tool usage                   │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ 2. Tool execution                       │
│    - Scrapes page / downloads files     │
│    - Runs analysis code                 │
│    - Submits answer                     │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ 3. Response evaluation                  │
│    - Checks if answer is correct        │
│    - Extracts next quiz URL (if exists) │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ 4. Decision                             │
│    - If new URL exists: Loop to step 1  │
│    - If no URL: exits loop              │
└─────────────────────────────────────────┘
```

## 📝 Key Design Decisions

1. **LangChain over Sequential Execution**: Allows flexible routing and complex decision-making
2. **Background Processing**: Prevents HTTP timeouts for long-running quiz chains
3. **Tool Modularity**: Each tool is independent and can be tested/debugged separately
4. **Code Execution**: Dynamically generates and runs Python for complex data tasks
5. **Playwright for Scraping**: Handles JavaScript-rendered pages that `requests` cannot
6. **uv for Dependencies**: Fast package resolution and installation

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```sh
git clone <your-repo-url>
cd <project-folder>
```

### 2️⃣ Create a Virtual Environment

Using `uv` (recommended):

```sh
uv venv
source .venv/bin/activate
```

Or using Python:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```sh
uv pip install -r requirements.txt
```

> ⚠️ Ensure Playwright browsers are installed:

```sh
playwright install chromium
```

---

## 🔐 Environment Variables

Create a `.env` file:

```
EXPECTED_SECRET=your_secret_here
EMAIL=your_email_here
AIPIPE_TOKEN=your_ai_model_token_here
AIPIPE_URL=https://aipipe.org/openrouter/v1/chat/completions
```



## ▶️ Running the Server

```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```



## 📡 API Endpoints

### ✔️ Health Check

```
GET /health
```

Response:

```json
{
  "status": "ok",
  "message": "Server is running 🚀"
}
```



### 🚀 Start Solving a Quiz

```
POST /quiz-task
```

#### Request Body:

```json
{
  "email": "your_email@example.com",
  "secret": "your_secret_string",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}
```

#### Successful Response:

```json
{
  "message": "Quiz processing started successfully in the background."
}
```

> The quiz will continue solving automatically until:
> ✔ Completed
> ❌ Error occurs
> ⏳ Timeout expires


## 🛠 Tools Available to the Agent

| Tool                     | Purpose                                |
| ------------------------ | -------------------------------------- |
| `scrape_data`            | Smart browser scraping with Playwright |
| `send_post_request`      | Submit quiz answers                    |
| `run_code`               | Execute sandboxed Python               |
| `transcribe_audio`       | Whisper audio transcription            |
| `download_file`          | Download CSV/PDF/ZIP etc               |
| `analyze_csv_data`       | Pandas execution w/ expressions        |
| `extract_pdf_text`       | PDF parsing                            |
| `ocr_image`              | OCR via Tesseract                      |
| `add_dependencies`       | Install packages dynamically           |
| `plot_with_matplotlib`   | Create and return plot images          |
| `extract_archive`        | Extract tar/zip archives               |
| `fetch_api_data`         | Authenticated GET Requests             |
| `analyze_tabular_file`   | Smart detection + data analysis        |
| `encode_image_to_base64` | Base64 encoding utility                |



## ⏱ Timeouts & Limits

| Item                        | Limit          |
| --------------------------- | -------------- |
| Per-quiz total solving time | **1 hour**     |
| Per question                | **150s**       |
| Retries per question        | **2 attempts** |



## 🧪 Logging & Debugging

All major decisions, errors, and retries print to stdout.




## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



## 🙌 Credits

Built using:

* FastAPI
* LangChain
* Playwright
* Whisper
* Pandas
* aio/uv tooling

---

**Author**: Namit Gupta
**Course**: Tools in Data Science (TDS)
**Institution**: IIT Madras

For questions or issues, please open an issue on the [GitHub repository](https://github.com/23f2003526/tds-project-2).
