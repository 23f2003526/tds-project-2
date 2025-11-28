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

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121.3+-green.svg)](https://fastapi.tiangolo.com/)

An intelligent, autonomous agent built with LangGraph that solves data-related quizzes involving web scraping, data processing, analysis, and visualization tasks. The system uses OpenAI's GPT-4.1-Mini model to orchestrate tool usage and make decisions.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Tools &amp; Capabilities](#tools--capabilities)
- [Docker Deployment](#docker-deployment)
- [How It Works](#how-it-works)
- [License](#license)

## 🔍 Overview

This project was developed for the TDS (Tools in Data Science) course project, where the objective is to build an application that can autonomously solve multi-step quiz tasks involving:

- **Data sourcing**: Scraping websites, calling APIs, downloading files
- **Data preparation**: Cleaning text, PDFs, and various data formats
- **Data analysis**: Filtering, aggregating, statistical analysis, ML models
- **Data visualization**: Generating charts, narratives, and presentations

The system receives quiz URLs via a REST API, navigates through multiple quiz pages, solves each task using LLM-powered reasoning and specialized tools, and submits answers back to the evaluation server.

## 🏗️ Architecture

The project uses a **LangGraph state machine** architecture with the following components:

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

### Key Components:

1. **FastAPI Server** (`main.py`): Handles incoming POST requests, validates secrets, and triggers the agent
2. **LangGraph Agent** (`agent.py`): State machine that coordinates tool usage and decision-making
3. **Tools Package** (`tools/`): Modular tools for different capabilities
4. **LLM**: OpenAI GPT-4.1-Mini

## ✨ Features

- ✅ **Autonomous multi-step problem solving**: Chains together multiple quiz pages
- ✅ **Dynamic JavaScript rendering**: Uses Playwright for client-side rendered pages
- ✅ **Code generation & execution**: Writes and runs Python code for data tasks
- ✅ **Flexible data handling**: Downloads files, processes PDFs, CSVs, images, etc.
- ✅ **Self-installing dependencies**: Automatically adds required Python packages
- ✅ **Robust error handling**: Retries failed attempts within time limits
- ✅ **Docker containerization**: Ready for deployment on HuggingFace Spaces or cloud platforms
- ✅ **Rate limiting**: Respects API quotas with exponential backoff

## 📁 Project Structure

```
LLM-Analysis-TDS-Project-2/
├── quiz_solver.py                    # LangChain state machine & orchestration
├── main.py                     # FastAPI server with /solve endpoint
├── pyproject.toml              # Project dependencies & configuration
├── Dockerfile                  # Container image with Playwright
├── .env                        # Environment variables (not in repo)
├── agent_tools.py
└── README.md
```

## 🧠 How It Works

### 1. Request Reception

- FastAPI receives a POST request with quiz URL
- Validates the secret against environment variables
- Returns 200 OK and starts the agent in the background

### 2. Agent Initialization

- Langchain....
- The initial state contains the quiz URL as a user message

### 3. Task Loop

The agent follows this loop:

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
│    - If no URL: Return "END"            │
└─────────────────────────────────────────┘
```

### 4. State Management

- All messages (user, assistant, tool) are stored in state
- The LLM uses full history to make informed decisions
- Recursion limit set to 200 to handle long quiz chains

### 5. Completion

- Agent returns "END" when no new URL is provided
- Background task completes
- Logs indicate success or failure

## 📝 Key Design Decisions

1. **LangGraph over Sequential Execution**: Allows flexible routing and complex decision-making
2. **Background Processing**: Prevents HTTP timeouts for long-running quiz chains
3. **Tool Modularity**: Each tool is independent and can be tested/debugged separately
4. **Code Execution**: Dynamically generates and runs Python for complex data tasks
5. **Playwright for Scraping**: Handles JavaScript-rendered pages that `requests` cannot
6. **uv for Dependencies**: Fast package resolution and installation

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Author**: Namit Gupta
**Course**: Tools in Data Science (TDS)
**Institution**: IIT Madras

For questions or issues, please open an issue on the [GitHub repository](https://github.com/...).
