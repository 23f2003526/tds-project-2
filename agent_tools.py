# agent_tools.py
import os
import requests
import pandas as pd
from typing import Any
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# --- Tool 1: Playwright Scraper ---
@tool
def scrape_data(url: str, selector: str = 'body') -> str:
    """Scrape the text content of a given CSS selector on a JavaScript-rendered page."""
    return f"Scraped content from {url} selector {selector}: ... (Placeholder for Playwright output)"

# --- Tool 2: File Downloader ---
@tool
def download_file(url: str, local_filename: str) -> str:
    """Downloads a file (e.g., CSV, PDF) to a local path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return f"File saved to: {local_filename}"
    except Exception as e:
        return f"Error downloading {url}: {e}"

# --- Tool 3: Pandas Analyzer ---
@tool
def calculate_sum_of_column(file_path: str, column_name: str) -> str:
    """Calculates the sum of a numeric column in a CSV."""
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found."
        return f"The sum of '{column_name}' is {df[column_name].sum()}"
    except Exception as e:
        return f"Error analyzing {file_path}: {e}"

# --- Agent Factory ---
def get_langchain_agent() -> Any:
    """
    Creates a modern ReAct-style agent compatible with LangChain 1.0+.
    """
    tools = [scrape_data, download_file, calculate_sum_of_column]

    agent = create_agent(
        "gemini",
        tools=tools,
        system_prompt="You are a helpful assistant."
    )
    return agent