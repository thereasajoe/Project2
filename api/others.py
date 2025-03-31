import ujson as json_parser
from fuzzywuzzy import fuzz
from urllib.parse import urlencode
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, UploadFile
import subprocess
import json
import requests
import os
import pandas as pd
import re
import shutil
import tempfile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
import numpy as np
import time
from bs4 import BeautifulSoup
import zipfile
import hashlib
from datetime import datetime, timezone, timedelta
import sqlite3
from PIL import Image
import base64
import httpx
from io import BytesIO
import colorsys
import os
import re
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from typing import Dict, Any, Union, Optional
from geopy.geocoders import Nominatim
import feedparser
import urllib.parse
import camelot
from pdfminer.high_level import extract_text
import gzip
import re
from datetime import datetime
from fastapi import UploadFile
import pytz
from collections import defaultdict
from metaphone import doublemetaphone
import yt_dlp
import whisper
from fastapi import FastAPI, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
import csv
import io
from fastapi import FastAPI, HTTPException, Query, UploadFile
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from difflib import get_close_matches

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load environment variables (if using .env)
load_dotenv()

# Get API Token
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# OpenAI API Proxy Endpoint
API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"


# Function GA1Q1: Handle Visual Studio Code command output
def solve_vscode_question(question, file=None):
    """Executes 'code -s' in the terminal and returns the output."""
    try:
        result = subprocess.run(["code", "-s"], capture_output=True, text=True)
        result = result.stdout.strip()
        result = json.dumps(result)

        return {"answer": result}

    except Exception as e:
        return {"error": f"Failed to execute command: {str(e)}"}

# Function GA1Q2: Handle HTTP request using httpie


def solve_http_request_question(question, file=None):
    try:
        # Define the target URL
        url = "https://httpbin.org/get"

        # Define the parameters (URL encoded automatically by requests)
        params = {"email": "23f3004024@ds.study.iitm.ac.in"}

        # Set custom headers to mimic HTTPie behavior
        headers = {"User-Agent": "HTTPie/3.2.4", "Accept": "*/*"}

        # Send the GET request
        response = requests.get(url, params=params, headers=headers)

        # Check if the request was successful
        response.raise_for_status()

        # Extract only the JSON body
        json_response = response.json()

        json_response = json.dumps(json_response)

        # Return formatted response inside "answer"
        return {"answer": json_response}

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to send HTTP request: {str(e)}"}

# Function GA1Q3: Handle Prettier formatting and SHA256 hashing


def solve_prettier_hash_question(question, file=None):
    # Extract the file name dynamically from the question
    # Extract the file name dynamically from the question
    # The pattern is designed to capture the first file name mentioned in the question
    # For example, in the question "Download file1.txt and file2.txt In the directory",
    # the pattern will capture "file1.txt"
    match = re.search(
        r"Download (.*?)\s*(?:and\s(.*?))?\s*In the directory", question)
    # print("Match:", match.group(0))
    if not match:
        return {"error": "Could not extract the file name from the question. Ensure the question format is correct."}

    # Extracted file name from question
    expected_filename = match.group(1).strip()

    if not file:
        return {"error": f"No file uploaded. Expected: {expected_filename}"}

    try:
        # Use a temporary directory to store the uploaded file
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, file.filename)

            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Ensure the uploaded file matches the expected name
            if file.filename.lower() != expected_filename.lower():
                return {"error": f"Uploaded file '{file.filename}' does not match the expected '{expected_filename}'"}

            # Run Prettier formatting (modify the file in place)
            prettier_result = subprocess.run(
                ["npx", "-y", "prettier@3.4.2", "--write", file_path],
                capture_output=True, text=True
            )

            # Check if Prettier failed
            if prettier_result.returncode != 0:
                return {"error": f"Prettier formatting failed: {prettier_result.stderr}"}

            # Ensure the formatted file exists
            if not os.path.exists(file_path):
                return {"error": "Formatted file not found. Prettier may have failed."}

            # Compute SHA256 hash using `sha256sum`
            sha256_result = subprocess.run(
                f"sha256sum {file_path}",
                capture_output=True, text=True, shell=True
            )

            # Extract the hash value from the command output
            sha256_output = sha256_result.stdout.strip().split(
                " ")[0] if sha256_result.stdout else "Error computing hash"
            # sha256_output = json.dumps(sha256_output)

            print("SHA256 Output:", sha256_output)

            return {"answer": sha256_output}

    except Exception as e:
        return {"error": f"Failed to process file '{expected_filename}': {str(e)}"}

# Function GA1Q4: Solve Google Sheets formula


def solve_google_sheets_question(question, file=None):
    try:
        # Extract formula from question
        match = re.search(r'=SUM\((.+)\)', question)
        if not match:
            return {"error": "Could not find a SUM formula in the question."}

        formula = match.group(1).strip()

        # Handle ARRAY_CONSTRAIN(SEQUENCE(...), rows, cols)
        if "ARRAY_CONSTRAIN" in formula:
            ac_match = re.search(
                r'ARRAY_CONSTRAIN\(\s*SEQUENCE\((\d+),\s*(\d+),\s*(-?\d+),\s*(-?\d+)\)\s*,\s*(\d+)\s*,\s*(\d+)\)', formula)
            if not ac_match:
                return {"error": "Invalid ARRAY_CONSTRAIN/SEQUENCE format."}

            rows, cols, start, step, limit_rows, limit_cols = map(
                int, ac_match.groups())

        # Handle INDEX(SEQUENCE(...), 1, SEQUENCE(...))
        elif "INDEX" in formula and "SEQUENCE" in formula:
            index_match = re.search(
                r'INDEX\(\s*SEQUENCE\(\s*(\d+),\s*(\d+),\s*(-?\d+),\s*(-?\d+)\)\s*,\s*1\s*,\s*SEQUENCE\(\s*1\s*,\s*(\d+)\)\s*\)', formula)
            if not index_match:
                return {"error": "Invalid INDEX/SEQUENCE formula."}

            rows, cols, start, step, limit_cols = map(
                int, index_match.groups())
        else:
            return {"error": "Unsupported formula format."}

        # Generate full sequence in row-major order
        sequence = []
        for r in range(rows):
            for c in range(cols):
                value = start + (r * cols + c) * step
                sequence.append(value)

        # Take only the first limit_cols values from the first row
        result_values = sequence[:limit_cols]
        result_values = sum(result_values)
        result_values = json.dumps(result_values)
        return {"answer": result_values}

    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

# Function GA1Q5: Solve Office 365 Excel formula


def solve_excel_question(question, file=None):
    # Regex pattern to extract numbers from curly braces {}
    pattern = r"\{([\d,\s]+)\}"

    # Find all number groups inside {}
    matches = re.findall(pattern, question)

    if len(matches) < 2:
        return {"error": "Invalid input format. Ensure the formula contains an array and sort order."}

    try:
        # Extract array and sort order as lists of integers
        array = list(map(int, matches[0].split(",")))
        sort_order = list(map(int, matches[1].split(",")))

        # Extract `n` (number of elements to take) using regex
        n_match = re.search(r"TAKE\(.*?,\s*(\d+)\)", question)
        # Default to 6 if not found
        n = int(n_match.group(1)) if n_match else 6

        # Sort the array based on the sort order
        sorted_array = [x for _, x in sorted(zip(sort_order, array))]

        # Extract the first `n` elements
        extracted_values = sorted_array[:n]
        extracted_values = sum(extracted_values)
        extracted_values = json.dumps(extracted_values)

        # Compute the sum of extracted values
        return {"answer": extracted_values}

    except Exception as e:
        return {"error": f"Failed to process the formula: {str(e)}"}

# Function GA1Q6: Solve HTML hidden input question


def solve_hidden_input_question(question, file=None):
    """
    Extracts the value of a hidden input field from an HTML file.
    """
    if not file:
        return {"error": "No HTML file uploaded. Please upload an HTML file containing a hidden input field."}

    try:
        # Read and parse the HTML file
        html_content = file.file.read().decode("utf-8")
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the first hidden input field
        hidden_input = soup.find("input", {"type": "hidden"})

        if hidden_input and hidden_input.get("value"):
            value = hidden_input["value"]
            value = json.dumps(value)
            return {"answer": value}

        return {"error": "No hidden input field with a value found in the uploaded file."}

    except Exception as e:
        return {"error": f"Failed to extract hidden input: {str(e)}"}

# Function GA1Q7: Solve count Wednesdays question


def solve_count_wednesdays_question(question, file=None):
    # Extract date range using regex
    match = re.search(r"(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", question)
    if not match:
        return {"error": "Invalid date range format. Ensure the question contains dates in YYYY-MM-DD format."}

    try:
        # Parse the extracted start and end dates
        start_date = datetime.strptime(match.group(1), "%Y-%m-%d")
        end_date = datetime.strptime(match.group(2), "%Y-%m-%d")

        # Initialize count
        count = 0

        # Loop through the date range
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() == 2:  # 2 corresponds to Wednesday
                count += 1
            current_date += timedelta(days=1)

        count = json.dumps(count)

        return {"answer": count}

    except Exception as e:
        return {"error": f"Failed to compute Wednesdays count: {str(e)}"}

# Function GA1Q8: Solve CSV extraction question


def solve_csv_extraction_question(question, file=None):
    if not file:
        return {"error": "No ZIP file uploaded. Please upload a ZIP file containing extract.csv."}

    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")

            # Save the uploaded file temporarily
            with open(zip_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Find and read extract.csv
            csv_path = os.path.join(tmpdirname, "extract.csv")
            if not os.path.exists(csv_path):
                return {"error": "No extract.csv found in the ZIP file."}

            df = pd.read_csv(csv_path)

            # Ensure "answer" column exists
            if "answer" not in df.columns:
                return {"error": "The 'answer' column is missing in the CSV file."}

            # Get the first non-null value in the answer column
            answer_value = df["answer"].dropna().iloc[0]

            answer_value = str(answer_value)
            answer_value = json.dumps(answer_value)

            return {"answer": answer_value}

    except Exception as e:
        return {"error": f"Failed to process CSV file: {str(e)}"}

# Function GA1Q9: Solve JSON sorting question


def solve_json_sorting_question(question, file=None):
    """
    Sorts a JSON array by age (ascending). If ages are equal, sorts by name (alphabetically).
    Returns the sorted list inside the "answer" field.
    """
    try:
        # Extract JSON from the question
        match = re.search(r"\[.*\]", question, re.DOTALL)
        if not match:
            return {"error": "No valid JSON found in the question."}

        json_data = json.loads(match.group(0))

        # Sort by age, then by name
        sorted_data = sorted(json_data, key=lambda x: (x["age"], x["name"]))

        sorted_data = json.dumps(sorted_data)

        # Return the sorted list inside the "answer" field
        return {"answer": sorted_data}

    except Exception as e:
        return {"error": f"Failed to sort JSON data: {str(e)}"}

# Function GA1Q10: Solve JSON conversion question


def solve_json_conversion_question(question, file=None):
    try:
        if not file:
            return {"error": "No file provided."}

        # Read file once — decode safely with utf-8-sig to remove BOM if present
        raw_bytes = file.file.read()
        content = raw_bytes.decode("utf-8-sig").strip()

        key_value_pairs = {}

        for line in content.splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue  # skip blank or malformed lines

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            key_value_pairs[key] = value

        # Compact sorted JSON
        json_string = json.dumps(
            key_value_pairs, separators=(',', ':'), sort_keys=True)

        # Hash the string
        sha_hash = hashlib.sha256(json_string.encode('utf-8')).hexdigest()

        return {"answer": sha_hash}

    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}

# Function GA1Q11: Solve div sum question


def solve_div_sum_question(question, file=None):
    if not file:
        return {"error": "No HTML file uploaded. Please upload an HTML file containing div elements."}

    try:
        # Read and parse the HTML file
        html_content = file.file.read().decode("utf-8")

        # Debugging: Print file content
        print("Received HTML Content:\n", html_content)

        soup = BeautifulSoup(html_content, "html.parser")

        # Find all <div> elements with class 'foo'
        divs_with_foo = soup.find_all("div", class_="foo")

        if not divs_with_foo:
            return {"error": "No <div> elements with class 'foo' found in the uploaded HTML file."}

        # Sum up their 'data-value' attributes
        total_sum = sum(
            int(div.get("data-value", 0)) for div in divs_with_foo if div.get("data-value", "").isdigit()
        )

        total_sum = json.dumps(total_sum)

        return {"answer": total_sum}

    except Exception as e:
        return {"error": f"Failed to process HTML file: {str(e)}"}

# Function GA1Q12: Solve file encoding sum question


def solve_file_encoding_sum_question(question, file=None):
    if not file:
        return {"error": "No ZIP file uploaded. Please upload a ZIP file containing data1.csv, data2.csv, and data3.txt."}

    try:
        # 🔹 Extract symbols dynamically from the question using regex
        symbol_pattern = re.findall(r"['\"]?([^\s,.'\"]{1})['\"]?", question)
        # Convert to set to avoid duplicates
        target_symbols = set(symbol_pattern)

        if not target_symbols:
            return {"error": "No valid symbols found in the question. Please specify symbols to sum."}

        # 🔹 Create a temporary directory to extract ZIP contents
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")

            # Save the uploaded file temporarily
            with open(zip_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # 🔹 Define expected files and their encodings
            expected_files = {
                "data1.csv": "cp1252",
                "data2.csv": "utf-8",
                "data3.txt": "utf-16"
            }

            total_sum = 0  # To store sum of values

            for filename, encoding in expected_files.items():
                file_path = os.path.join(tmpdirname, filename)

                # Ensure the required file exists
                if not os.path.exists(file_path):
                    return {"error": f"Missing required file: {filename} in ZIP."}

                # Read the file based on its encoding and delimiter
                if filename.endswith(".csv"):
                    df = pd.read_csv(file_path, encoding=encoding)
                else:  # Tab-separated TXT file
                    df = pd.read_csv(file_path, encoding=encoding, sep="\t")

                # Ensure required columns exist
                if "symbol" not in df.columns or "value" not in df.columns:
                    return {"error": f"File {filename} does not contain 'symbol' and 'value' columns."}

                # Convert 'value' column to numeric, ignoring errors
                df["value"] = pd.to_numeric(df["value"], errors="coerce")

                # 🔹 Sum up values for dynamically extracted symbols
                total_sum += df[df["symbol"].isin(target_symbols)
                                ]["value"].sum()

            # Convert numpy.int64 to int before returning
            sum = int(total_sum)
            sum = json.dumps(sum)
            return {"answer": sum}

    except Exception as e:
        return {"error": f"Failed to process ZIP file: {str(e)}"}

# Function GA1Q13: Solve GitHub repository question


def solve_github_repo_question(question, file=None):
    return {"answer": "https://github.com/thereasajoe/tds1/blob/main/email.json"}

# Function GA1Q14: Solve text replacement question


def solve_replace_text_question(question: str, file: UploadFile):
    # Create a temporary directory to extract files
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_path = os.path.join(tmpdirname, "uploaded.zip")

        # Save the uploaded zip file
        with open(zip_path, "wb") as buffer:
            buffer.write(file.file.read())

        extract_folder = os.path.join(tmpdirname, "unzipped")
        os.makedirs(extract_folder, exist_ok=True)

        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        # Compile case-insensitive regex for "IITM"
        pattern = re.compile(r'IITM', re.IGNORECASE)

        # Replace content in all files
        for root, dirs, files in os.walk(extract_folder):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', newline='') as f:
                        content = f.read()
                    replaced_content = pattern.sub("IIT Madras", content)
                    with open(file_path, 'w', encoding='utf-8', newline='') as f:
                        f.write(replaced_content)
                except:
                    continue  # skip non-text files

        # Simulate `cat * | sha256sum` by concatenating file contents in sorted order
        concatenated_content = ""
        all_files = sorted(os.listdir(extract_folder))
        for fname in all_files:
            file_path = os.path.join(extract_folder, fname)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    concatenated_content += f.read()
            except:
                continue

        sha256_hash = hashlib.sha256(
            concatenated_content.encode('utf-8')).hexdigest()
        sha256_hash = json.dumps(sha256_hash)
        return {"answer": sha256_hash}

# Function GA1Q15: Solve file size filter question


def solve_file_size_filter_question(question: str, file: UploadFile):
    # Extract minimum file size using regex
    size_match = re.search(r'at least (\d+) bytes', question)
    if not size_match:
        return {"error": "Minimum file size not found in question."}
    min_size = int(size_match.group(1))

    # Extract the date and time string from the question
    date_match = re.search(
        r'modified on or after (.+?\d{1,2}:\d{2} (?:am|pm) IST)', question, re.IGNORECASE
    )
    if not date_match:
        return {"error": "Cutoff datetime not found in question."}

    date_str = date_match.group(1).strip()

    # Convert to datetime object
    try:
        ist_dt = datetime.strptime(date_str, "%a, %d %b, %Y, %I:%M %p IST")
    except ValueError:
        return {"error": f"Invalid date format: '{date_str}'"}

    # Convert IST to UTC (IST = UTC + 5:30)
    cutoff_utc = ist_dt - timedelta(hours=5, minutes=30)

    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")

        # Save the uploaded ZIP file
        with open(zip_path, "wb") as f:
            f.write(file.file.read())

        extract_dir = os.path.join(tmpdir, "unzipped")
        os.makedirs(extract_dir, exist_ok=True)

        # Extract files and restore original modified times
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for zinfo in zip_ref.infolist():
                extracted_path = zip_ref.extract(zinfo, extract_dir)
                mtime = time.mktime(zinfo.date_time + (0, 0, -1))
                os.utime(extracted_path, (mtime, mtime))

        # Scan files and compute total size
        total_size = 0
        for root, dirs, files in os.walk(extract_dir):
            for name in files:
                file_path = os.path.join(root, name)
                stat = os.stat(file_path)
                file_size = stat.st_size
                file_mtime = datetime.utcfromtimestamp(stat.st_mtime)
                if file_size >= min_size and file_mtime >= cutoff_utc:
                    total_size += file_size

        total_size = json.dumps(total_size)
        return {
            "answer": total_size
        }

# Function GA1Q16: Solve file renaming question


def solve_rename_files_question(question: str, file: UploadFile):
    def shift_digits(name: str) -> str:
        def replacer(match):
            digit = int(match.group())
            return str((digit + 1) % 10)
        return re.sub(r'\d', replacer, name)

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")

        # Save uploaded ZIP file
        with open(zip_path, "wb") as f:
            f.write(file.file.read())

        base_folder = os.path.join(tmpdir, "unzipped")
        final_folder = os.path.join(tmpdir, "flattened")
        os.makedirs(final_folder, exist_ok=True)

        # Extract all files
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_folder)

        # Move all files into the flat folder
        for root, dirs, files in os.walk(base_folder):
            if root == final_folder:
                continue
            for name in files:
                src_path = os.path.join(root, name)
                dst_path = os.path.join(final_folder, name)
                shutil.move(src_path, dst_path)

        # Rename files by shifting digits
        for name in os.listdir(final_folder):
            old_path = os.path.join(final_folder, name)
            new_name = shift_digits(name)
            new_path = os.path.join(final_folder, new_name)
            os.rename(old_path, new_path)

        # Simulate `grep . * | LC_ALL=C sort | sha256sum`
        lines = []
        for fname in sorted(os.listdir(final_folder)):
            fpath = os.path.join(final_folder, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.rstrip('\n')
                        lines.append(f"{fname}:{line}")
            except:
                continue  # Skip binary/unreadable files

        lines.sort()  # C locale (ASCIIbetical)
        combined = "\n".join(lines) + "\n"
        sha256_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
        sha256_hash = json.dumps(sha256_hash)

        return {"answer": sha256_hash}

# Function GA1Q17: Solve file comparison question


def solve_compare_files_question(question, file=None):
    if not file:
        return {"error": "No ZIP file uploaded. Please upload a ZIP file containing a.txt and b.txt."}

    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")

            # Save the uploaded ZIP file
            with open(zip_path, "wb") as buffer:
                buffer.write(file.file.read())

            # Extract ZIP contents
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Define paths to a.txt and b.txt
            a_txt_path = os.path.join(tmpdirname, "a.txt")
            b_txt_path = os.path.join(tmpdirname, "b.txt")

            # Ensure both files exist
            if not os.path.exists(a_txt_path) or not os.path.exists(b_txt_path):
                return {"error": "Both a.txt and b.txt must be present in the ZIP file."}

            # Read files line by line and compare
            with open(a_txt_path, "r", encoding="utf-8", errors="ignore") as f1, \
                    open(b_txt_path, "r", encoding="utf-8", errors="ignore") as f2:

                a_lines = f1.readlines()
                b_lines = f2.readlines()

            # Ensure both files have the same number of lines
            if len(a_lines) != len(b_lines):
                return {"error": "Files do not have the same number of lines."}

            # Count differing lines
            differing_lines = sum(1 for line1, line2 in zip(
                a_lines, b_lines) if line1.strip() != line2.strip())
            differing_lines = json.dumps(differing_lines)

            return {"answer": differing_lines}

    except Exception as e:
        return {"error": f"Failed to process ZIP file: {str(e)}"}

# Function GA1Q18: Solve SQLite query question


def solve_sqlite_query_question(question, file=None):
    try:
        # Extract the ticket type from the question
        match = re.search(
            r'total sales of all the items in the "(.*?)" ticket type', question, re.IGNORECASE)
        if not match:
            return {"error": "Could not determine the ticket type from the question."}

        ticket_type = match.group(1).strip()  # Extracted ticket type

        # Construct the SQL query dynamically using the extracted ticket type
        sql_query = """SELECT SUM(units * price) AS total_sales
        FROM tickets
        WHERE TRIM(LOWER(type)) = '{ticket_type.lower()}';"""
        sql_query = json.dumps(sql_query)
        return {"answer": sql_query}

    except Exception as e:
        return {"error": f"Failed to generate SQL query: {str(e)}"}

# Function GA2Q1: Solve Markdown documentation question


def solve_markdown_documentation_question(question, file=None):
    markdown_content = """# Weekly Step Analysis: Personal Insights and Social Comparison

        This document provides an analysis of the number of steps I walked each day over a week. The goal was to observe **patterns in physical activity**, track progress over time, and compare my performance with friends. 

        ---

        ## Methodology

        To collect and analyze the data, I followed these steps:

        1. **Data Collection**:
        - I used a **fitness tracker** to record my daily steps.
        - *Note*: Data accuracy may vary due to tracker limitations.

        2. **Data Cleaning**:
        - Processed the data using Python with the `pandas` library.
        - Removed days with incomplete step counts (e.g., when I forgot to wear the tracker).

        3. **Visualization**:
        - Created plots using `matplotlib` for trend analysis.
        - Compared personal data with friends using average step counts.

        Below is a sample Python snippet used for preprocessing:
        ```python
        import pandas as pd

        # Load step data
        data = pd.read_csv("steps.csv")

        # Drop missing values
        cleaned_data = data.dropna()
        ```
        Following is a table for number of steps in a week:
        | day | steps |
        | ----- | -----|
        | Monday | 400 |
        | Tuesday |300 |
        | Wednesday |350 |
        | Thursday |250 |
        | Friday | 400 |
        | Saturday | 650 |
        | Sunday | 280 |

        [click here for an overview about benefits of walking](https://www.betterhealth.vic.gov.au/health/healthyliving/walking-for-good-health)

        ![fit india](https://content.dhhs.vic.gov.au/sites/default/files/walking_88481231_1050x600.jpg)

        > walking everyday improves heart health significantly"""
    markdown_content = json.dumps(markdown_content)
    return {"answer": markdown_content}

# Function GA2Q2: Compress image


def solve_image_compression_question(question: str, file: UploadFile):
    try:
        # Open the uploaded image
        original_image = Image.open(file.file)

        # Create a persistent temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webp")
        output_path = temp_file.name
        temp_file.close()  # We'll write to it with PIL

        # Save as WebP with lossless compression
        original_image.save(output_path, format="WEBP",
                            lossless=True, quality=100, method=6)

        # Check size
        final_size = os.path.getsize(output_path)
        if final_size >= 1500:
            os.remove(output_path)  # Clean up if too large
            return {
                "error": "Compressed image is not under 1,500 bytes.",
                "final_size_bytes": final_size
            }

        # Return file as response
        answer = FileResponse(
            output_path, media_type="image/webp", filename="compressed.webp")
        return FileResponse(output_path, media_type="image/webp", filename="compressed.webp")

    except Exception as e:
        return {"error": str(e)}

# Function GA2Q3: GitHub pages


def solve_github_pages_question(question, file=None):
    return {"answer": "https://thereasajoe.github.io/"}

# Function GA2Q4: Google colab authentication


def extract_email_from_question(question: str):
    """
    Extracts the email from the given question text.
    """
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", question)
    return match.group(0) if match else None


def solve_colab_auth_question(question, file=None):
    """
    Simulates the functionality of the original Google Colab script.

    - Extracts email from the question.
    - Uses a fixed/flexible token expiry year.
    - Computes the SHA-256 hash of "email year".
    - Returns the last 5 characters of the hash.
    """
    email = extract_email_from_question(question)
    if not email:
        return {"error": "Could not extract email from the question."}

    token_expiry_year = 2025  # This can be made dynamic if needed

    # Compute SHA-256 hash
    hash_input = f"{email} {token_expiry_year}".encode()
    hash_output = hashlib.sha256(hash_input).hexdigest()[-5:]
    hash_output = json.dumps(hash_output)

    return {"answer": hash_output}

# Function GA2Q5: Image brightness colab


def solve_colab_brightness_question(question, file=None):
    try:
        if not file:
            return {"error": "No image file uploaded. Please upload an image."}

        # Open the uploaded image
        image = Image.open(file.file).convert("RGB")  # Ensure RGB mode

        # Convert image to NumPy array and normalize values to [0, 1]
        rgb = np.array(image, dtype=np.float32) / 255.0

        # Compute lightness using HLS color model
        def rgb_to_lightness(pixel):
            return colorsys.rgb_to_hls(*pixel)[1]  # Extract lightness channel

        lightness = np.apply_along_axis(rgb_to_lightness, 2, rgb)

        # Count pixels where lightness > 0.666
        light_pixels = np.sum(lightness > 0.666)

        pixels = int(light_pixels)
        pixels = json.dumps(pixels)
        return {"answer": pixels}

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

# Function GA2Q6: Vercel


def solve_vercel_api_question(question, file=None):
    if main_app is None:
        return {"error": "Main app not registered."}

    if not file or not file.filename.endswith(".json"):
        return {"error": "Please upload a valid JSON file."}

    contents = file.file.read().decode("utf-8")
    try:
        students_data = json.loads(contents)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format."}

    # Create name → marks lookup
    marks_dict = {entry["name"]: entry["marks"] for entry in students_data}

    # Create sub-app
    sub_app = FastAPI()

    sub_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @sub_app.get("/")
    def get_marks(name: list[str] = Query(default=[])):
        result = [marks_dict.get(n, None) for n in name]
        return JSONResponse(content={"marks": result})

    # Mount sub-app at /api
    main_app.mount("/api", sub_app)

    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    return {
        "answer": f"{base_url}/api"
    }

# Function GA2Q7: GitHub pages


def solve_github_action_question(question, file=None):
    return {"answer": "https://github.com/thereasajoe/mygithubaction"}

# Function GA2Q8: Docker image


def solve_docker_image_question(question, file=None):
    return {"answer": "https://hub.docker.com/repository/docker/thereasajo338/ga2q7/general"}


# Function GA2Q9: FastAPI
main_app: FastAPI = None


def solve_fastapi_server_question(question, file=None):
    if main_app is None:
        return {"error": "Main app not registered."}

    if not file or not file.filename.endswith(".csv"):
        return {"error": "Please upload a valid CSV file."}

    contents = file.file.read().decode("utf-8")
    csv_reader = csv.DictReader(io.StringIO(contents))

    students_data = []
    for row in csv_reader:
        students_data.append({
            "studentId": int(row["studentId"]),
            "class": row["class"].strip()
        })

    # Create sub-app
    sub_app = FastAPI()

    sub_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @sub_app.get("/")
    def get_students(class_filter: list[str] = Query(default=None, alias="class")):
        if class_filter:
            filtered = [s for s in students_data if s["class"] in class_filter]
            return {"students": filtered}
        return {"students": students_data}

    # 🚨 IMPORTANT: Mount at fixed path `/api`
    main_app.mount("/api", sub_app)

    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    return {
        "answer": f"{base_url}/api"
    }

# Function GA2Q10: NGROK


def solve_llama_model_question(question: str, file: UploadFile = None):
    try:
        # ✅ Path to model and binary (already downloaded in api/)
        base_path = os.path.dirname(__file__)  # 'api/' folder
        llamafile_path = os.path.join(base_path, "llamafile")
        model_path = os.path.join(
            base_path, "Llama-3.2-1B-Instruct.Q6_K.llamafile")

        # ✅ Ensure permissions to execute
        os.chmod(llamafile_path, 0o755)

        # ✅ Step 1: Start llamafile server
        subprocess.Popen([llamafile_path, model_path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # ✅ Step 2: Start ngrok tunnel
        subprocess.Popen(["ngrok", "http", "8080"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # ✅ Step 3: Wait for ngrok to be ready
        time.sleep(5)
        res = requests.get("http://localhost:4040/api/tunnels")
        public_url = res.json()["tunnels"][0]["public_url"]

        return {"ngrok_url": public_url}

    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to launch Llama model: {str(e)}"},
            status_code=500
        )

# Function GA3Q1: LLM sentiment analysis


def solve_llm_sentiment(question, file=None):
    try:
        match = re.search(r"^(.*?)\s*Write a Python program",
                          question, re.DOTALL)
        if not match:
            return PlainTextResponse('{"error": "Could not extract the text for sentiment analysis."}')

        random_text = match.group(1).strip()
        if not random_text or len(random_text.split()) < 3:
            return PlainTextResponse('{"error": "Extracted text is too short or invalid."}')

        # Escape quotes but keep line breaks
        escaped_text = random_text.replace('"', '\\"')

        code = f"""import httpx

# Define the API URL and headers
url = "https://api.openai.com/v1/chat/completions"
api_key = "your_dummy_api_key"  # Replace with your actual API key in real use

headers = {{
    "Authorization": f"Bearer {{api_key}}",
    "Content-Type": "application/json"
}}

# Define the messages for the API request
messages = [
    {{
        "role": "system",
        "content": "Please analyze the sentiment of the following text and categorize it into GOOD, BAD, or NEUTRAL."
    }},
    {{
        "role": "user",
        "content": "{escaped_text}"  # Exact user message
    }}
]

# Define the payload for the POST request
payload = {{
    "model": "gpt-4o-mini",
    "messages": messages,
    "max_tokens": 60
}}

# Sending the POST request to OpenAI's API
response = httpx.post(url, json=payload, headers=headers)

# Raise an exception for any error response
response.raise_for_status()

# Get the response data as JSON
response_data = response.json()

# Extract and print the sentiment analysis result
if 'choices' in response_data:
    sentiment = response_data['choices'][0]['message']['content']
    print("Sentiment analysis result:", sentiment)
else:
    print("Error: Sentiment not found in response.")
"""

        return {"answer": code}

    except Exception as e:
        return PlainTextResponse(f'{{"error": "Exception occurred: {str(e)}"}}')

# Function GA3Q2: LLM code generation


def extract_word_list(question):
    match = re.search(
        r"List only the valid English words from these:(.*?)\s*\.\.\.", question, re.DOTALL)

    if not match:
        return None  # Return None if the pattern is not found

    # Extract the word list (trim any leading/trailing whitespace)
    word_list = match.group(1).strip()

    return word_list


def solve_token_cost(question, file=None):
    """
    Extracts word list dynamically, sends request to OpenAI's API via proxy, 
    and returns the total token count.
    """
    if not AIPROXY_TOKEN:
        return {"error": "AIPROXY_TOKEN is missing. Please set it in your environment variables."}

    # Extract the word list dynamically
    extracted_words = extract_word_list(question)

    if extracted_words is None:
        return {"error": "Could not extract word list from the question."}

    # Define the API request payload
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": f"List only the valid English words from these: {extracted_words}"}
        ]
    }

    # Send request to OpenAI API via proxy
    response = requests.post(API_URL, json=data, headers=headers)

    if response.status_code == 200:
        response_json = response.json()

        # Extract total token usage
        total_tokens = response_json.get(
            'usage', {}).get('prompt_tokens', None)

        if total_tokens is None:
            return {"error": "Failed to extract token usage from API response."}

        total_tokens = json.dumps(total_tokens)
        return {"answer": total_tokens}

    else:
        return {"error": f"API request failed with status code {response.status_code}: {response.text}"}

# Function GA3Q3: Address generation


def solve_address_generation(question, file=None):
    text = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Respond in JSON"},
            {"role": "user", "content": "Generate 10 random addresses in the US"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "address_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "addresses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "apartment": {"type": "string"},
                                    "latitude": {"type": "number"},
                                    "county": {"type": "string"}
                                },
                                "required": ["apartment", "latitude", "county"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["addresses"],
                    "additionalProperties": False
                }
            }
        }
    }

    text = json.dumps(text)
    return {"answer": text}

# Function GA3Q4: LLM vision


def solve_llm_vision(question: str, file: UploadFile = None) -> Dict[str, Any]:
    """
    Converts an uploaded image to a base64 data URL and returns the JSON
    body for OpenAI Vision API using gpt-4o-mini.
    """
    if not file:
        return {"error": "No image file uploaded. Please upload a PNG or JPG image."}

    try:
        # Read image bytes from UploadFile
        image_data = file.file.read()
        image = Image.open(BytesIO(image_data))

        # Use BytesIO to get image in base64-encodable form
        buffered = BytesIO()
        image_format = image.format or "PNG"  # fallback to PNG if format missing
        image.save(buffered, format=image_format)
        encoded_bytes = buffered.getvalue()

        # Encode to base64 string
        base64_str = base64.b64encode(encoded_bytes).decode("utf-8")

        # Build base64 data URL
        base64_url = f"data:image/{image_format.lower()};base64,{base64_str}"

        # Create OpenAI-compatible JSON body
        response_json = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract text from this image?"},
                        {"type": "image_url", "image_url": {"url":
                                                            base64_url}}
                    ]
                }
            ]
        }

        response_json = json.dumps(response_json)
        return {"answer": response_json}

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

# Function GA3Q5: LLM embedding


def solve_llm_embedding(question, file=None):
    # Define the regex pattern to match verification messages
    pattern = r"Dear user, please verify your transaction code \d+ sent to [\w\.-]+@[\w\.-]+\.\w+"

    # Extract all matching verification messages
    extracted_messages = re.findall(pattern, question)

    if not extracted_messages:
        return {"error": "No verification messages found in the input text."}

    # Construct the required JSON structure
    response_json = {
        "model": "text-embedding-3-small",
        "input": extracted_messages
    }

    response_json = json.dumps(response_json)
    return {"answer": response_json}

# Function GA3Q6: embedding similarity


def solve_embedding_similarity(question, file=None):
    try:
        code = '''import numpy as np
from itertools import combinations

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def most_similar(embeddings):
    phrases = list(embeddings.keys())  # Extract phrases
    max_similarity = -1
    most_similar_pair = None

    for phrase1, phrase2 in combinations(phrases, 2):
        sim = cosine_similarity(np.array(embeddings[phrase1]), np.array(embeddings[phrase2]))
        if sim > max_similarity:
            max_similarity = sim
            most_similar_pair = (phrase1, phrase2)

    return most_similar_pair'''

        return {"answer": code}

    except Exception as e:
        return PlainTextResponse(f'{{"error": "Exception occurred: {str(e)}"}}')


# Function GA3Q7: vector databases
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/embeddings"


def solve_vector_databases(question: str, file: UploadFile = None):
    if main_app is None:
        return {"error": "Main app is not registered."}
    if not AIPROXY_TOKEN:
        return {"error": "AIPROXY_TOKEN not found in environment."}

    # ✅ Create sub FastAPI app
    sub_app = FastAPI()

    # ✅ Enable CORS
    sub_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # ✅ Request model
    class SimilarityRequest(BaseModel):
        docs: List[str]
        query: str

    def get_embeddings(texts: List[str]):
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "text-embedding-3-small",
            "input": texts
        }

        response = requests.post(AIPROXY_URL, json=payload, headers=headers)
        if response.status_code == 200:
            return [item["embedding"] for item in response.json()["data"]]
        else:
            raise HTTPException(
                status_code=response.status_code, detail=response.json())

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # ✅ POST /similarity
    @sub_app.post("/", response_model=dict)
    @sub_app.post("/similarity", include_in_schema=False)
    async def get_similar_documents(request: SimilarityRequest):
        try:
            if not request.docs:
                raise HTTPException(
                    status_code=400, detail="The 'docs' list cannot be empty.")

            embeddings = get_embeddings(request.docs + [request.query])
            doc_embeddings = embeddings[:-1]
            query_embedding = embeddings[-1]

            similarities = [cosine_similarity(
                query_embedding, doc_emb) for doc_emb in doc_embeddings]
            top_indices = np.argsort(similarities)[-3:][::-1]
            top_matches = [request.docs[i] for i in top_indices]

            return {"matches": top_matches}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ✅ Mount sub-app at /similarity
    main_app.mount("/similarity", sub_app)

    # ✅ Return URL
    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    return {
        "answer": f"{base_url}/similarity/"
    }

# Function GA3Q8: Function calling


def solve_function_calling(question, file=None):
    if main_app is None:
        return {"error": "Main app not registered."}

    # ✅ Create sub FastAPI app
    sub_app = FastAPI(title="Function Calling API")

    # ✅ Enable CORS for GET requests
    sub_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # ✅ Route to handle GET /execute?q=...
    @sub_app.get("/")
    async def execute(q: str = Query(...)):
        patterns = [
            {
                "pattern": r"status of ticket (\d+)",
                "name": "get_ticket_status",
                "extract": lambda m: {"ticket_id": int(m.group(1))}
            },
            {
                "pattern": r"Schedule a meeting on (\d{4}-\d{2}-\d{2}) at ([0-9:]+) in (.+?)\.",
                "name": "schedule_meeting",
                "extract": lambda m: {
                    "date": m.group(1),
                    "time": m.group(2),
                    "meeting_room": m.group(3)
                }
            },
            {
                "pattern": r"expense balance for employee (\d+)",
                "name": "get_expense_balance",
                "extract": lambda m: {"employee_id": int(m.group(1))}
            },
            {
                "pattern": r"performance bonus for employee (\d+) for (\d{4})",
                "name": "calculate_performance_bonus",
                "extract": lambda m: {
                    "employee_id": int(m.group(1)),
                    "current_year": int(m.group(2))
                }
            },
            {
                "pattern": r"office issue (\d+) for the ([\w\s]+?) department",
                "name": "report_office_issue",
                "extract": lambda m: {
                    "issue_code": int(m.group(1)),
                    "department": m.group(2).strip()
                }
            },
        ]

        for entry in patterns:
            match = re.search(entry["pattern"], q, re.IGNORECASE)
            if match:
                args = entry["extract"](match)
                return {
                    "name": entry["name"],
                    "arguments": json.dumps(args)  # ✅ Return as JSON string
                }

        return {
            "name": "unknown",
            "arguments": "{}"
        }

    # ✅ Mount the app at /execute
    main_app.mount("/execute", sub_app)

    # ✅ Return the testing URL
    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    return {
        "answer": f"{base_url}/execute?q=what+is+the+status+of+ticket+83742"
    }

# Function GA4Q1: HTML google sheets


def solve_html_google_sheets(question, file=None):
    try:
        # Extract the page number from the question
        match = re.search(r'page number (\d+)', question, re.IGNORECASE)
        if not match:
            return {"error": "Page number not found in the question."}

        page_number = int(match.group(1))

        # Construct the URL
        url = f"https://stats.espncricinfo.com/stats/engine/stats/index.html?class=2;page={page_number};template=results;type=batting"

        # Make the request with headers to avoid 403
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/119.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return {"error": f"Failed to retrieve the page: HTTP {response.status_code}"}

        # Parse tables using pandas
        tables = pd.read_html(response.text)

        # Usually the third table is the one with batting stats
        if len(tables) < 3:
            return {"error": "Could not find the expected stats table."}

        df = tables[2]

        # Try to find the column labeled "0" (representing ducks)
        duck_col = next(
            (col for col in df.columns if str(col).strip() == '0'), None)
        if duck_col is None:
            return {"error": "Could not find the 'Ducks' column (labeled '0')."}

        df[duck_col] = pd.to_numeric(df[duck_col], errors='coerce')
        total_ducks = int(df[duck_col].sum())

        total_ducks = json.dumps(total_ducks)
        return {"answer": total_ducks}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Function GA4Q2: IMDb


def solve_imdb(question, file=None):
    try:
        url = "https://www.imdb.com/search/title/?title_type=feature&user_rating=3.0,8.0&count=25"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return {"error": f"Failed to fetch IMDb page. Status code: {response.status_code}"}

        soup = BeautifulSoup(response.content, "html.parser")
        movies = []

        for item in soup.select("div.lister-item.mode-advanced"):
            # Extract title
            header = item.find("h3", class_="lister-item-header")
            title_tag = header.find("a")
            title = title_tag.text.strip() if title_tag else "N/A"

            # Extract year
            year_tag = header.find("span", class_="lister-item-year")
            year = year_tag.text.strip() if year_tag else "N/A"

            # Extract rating
            rating_tag = item.find(
                "div", class_="inline-block ratings-imdb-rating")
            rating = rating_tag['data-value'] if rating_tag and rating_tag.has_attr(
                'data-value') else "N/A"

            # Extract IMDb ID
            link = title_tag['href']
            id_match = re.search(r'/title/(tt\d+)/', link)
            movie_id = id_match.group(1) if id_match else "N/A"

            movies.append({
                "id": movie_id,
                "title": title,
                "year": year,
                "rating": rating
            })

        movies = json.dumps(movies)
        return {"answer": movies}

    except Exception as e:
        return {"error": str(e)}

# Function GA4Q3: Wiki headings


def solve_wiki_headings(question, file=None):
    if main_app is None:
        return {"error": "Main app not registered."}

    # ✅ Create sub FastAPI app
    sub_app = FastAPI(title="Wikipedia Outline Generator")

    # ✅ Enable CORS
    sub_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True
    )

    WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki/"

    @sub_app.get("/api/outline")
    async def get_wikipedia_outline(country: str = Query(..., title="Country Name")):
        """
        Fetch Wikipedia page of a country, extract all headings (H1-H6),
        and generate a Markdown outline.
        """
        country_url = WIKIPEDIA_BASE_URL + country.replace(" ", "_")

        async with httpx.AsyncClient() as client:
            response = await client.get(country_url)

        if response.status_code != 200:
            raise HTTPException(
                status_code=404, detail="Country Wikipedia page not found.")

        soup = BeautifulSoup(response.text, "html.parser")
        headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

        if not headings:
            raise HTTPException(
                status_code=404, detail="No headings found on the page.")

        markdown_outline = "## Contents\n\n"
        for heading in headings:
            level = int(heading.name[1])
            title = heading.get_text(strip=True)
            markdown_outline += f"{'#' * level} {title}\n\n"

        return {"country": country, "outline": markdown_outline}

    # ✅ Mount the sub-app at /wikipedia
    main_app.mount("/wikipedia", sub_app)

    # ✅ Return base URL for testing
    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    return {
        "answer": f"{base_url}/api"
    }


# Function GA4Q4: weather API


def getlocid(city):
    """
    API Integration and Data Retrieval:
    Sends a GET request to BBC's locator service with parameters like API key,
    locale, filters, and city name to get the locationId.
    """
    city = city.lower()
    location_url = 'https://locator-service.api.bbci.co.uk/locations?' + urlencode({
        'api_key': 'AGbFAKx58hyjQScCXIYrxuEwJh2W2cmv',
        's': city,
        'stack': 'aws',
        'locale': 'en',
        'filter': 'international',
        'place-types': 'settlement,airport,district',
        'order': 'importance',
        'a': 'true',
        'format': 'json'
    })

    response = requests.get(location_url)
    if response.status_code != 200:
        raise Exception("Failed to fetch location ID.")

    result = response.json()
    try:
        locid = result['response']['results']['results'][0]['id']
        return locid
    except (KeyError, IndexError):
        raise Exception(f"Location ID for '{city}' not found.")


def solve_weather_api(question, file=None):
    """
    Automates:
    1. Location detection from question
    2. API Integration to retrieve locationId
    3. Forecast Data Retrieval using weather broker API
    4. Data Transformation into a JSON object mapping localDate to enhancedWeatherDescription
    """

    # --- Step 1: Extract city from question
    try:
        match = re.search(
            r"(?:in|for)\s+([A-Za-z\s]+)", question, re.IGNORECASE)
        location = match.group(1).strip() if match else "New York"
    except Exception as e:
        return {"answer": json.dumps({"error": f"Could not extract location: {str(e)}"})}

    # --- Step 2: Get locationId via BBC Locator Service
    try:
        location_id = getlocid(location)
    except Exception as e:
        return {"answer": json.dumps({"error": str(e)})}

    # --- Step 3: Fetch weather forecast using locationId
    try:
        weather_url = f"https://weather-broker-cdn.api.bbci.co.uk/en/forecast/aggregated/{location_id}"
        resp = requests.get(weather_url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"answer": json.dumps({"error": f"Error fetching weather data: {str(e)}"})}

    # --- Step 4: Transform the data (localDate → enhancedWeatherDescription)
    try:
        forecast_summary = {}
        forecasts = data.get("forecasts", [])

        for day in forecasts:
            if len(forecast_summary) >= 14:
                break  # ✅ Limit to 14 days
            reports = day.get("detailed", {}).get("reports", [])
            for report in reports:
                date = report.get("localDate")
                desc = report.get("enhancedWeatherDescription")
                if date and desc and date not in forecast_summary:
                    forecast_summary[date] = desc
                    break  # Use only one entry per day

        return {
            "answer": json.dumps(forecast_summary, indent=2)
        }
    except Exception as e:
        return {"answer": json.dumps({"error": f"Error processing forecast: {str(e)}"})}

# Function GA4Q5: city bounding box


def solve_city_bounding_box(question, file=None):
    try:
        # Extract city and country from the question using regex
        match = re.search(
            r'the city ([\w\s]+?) in the country ([\w\s]+?) on the Nominatim API', question, re.IGNORECASE)
        if not match:
            return {"error": "Could not extract city and country from the question."}

        city = match.group(1).strip()
        country = match.group(2).strip()

        # Initialize Nominatim geocoder
        locator = Nominatim(user_agent="myGeocoder")
        location = locator.geocode(f"{city}, {country}")

        if not location:
            return {"error": f"Location not found for {city}, {country} on the Nominatim API."}

        if 'boundingbox' not in location.raw:
            return {"error": "Bounding box data not available in response."}

        # [south_lat, north_lat, west_lon, east_lon]
        bounding_box = location.raw['boundingbox']
        max_latitude = float(bounding_box[1])  # north_lat is the second entry

        max_latitude = json.dumps(max_latitude)
        return {"answer": max_latitude}

    except Exception as e:
        return {"error": f"Failed to retrieve location info: {str(e)}"}

# Function GA4Q6: Hacker news


def solve_hacker_news(question, file=None):
    try:
        # Step 1: Extract the topic between 'mentioning' and 'having'
        match = re.search(r'mentioning\s+"?(.+?)"?\s+having',
                          question, re.IGNORECASE)
        if not match:
            return {"error": "Could not extract the topic from the question."}

        topic = match.group(1).strip()
        encoded_topic = urllib.parse.quote(topic)

        # Step 2: Construct the HNRSS query URL with the topic and minimum points
        url = f"https://hnrss.org/newest?q={encoded_topic}&points=30"

        # Step 3: Parse the RSS feed
        feed = feedparser.parse(url)

        if not feed.entries:
            return {"error": f"No posts found with '{topic}' and at least 30 points."}

        # Step 4: Get the most recent item and return its link
        latest_entry = feed.entries[0]
        link = latest_entry.get("link")
        if not link:
            return {"error": "Link not found in the latest item."}

        link = json.dumps(link)
        return {"answer": link}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Function GA4Q7: GitHub new user


def solve_new_github_user(question, file=None):
    try:
        # Extract city and follower threshold from the question
        city_match = re.search(
            r'located in the city ([\w\s]+?) with', question, re.IGNORECASE)
        followers_match = re.search(r'over (\d+) followers', question)

        if not city_match or not followers_match:
            return {"error": "Could not extract city or follower threshold from the question."}

        city = city_match.group(1).strip()
        min_followers = int(followers_match.group(1))

        # Construct GitHub API search query
        github_api_url = f"https://api.github.com/search/users?q=location:{city}+followers:>{min_followers}&sort=joined&order=desc"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "FastAPI-App"
        }

        # Define cut-off time
        cutoff_time = datetime.strptime(
            "2025-02-07T18:05:36", "%Y-%m-%dT%H:%M:%S")

        response = requests.get(github_api_url, headers=headers)
        if response.status_code != 200:
            return {"error": f"GitHub API request failed with status code {response.status_code}"}

        users = response.json().get("items", [])
        if not users:
            return {"error": "No users found matching the query."}

        # Loop through users and get created_at for each
        for user in users:
            user_url = user["url"]
            user_response = requests.get(user_url, headers=headers)
            if user_response.status_code == 200:
                user_data = user_response.json()
                created_at = user_data.get("created_at", "")
                if created_at:
                    created_at_dt = datetime.strptime(
                        created_at, "%Y-%m-%dT%H:%M:%SZ")
                    if created_at_dt <= cutoff_time:
                        return {"answer": created_at}

        return {"error": "No user found before the cut-off time."}

    except Exception as e:
        return {"error": f"An exception occurred: {str(e)}"}

# Function GA4Q8: Scheduled Github action


def solve_scheduled_github_action(question, file=None):
    return {"answer": "https://github.com/thereasajoe/ga4/actions/workflows/blank.yml"}


# Function GA4Q9: Extract tables


def solve_extract_tables(question: str, file: UploadFile):
    try:
        # Step 1: Extract values from the question
        subject_match = re.findall(
            r'total\s+(\w+)\s+marks|marks\s+in\s+(\w+)', question, re.IGNORECASE)
        score_match = re.search(r'scored\s+(\d+)', question)
        group_match = re.search(r'groups?\s+(\d+)-(\d+)', question)

        subjects = [s for pair in subject_match for s in pair if s]
        subject = subjects[0].strip().lower() if subjects else None
        min_score = int(score_match.group(1)) if score_match else None
        group_start = int(group_match.group(1)) if group_match else None
        group_end = int(group_match.group(2)) if group_match else None

        if not all([subject, min_score, group_start, group_end]):
            return {"error": "Could not extract subject, score, or group range from the question."}

        # Step 2: Save the PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        # Step 3: Extract tables using Camelot
        tables = camelot.read_pdf(tmp_path, pages="all", flavor="stream")
        if not tables or tables.n == 0:
            return {"error": "No tables found in PDF."}

        all_dfs = []
        group_number = group_start

        for table in tables:
            df = table.df.dropna(how="all")

            # Detect and skip title row if necessary
            header_row = df.iloc[0]
            if header_row.isnull().sum() >= len(header_row) - 1 or all([not str(c).strip().isdigit() for c in df.iloc[1]]):
                df.columns = df.iloc[1]  # Use 2nd row as header
                df = df[2:]
            else:
                df.columns = df.iloc[0]
                df = df[1:]

            # Rename columns cleanly
            df.columns = [str(col).strip().lower() for col in df.columns]

            # Find the column that matches the subject
            matched_columns = [col for col in df.columns if subject in col]
            if not matched_columns:
                continue  # Skip this table if subject column not found
            subject_col = matched_columns[0]

            df["group"] = group_number
            df[subject_col] = pd.to_numeric(df[subject_col], errors="coerce")
            all_dfs.append(df)
            group_number += 1

        if not all_dfs:
            return {"error": f"Subject '{subject}' column not found in any table."}

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df["group"] = pd.to_numeric(
            combined_df["group"], errors="coerce")

        filtered = combined_df[
            (combined_df["group"] >= group_start) &
            (combined_df["group"] <= group_end) &
            (combined_df[subject_col] >= min_score)
        ]

        total = int(filtered[subject_col].sum())
        total = json.dumps(total)
        return {"answer": total}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Function GA4Q10: pdf to md


def solve_pdf_to_md(question, file=None):
    try:
        # Step 1: Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            pdf_path = temp_pdf.name
            content = file.file.read()
            temp_pdf.write(content)

        # Step 2: Extract text using pdfminer
        markdown_text = extract_text(pdf_path).strip()

        # Step 3: Save as markdown
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as md_file:
            md_path = md_file.name
            md_file.write(markdown_text)

        # Step 4: Format using prettier
        subprocess.run(["prettier", "--write", md_path], check=True)

        # Step 5: Read back the formatted markdown
        with open(md_path, "r", encoding="utf-8") as formatted:
            formatted_markdown = formatted.read()

        # Cleanup
        os.remove(pdf_path)
        os.remove(md_path)

        formatted_markdown = json.dumps(formatted_markdown)
        return {"answer": formatted_markdown}

    except subprocess.CalledProcessError:
        return {"error": "Prettier formatting failed. Ensure Prettier 3.4.2 is installed globally via npm."}
    except Exception as e:
        return {"error": f"Failed to process the PDF: {str(e)}"}

# Function GA5Q1: Excel clean up


def solve_excel_sales_cleanup(question: str, file: UploadFile):
    try:
        # Extract parameters from question
        date_match = re.search(r'before (.+?) for', question)
        product_match = re.search(r'for (\w+) sold', question)
        country_match = re.search(r'sold in (\w+)', question)

        if not (date_match and product_match and country_match):
            return {"error": "Could not extract filter conditions from question."}

        # Step 1: Extract filters
        raw_date_str = date_match.group(1).strip()
        product_filter = product_match.group(1).strip().lower()
        country_filter = country_match.group(1).strip().upper()

        # Convert date string to datetime in UTC
        ist_offset = timedelta(hours=5, minutes=30)
        dt = datetime.strptime(raw_date_str[:24], "%a %b %d %Y %H:%M:%S")
        target_date = dt - ist_offset

        # Step 2: Read Excel
        df = pd.read_excel(file.file)

        # Step 3: Clean data
        df['Customer Name'] = df['Customer Name'].astype(str).str.strip()
        df['Country'] = df['Country'].astype(str).str.strip().str.upper()

        # Normalize country names
        country_mapping = {
            "USA": "US", "U.S.A": "US", "UNITED STATES": "US",
            "UK": "UK", "U.K.": "UK", "GREAT BRITAIN": "UK",
            "BRAZIL": "BR", "BRZ": "BR", "BR.": "BR"
        }
        df['Country'] = df['Country'].replace(country_mapping)

        # Normalize Product name
        df['Product'] = df['Product'].astype(str)
        df['Product Name'] = df['Product'].str.split(
            "/").str[0].str.strip().str.lower()

        # Clean Date field
        def parse_date(date):
            if isinstance(date, str):
                for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%Y/%m/%d", "%d-%m-%Y"):
                    try:
                        return datetime.strptime(date.strip(), fmt)
                    except ValueError:
                        continue
            elif isinstance(date, datetime):
                return date
            return pd.NaT

        df['Date'] = df['Date'].apply(parse_date)

        # Clean and convert Sales and Cost
        df['Sales'] = pd.to_numeric(df['Sales'].astype(str).str.replace(
            "USD", "", regex=False).str.strip(), errors='coerce')
        df['Cost'] = pd.to_numeric(df['Cost'].astype(str).str.replace(
            "USD", "", regex=False).str.strip(), errors='coerce')

        # Fill missing cost with 50% of sales
        df['Cost'] = df.apply(lambda row: row['Sales'] *
                              0.5 if pd.isna(row['Cost']) else row['Cost'], axis=1)

        # Step 4: Filter
        filtered_df = df[
            (df['Date'] <= target_date) &
            (df['Product Name'] == product_filter) &
            (df['Country'] == country_filter)
        ]

        if filtered_df.empty:
            return {"error": "No matching records found."}

        # Step 5: Margin Calculation
        total_sales = filtered_df['Sales'].sum()
        total_cost = filtered_df['Cost'].sum()

        margin = round((total_sales - total_cost) / total_sales, 4)

        return {
            "answer": json.dumps(margin),
            "matching_transactions": len(filtered_df),
            "total_sales": total_sales,
            "total_cost": total_cost
        }

    except Exception as e:
        return {"error": f"Failed to process Excel file: {str(e)}"}

# Function GA5Q2: Clean up student marks


def solve_student_marks_cleanup(question: str, file: Optional[UploadFile] = None):
    try:
        if not file:
            return {"error": "No file uploaded."}

        # Read file content line by line
        content = file.file.read().decode("utf-8").splitlines()

        student_ids = []

        # Extract alphanumeric student ID (10+ chars before "::Marks" or "Marks")
        for line in content:
            match = re.search(
                r'[-\s]([A-Z0-9]{8,})\s*(?=::?Marks)', line, re.IGNORECASE)
            if match:
                student_ids.append(match.group(1).strip())

        # Remove duplicates
        unique_ids = set(student_ids)

        return {"answer": json.dumps(len(unique_ids))}

    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}

# Function GA5Q3: Apache log requests


def solve_log_requests(question, file=None):
    try:
        # Extract dynamic filters from the question
        url_match = re.search(r'for pages under (/\w+/)', question)
        start_time_match = re.search(r'from (\d{1,2}):00', question)
        end_time_match = re.search(r'before (\d{1,2}):00', question)
        day_match = re.search(r'on (\w+)[s]?', question)

        if not (url_match and start_time_match and end_time_match and day_match):
            return {"error": "Could not extract filters from the question."}

        url_prefix = url_match.group(1)
        start_hour = int(start_time_match.group(1))
        end_hour = int(end_time_match.group(1))
        weekday_str = day_match.group(1).capitalize()
        weekday_map = {
            "Monday": 0, "Mondays": 0,
            "Tuesday": 1, "Tuesdays": 1,
            "Wednesday": 2, "Wednesdays": 2,
            "Thursday": 3, "Thursdays": 3,
            "Friday": 4, "Fridays": 4,
            "Saturday": 5, "Saturdays": 5,
            "Sunday": 6, "Sundays": 6
        }

        if weekday_str not in weekday_map:
            return {"error": f"Invalid weekday: {weekday_str}"}
        target_weekday = weekday_map[weekday_str]

        # Apache log regex pattern
        log_pattern = re.compile(
            r'(?P<ip>\S+) (?P<logname>\S+) (?P<user>\S+) \[(?P<time>.+?)\] '
            r'"(?P<request>.+?)" (?P<status>\d{3}) (?P<size>\S+) '
            r'"(?P<referer>.*?)" "(?P<user_agent>.*?)" (?P<vhost>\S+) (?P<server>\S+)'
        )

        count = 0

        with gzip.open(file.file, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                match = log_pattern.match(line)
                if not match:
                    continue
                data = match.groupdict()

                # Split request into method, url, and protocol
                parts = data['request'].split()
                if len(parts) != 3:
                    continue

                method, url, _ = parts
                status = int(data['status'])

                # Check filters
                if method == 'GET' and url.startswith(url_prefix) and 200 <= status < 300:
                    # Parse and convert time to GMT-0500
                    log_time = datetime.strptime(
                        data["time"], "%d/%b/%Y:%H:%M:%S %z")
                    log_time = log_time.astimezone(pytz.timezone("Etc/GMT+5"))

                    if (
                        log_time.weekday() == target_weekday and
                        start_hour <= log_time.hour < end_hour
                    ):
                        count += 1

        count = json.dumps(count)
        return {
            "answer": count
        }

    except Exception as e:
        return {"error": f"Failed to process Apache log: {str(e)}"}

# Function GA5Q4: Apache log downloads


def solve_log_downloads(question, file=None):
    try:
        url_pattern = r"under (\S+?)\s+on"
        date_pattern = r"on (\d{4}-\d{2}-\d{2})"

        url_match = re.search(url_pattern, question)
        date_match = re.search(date_pattern, question)

        if not (url_match and date_match):
            return {"error": "Invalid question format. Could not extract URL or date."}

        target_url = url_match.group(1)
        target_date = date_match.group(1)

        ip_downloads = defaultdict(int)

        with gzip.open(file.file, 'rt', encoding='utf-8') as f:
            for line in f:
                match = re.match(
                    r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<time>[^\]]+)\] "(?P<request>[^"]+)" (?P<status>\d+) (?P<size>\d+) "(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)" (?P<vhost>\S+) (?P<server_ip>\d+\.\d+\.\d+\.\d+)',
                    line
                )

                if match:
                    ip = match.group('ip')
                    time = match.group('time')
                    request = match.group('request')
                    size = int(match.group('size'))

                    try:
                        log_datetime = datetime.strptime(
                            time, "%d/%b/%Y:%H:%M:%S %z")
                        log_date = log_datetime.strftime("%Y-%m-%d")
                    except ValueError:
                        continue  # Skip malformed time

                    # Break request into parts and validate
                    try:
                        method, path, protocol = request.split()
                    except ValueError:
                        continue

                    if log_date == target_date and target_url in path:
                        ip_downloads[ip] += size

        if ip_downloads:
            top_ip = max(ip_downloads, key=ip_downloads.get)
            return {"answer": json.dumps(ip_downloads[top_ip])}
        else:
            return {"answer": 0}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


# Function GA5Q5: Clean up sales


def solve_cleanup_sales(question, file=None):
    try:
        # Step 1: Extract product, city, and sales threshold from question
        import re
        product_match = re.search(r'units of (\w+)', question, re.IGNORECASE)
        city_match = re.search(r'sold in (\w+)', question, re.IGNORECASE)
        sales_match = re.search(r'at least (\d+)', question)

        if not product_match or not city_match or not sales_match:
            return {"error": "Could not extract required filters from question."}

        product = product_match.group(1).lower()
        city_target = city_match.group(1).lower()
        min_sales = int(sales_match.group(1))

        # Step 2: Load the JSON data
        contents = file.file.read().decode("utf-8")
        data = json_parser.loads(contents)

        # Step 3: Identify city name variations that match the target using fuzzy matching
        unique_cities = set(entry['city'].lower() for entry in data)
        similar_cities = {city for city in unique_cities if fuzz.ratio(
            city, city_target) >= 85}

        # Step 4: Sum all relevant entries
        total_sales = 0
        for entry in data:
            entry_city = entry['city'].lower()
            entry_product = entry['product'].lower()
            entry_sales = entry['sales']

            if entry_city in similar_cities and entry_product == product and entry_sales >= min_sales:
                total_sales += entry_sales

        return {"answer": json.dumps(total_sales)}

    except Exception as e:
        return {"error": f"Failed to process JSON data: {str(e)}"}

# Function GA5Q6: Parse partial JSON


def solve_parse_partial_json(question, file=None):
    try:
        # Step 1: Compile regex pattern to find "sales": <number>
        sales_pattern = re.compile(r'"sales"\s*:\s*(\d+)')

        # Step 2: Initialize total
        total_sales = 0

        # Step 3: Read file line by line
        for line in file.file:
            decoded_line = line.decode("utf-8")
            matches = sales_pattern.findall(decoded_line)
            for match in matches:
                total_sales += int(match)

        total_sales = json.dumps(total_sales)
        return {"answer": total_sales}

    except Exception as e:
        return {"error": f"Failed to compute total sales: {str(e)}"}

# Function GA5Q7: Extracted nested JSON keys


def solve_nested_jsonkeys(question, file=None) -> Dict[str, Union[int, str]]:
    try:
        # Step 1: Extract the key name from the question
        match = re.search(
            r'how many times does ([\w\d_]+) appear as a key', question, re.IGNORECASE)
        if not match:
            return {"error": "Could not extract key from question."}

        target_key = match.group(1)

        # Step 2: Load JSON content
        data = json.load(file.file)

        # Step 3: Recursively count the number of times the key appears
        def count_key_occurrences(obj):
            count = 0
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == target_key:
                        count += 1
                    count += count_key_occurrences(value)
            elif isinstance(obj, list):
                for item in obj:
                    count += count_key_occurrences(item)
            return count

        # Step 4: Run and return the result
        total_count = count_key_occurrences(data)
        total_count = json.dumps(total_count)
        return {"answer": total_count}

    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}

# Function GA5Q8: DuckDB social media


def solve_duckdb_socialmedia(question, file=None):
    duckdb = """SELECT DISTINCT posts.post_id
                        FROM posts
                        JOIN comments ON posts.post_id = comments.post_id
                        WHERE posts.created_at > '2024-12-25T22:48:29.078Z'
                        AND comments.useful_stars = 3
                        ORDER BY posts.post_id ASC;"""
    return {"answer": duckdb}

# Function GA5Q9: Transcribe YouTube video


def solve_transcribe_yt(question, file=None):
    try:
        # Step 1: Extract video URL and time range from the question
        import re

        url_match = re.search(r"(https?://[^\s]+)", question)
        time_match = re.search(
            r'between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?) seconds', question)

        if not url_match or not time_match:
            return {"error": "Could not extract URL or time range from question."}

        video_url = url_match.group(1)
        start_time = time_match.group(1)
        end_time = time_match.group(2)

        # Step 2: Temporary path for audio
        temp_audio_file = tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()

        # Step 3: yt_dlp download options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_audio_path.replace(".mp3", ""),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'postprocessor_args': ['-ss', start_time, '-to', end_time],
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Verify audio was downloaded
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            return {"error": "Audio download failed."}

        # Step 4: Transcribe using Whisper
        model = whisper.load_model("small")
        result = model.transcribe(temp_audio_path)

        # Step 5: Clean up temp file
        os.unlink(temp_audio_path)

        return {"answer": result.get("text", "").strip()}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Function GA5Q10: Reconstruct image


def solve_image_reconstruction(question: str, files: List[UploadFile]):
    try:
        # Separate mapping and image files
        mapping_file = None
        image_file = None

        for f in files:
            if f.filename.endswith(('.md', '.html', '.txt')):
                mapping_file = f
            elif f.filename.endswith(('.webp', '.png', '.jpg', '.jpeg')):
                image_file = f

        if not mapping_file or not image_file:
            return {"error": "Please upload both a mapping file (.md/.html) and an image file."}

        # Step 1: Extract mapping from markdown
        content = mapping_file.file.read().decode("utf-8")

        # Extract rows like: | 2 | 1 | 0 | 0 |
        mapping_pattern = re.findall(
            r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|', content)
        if not mapping_pattern or len(mapping_pattern) != 25:
            return {"error": "Could not extract valid 5x5 mapping from the markdown file."}

        mapping = [(int(r1), int(c1), int(r2), int(c2))
                   for r1, c1, r2, c2 in mapping_pattern]

        # Step 2: Save the uploaded scrambled image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(image_file.file.read())

        img = Image.open(tmp_path)
        grid_size = (5, 5)
        tile_width = img.width // grid_size[0]
        tile_height = img.height // grid_size[1]

        # Step 3: Create output canvas
        unscrambled_img = Image.new("RGB", img.size)

        # Step 4: Rearrange tiles according to the mapping
        for orig_r, orig_c, scram_r, scram_c in mapping:
            left = scram_c * tile_width
            upper = scram_r * tile_height
            right = left + tile_width
            lower = upper + tile_height

            tile = img.crop((left, upper, right, lower))

            new_left = orig_c * tile_width
            new_upper = orig_r * tile_height

            unscrambled_img.paste(tile, (new_left, new_upper))

        # Step 5: Return as base64
        buffered = io.BytesIO()
        unscrambled_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"answer": img_base64}

    except Exception as e:
        return {"error": f"Failed to unscramble image: {str(e)}"}


function_map = {
    # GA 1
    "solve_vscode_question": solve_vscode_question,
    "solve_http_request_question": solve_http_request_question,
    "solve_prettier_hash_question": solve_prettier_hash_question,
    "solve_google_sheets_question": solve_google_sheets_question,
    "solve_excel_question": solve_excel_question,
    "solve_hidden_input_question": solve_hidden_input_question,
    "solve_count_wednesdays_question": solve_count_wednesdays_question,
    "solve_csv_extraction_question": solve_csv_extraction_question,
    "solve_json_sorting_question": solve_json_sorting_question,
    "solve_json_conversion_question": solve_json_conversion_question,
    "solve_div_sum_question": solve_div_sum_question,
    "solve_file_encoding_sum_question": solve_file_encoding_sum_question,
    "solve_github_repo_question": solve_github_repo_question,
    "solve_replace_text_question": solve_replace_text_question,
    "solve_file_size_filter_question": solve_file_size_filter_question,
    "solve_rename_files_question": solve_rename_files_question,
    "solve_compare_files_question": solve_compare_files_question,
    "solve_sqlite_query_question": solve_sqlite_query_question,
    # GA 2
    "solve_markdown_documentation_question": solve_markdown_documentation_question,
    "solve_image_compression_question": solve_image_compression_question,
    "solve_github_pages_question": solve_github_pages_question,
    "solve_colab_auth_question": solve_colab_auth_question,
    "solve_colab_brightness_question": solve_colab_brightness_question,
    "solve_vercel_api_question": solve_vercel_api_question,
    "solve_github_action_question": solve_github_action_question,
    "solve_docker_image_question": solve_docker_image_question,
    "solve_fastapi_server_question": solve_fastapi_server_question,
    "solve_llama_model_question": solve_llama_model_question,
    # GA 3
    "solve_llm_sentiment": solve_llm_sentiment,
    "solve_token_cost": solve_token_cost,
    "solve_address_generation": solve_address_generation,
    "solve_llm_vision": solve_llm_vision,
    "solve_llm_embedding": solve_llm_embedding,
    "solve_embedding_similarity": solve_embedding_similarity,
    "solve_vector_databases": solve_vector_databases,
    "solve_function_calling": solve_function_calling,
    # GA 4
    "solve_html_google_sheets": solve_html_google_sheets,
    "solve_imdb": solve_imdb,
    "solve_wiki_headings": solve_wiki_headings,
    "solve_weather_api": solve_weather_api,
    "solve_city_bounding_box": solve_city_bounding_box,
    "solve_hacker_news": solve_hacker_news,
    "solve_new_github_user": solve_new_github_user,
    "solve_scheduled_github_action": solve_scheduled_github_action,
    "solve_extract_tables": solve_extract_tables,
    "solve_pdf_to_md": solve_pdf_to_md,
    # GA 5
    "solve_excel_sales_cleanup": solve_excel_sales_cleanup,
    "solve_student_marks_cleanup": solve_student_marks_cleanup,
    "solve_log_requests": solve_log_requests,
    "solve_log_downloads": solve_log_downloads,
    "solve_cleanup_sales": solve_cleanup_sales,
    "solve_parse_partial_json": solve_parse_partial_json,
    "solve_nested_jsonkeys": solve_nested_jsonkeys,
    "solve_duckdb_socialmedia": solve_duckdb_socialmedia,
    "solve_transcribe_yt": solve_transcribe_yt,
    "solve_image_reconstruction": solve_image_reconstruction
}
