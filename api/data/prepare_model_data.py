# prepare_model_data.py

import pickle
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# --- STEP 1: Define stored questions and corresponding function names

stored_questions = [
    # GA 1
    "Install and run Visual Studio Code. In your Terminal (or Command Prompt), type code -s and press Enter. Copy and paste the entire output below. What is the output of code -s?",

    "Running uv run --with httpie -- https [URL] installs the Python package httpie and sends a HTTPS request to the URL. Send a HTTPS request to https://httpbin.org/get with the URL encoded parameter email set to 24ds2000125@ds.study.iitm.ac.in. What is the JSON output of the command? (Paste only the JSON body, not the headers)",

    "Download Readme.md In the directory where you downloaded it, make sure it is called README.md, and run npx -y prettier@3.4.2 README.md | sha256sum. What is the output of the command?",

    "Type this formula into Google Sheets. (It won't work in Excel) =SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 9, 15), 1, 10)). What is the result?",

    "This will ONLY work in Office 365. =SUM(TAKE(SORTBY({1,11,2,15,14,12,4,15,14,3,2,2,9,7,1,7}, {10,9,13,2,11,8,16,14,7,15,5,4,6,1,3,12}), 1, 6)). What is the result?",

    "Just above this paragraph, there's a hidden input with a secret value. What is the value in the hidden input?",

    "How many Wednesdays are there in the date range 1988-06-29 to 2008-08-05?",

    "Download and unzip file file_name.zip which has a single extract.csv file inside. What is the value in the 'answer' column of the CSV file?",

    "Sort this JSON array of objects by the value of the age field. In case of a tie, sort by the name field. Paste the resulting JSON below without any spaces or newlines. [{\"name\":\"Alice\",\"age\":85},{\"name\":\"Bob\",\"age\":13},{\"name\":\"Charlie\",\"age\":8},{\"name\":\"David\",\"age\":33},{\"name\":\"Emma\",\"age\":60},{\"name\":\"Frank\",\"age\":43},{\"name\":\"Grace\",\"age\":26},{\"name\":\"Henry\",\"age\":15},{\"name\":\"Ivy\",\"age\":79},{\"name\":\"Jack\",\"age\":86},{\"name\":\"Karen\",\"age\":5},{\"name\":\"Liam\",\"age\":66},{\"name\":\"Mary\",\"age\":64},{\"name\":\"Nora\",\"age\":3},{\"name\":\"Oscar\",\"age\":95},{\"name\":\"Paul\",\"age\":91}]. What is the sorted JSON?",

    "Download file_name.txt and use multi-cursors and convert it into a single JSON object, where key=value pairs are converted into {key: value, key: value, ...}. What's the result when you paste the JSON at tools-in-data-science.pages.dev/jsonhash and click the Hash button?",

    "Find all <div>s having a foo class in the hidden element below. What's the sum of their data-value attributes? Sum of data-value attributes:",

    "Download and process the files in file_name.zip which contains three files with different encodings: data1.csv: CSV file encoded in CP-1252 data2.csv: CSV file encoded in UTF-8 data3.txt: Tab-separated file encoded in UTF-16. Each file has 2 columns: symbol and value. Sum up all the values where the symbol matches š OR œ OR Ÿ across all three files. What is the sum of all values associated with these symbols?",

    "Create a GitHub account if you don't have one. Create a new public repository. Commit a single JSON file called email.json with the value {\"email\": \"24ds2000125@ds.study.iitm.ac.in\"} and push it. Enter the raw Github URL of email.json so we can verify it. (It might look like https://raw.githubusercontent.com/[GITHUB ID]/[REPO NAME]/main/email.json.)",

    "Download q-replace-across-files.zip and unzip it into a new folder, then replace all 'IITM' (in upper, lower, or mixed case) with 'IIT Madras' in all files. Leave everything as-is - don't change the line endings. What does running cat * | sha256sum in that folder show in bash?",

    "Download file_name.zip and extract it. Use ls with options to list all files in the folder along with their date and file size. What's the total size of all files at least 1978 bytes large and modified on or after Sun, 24 Jul, 2011, 10:43 pm IST?",

    "Download file_name.zip and extract it. Use mv to move all files under folders into an empty folder. Then rename all files replacing each digit with the next. 1 becomes 2, 9 becomes 0, a1b9c.txt becomes a2b0c.txt. What does running grep . * | LC_ALL=C sort | sha256sum in bash on that folder show?",
    
    "Download file_name.zip and extract it. It has 2 nearly identical files, a.txt and b.txt, with the same number of lines. How many lines are different between a.txt and b.txt?",
    
    "There is a tickets table in a SQLite database that has columns type, units, and price. Each row is a customer bid for a concert ticket.\n\ntype        units        price\nbronze        278        1.08\nsilver        563        1.85\nBRONZE        855        1.86\ngold        429        1.71\nSILVER        510        0.85\n...\nWhat is the total sales of all the items in the \"Gold\" ticket type? Write SQL to calculate it.",
    
    # GA 2
    "Write documentation in Markdown for an **imaginary** analysis of the number of steps you walked each day for a week, comparing over time and with friends. The Markdown must include:\n\n- Top-Level Heading\n- Subheadings\n- Bold Text\n- Italic Text\n- Inline Code\n- Code Block\n- Bulleted List\n- Numbered List\n- Table\n- Hyperlink\n- Image\n- Blockquote",
    
    "Download the image below and compress it losslessly to an image that is less than 1,500 bytes. By losslessly, we mean that every pixel in the new image should be identical to the original image. Upload your losslessly compressed image (less than 1,500 bytes).",

    "Publish a page using GitHub Pages that showcases your work. Ensure that your email address 24ds2000125@ds.study.iitm.ac.in is in the page's HTML.\n\nGitHub pages are served via CloudFlare which obfuscates emails. So, wrap your email address inside a:\n\n<!--email_off-->24ds2000125@ds.study.iitm.ac.in<!--/email_off-->\nWhat is the GitHub Pages URL? It might look like: https://[USER].github.io/[REPO]/\nIf a recent change that's not reflected, add ?v=1, ?v=2 to the URL to bust the cache.",
    
    "Run this program on Google Colab, allowing all required access to your email ID: 24ds2000125@ds.study.iitm.ac.in. What is the result? (It should be a 5-character string)", 
    
    "Download this image. Create a new Google Colab notebook and run this code (after fixing a mistake in it) to calculate the number of pixels with a certain minimum brightness, What is the result? (It should be a number)",

    "Download this file which has the marks of 100 imaginary students.\n\nCreate and deploy a Python app to Vercel. Expose an API so that when a request like https://your-app.vercel.app/api?name=X&name=Y is made, it returns a JSON response with the marks of the names X and Y in the same order, like this:\n\n{ \"marks\": [10, 20] }\nMake sure you enable CORS to allow GET requests from any origin.\n\nWhat is the Vercel URL? It should look like: https://your-app.vercel.app/api",
    
    "Create a GitHub action on one of your GitHub repositories. Make sure one of the steps in the action has a name that contains your email address 24ds2000125@ds.study.iitm.ac.in. For example:\n\n\njobs:\n  test:\n    steps:\n      - name: 24ds2000125@ds.study.iitm.ac.in\n        run: echo \"Hello, world!\"\n      \nTrigger the action and make sure it is the most recent action.\n\nWhat is your repository URL? It will look like: https://github.com/USER/REPO",

    "Create and push an image to Docker Hub. Add a tag named 24ds2000125 to the image. What is the Docker image URL? It should look like: https://hub.docker.com/repository/docker/$USER/$REPO/general",

    "Download . This file has 2-columns: studentId: A unique identifier for each student, e.g. 1, 2, 3, ... class: The class (including section) of the student, e.g. 1A, 1B, ... 12A, 12B, ... 12Z Write a FastAPI server that serves this data. What is the API URL endpoint for FastAPI? It might look like: http://127.0.0.1:8000/api",

    "Download Llamafile. Run the Llama-3.2-1B-Instruct.Q6_K.llamafile model with it. Create a tunnel to the Llamafile server using ngrok. What is the ngrok URL? It might look like: https://[random].ngrok-free.app/",

    #GA 3
    "Write a Python program that uses httpx to send a POST request to OpenAI's API to analyze the sentiment of this (meaningless) text into GOOD, BAD or NEUTRAL.",

    "when you make a request to OpenAI's GPT-4o-Mini with just this user message: List only the valid English words from these: k3ADrvzier, VA7p8, AFnT, BKp7, 7CzrQI8L3, cg2, qHXVa4, 1TMgV, 7Q, 7z, 23ZKgSKuE1, n, D, q0S, 9F6Ht, 1P4s, ieK, 2laK9miOr, 9yvQ3, AL0iIKk5UR, VcAMGZkC, 14qRSZ0Jlm, Qt, TmCSbnaOi, GHvIz34qp, S, nfyU8, UD9qc, hv, ZDu0Anl, e, Y, PU, aF, t0W, fCmSl1PObS, EXk, VHcfIyUv, efD1bujZB9, pdAPN6IzNA, W06Xim0Kj, KDaBjaAd, lBORNzf, IzjxPpr, JV, A, uQBWWzi, PrTQi, m, b6zON, w6CDI, Um7Wt, Ues2RDsrO, rA, Ef, Fs, J2Nsqso, bdydMBsIm, 9C5ZO187, G3JHvk, AtHM, rKBKaxl7... how many input tokens does it use up? Number of tokens:",

    "you need to write the body of the request to an OpenAI chat completion call that: Uses model gpt-4o-mini Has a system message: Respond in JSON Has a user message: Generate 10 random addresses in the US Uses structured outputs to respond with an object addresses which is an array of objects with required fields: apartment (string) latitude (number) county (string) . Sets additionalProperties to false to prevent additional properties. Note that you don't need to run the request or use an API key; your task is simply to write the correct JSON body. What is the JSON body we should send to https://api.openai.com/v1/chat/completions for this? (No need to run it or to use an API key. Just write the body of the request below.)", 

    "Here is an example invoice image: Write just the JSON body (not the URL, nor headers) for the POST request that sends these two pieces of content (text and image URL) to the OpenAI API endpoint. Use gpt-4o-mini as the model. Send a single user message to the model that has a text and an image_url content (in that order). The text content should be Extract text from this image. Send the image_url as a base64 URL of the image above. CAREFUL: Do not modify the image. Write your JSON body here:",

    "Here are 2 verification messages: Dear user, please verify your transaction code 63468 sent to 24ds2000125@ds.study.iitm.ac.in Dear user, please verify your transaction code 82151 sent to 24ds2000125@ds.study.iitm.ac.in The goal is to capture this message, convert it into a meaningful embedding using OpenAI's text-embedding-3-small model, and subsequently use the embedding in a machine learning model to detect anomalies. Your task is to write the JSON body for a POST request that will be sent to the OpenAI API endpoint to obtain the text embedding for the 2 given personalized transaction verification messages above. This will be sent to the endpoint https://api.openai.com/v1/embeddings. Write your JSON body here:",

    "Your task is to write a Python function most_similar(embeddings) that will calculate the cosine similarity between each pair of these embeddings and return the pair that has the highest similarity. The result should be a tuple of the two phrases that are most similar. Write your Python code here:",

    "Your task is to build a FastAPI POST endpoint that accepts an array of docs and query string via a JSON body. The endpoint is structured as follows: POST /similarity. The JSON response might look like this: What is the API URL endpoint for your implementation? It might look like: http://127.0.0.1:8000/similarity",

    "Develop a FastAPI application that: Exposes a GET endpoint /execute?q=... where the query parameter q contains one of the pre-templatized questions. Analyzes the q parameter to identify which function should be called. Extracts the parameters from the question text. Returns a response in the following JSON format:",

    #GA 4
    "What is the total number of ducks across players on page number 8 of ESPN Cricinfo's ODI batting stats?",

    "Source: Utilize IMDb's advanced web search at https://www.imdb.com/search/title/ to access movie data.Filter: Filter all titles with a rating between 3 and 8. Format: For up to the first 25 titles, extract the necessary details: ID, title, year, and rating. The ID of the movie is the part of the URL after tt in the href attribute. For example, tt10078772. Organize the data into a JSON structure as follows: Submit: Submit the JSON data",
    
    "Write a web application that exposes an API with a single query parameter: ?country=. It should fetch the Wikipedia page of the country, extracts all headings (H1 to H6), and create a Markdown outline for the country. What is the URL of your API endpoint?",

    "you are tasked with developing a system that automates the following: API Integration and Data Retrieval: Use the BBC Weather API to fetch the weather forecast for New York. Send a GET request to the locator service to obtain the city's locationId. Include necessary query parameters such as API key, locale, filters, and search term (city). What is the JSON weather forecast description for New York?",

    "What is the maximum latitude of the bounding box of the city Addis Ababa in the country Ethiopia on the Nominatim API? Value of the maximum latitude",

    "What is the link to the latest Hacker News post mentioning Open Source having at least 30 points?",

    "Using the GitHub API, find all users located in the city Barcelona with over 70 followers. When was the newest user's GitHub profile created?",

    "Create a scheduled GitHub action that runs daily and adds a commit to your repository. Enter your repository URL (format: https://github.com/USER/REPO)",

    "What is the total Economics marks of students who scored 62 or more marks in Economics in groups 76-100 (including both groups)?",

    "What is the markdown content of the PDF, formatted with prettier@3.4.2?",

    #GA5
    "Download the Sales Excel file: What is the total margin for transactions before Tue Nov 01 2022 03:45:13 GMT+0530 (India Standard Time) for Epsilon sold in BR (which may be spelt in different ways)?",

    "Download the text file with student marks How many unique students are there in the file?",

    "What is the number of successful GET requests for pages under /kannada/ from 5:00 until before 14:00 on Sundays?",

    "Across all requests under hindimp3/ on 2024-05-23, how many bytes did the top IP address (by volume of downloads) download?",

    "How many units of Ball were sold in Shenzhen on transactions with at least 123 units?",

    "Download the data from What is the total sales value?",

    "Download the data from How many times does HB appear as a key?",

    "Write a DuckDB SQL query to find all posts IDs after 2024-12-16T06:01:02.983Z with at least 1 comment with 3 useful stars, sorted. The result should be a table with a single column called post_id, and the relevant post IDs should be sorted in ascending order. ",

    "What is the text of the transcript of this Mystery Story Audiobook between 35.8 and 213.1 seconds?",

    "Here is the image. It is a 500x500 pixel image that has been cut into 25 (5x5) pieces: Here is the mapping of each piece: Original Row        Original Column        Scrambled Row        Scrambled Column 2        1        0        0 1        1        0        1 4        1        0        2 0        3        0        3 0        1        0        4 1        4        1        0 2        0        1        1 2        4        1        2 4        2        1        3 2        2        1        4 0        0        2        0 3        2        2        1 4        3        2        2 3        0        2        3 3        4        2        4 1        0        3        0 2        3        3        1 3        3        3        2 4        4        3        3 0        2        3        4 3        1        4        0 1        2        4        1 1        3        4        2 0        4        4        3 4        0        4        4 Upload the reconstructed image by moving the pieces from the scrambled position to the original position:"
]

function_names = [
    # GA 1 
    "solve_vscode_question",              
    "solve_http_request_question",         
    "solve_prettier_hash_question",        
    "solve_google_sheets_question",        
    "solve_excel_question",               
    "solve_hidden_input_question",        
    "solve_count_wednesdays_question",    
    "solve_csv_extraction_question",      
    "solve_json_sorting_question",        
    "solve_json_conversion_question",      
    "solve_div_sum_question",              
    "solve_file_encoding_sum_question",   
    "solve_github_repo_question",         
    "solve_replace_text_question",         
    "solve_file_size_filter_question",
    "solve_rename_files_question",  
    "solve_compare_files_question", 
    "solve_sqlite_query_question", 
    # GA 2       
    "solve_markdown_documentation_question",  
    "solve_image_compression_question",
    "solve_github_pages_question",   
    "solve_colab_auth_question",    
    "solve_colab_brightness_question", 
    "solve_vercel_api_question",     
    "solve_github_action_question",
    "solve_docker_image_question",
    "solve_fastapi_server_question",
    "solve_llama_model_question",
    # GA 3
    "solve_llm_sentiment",
    "solve_token_cost",
    "solve_address_generation",
    "solve_llm_vision",
    "solve_llm_embedding",
    "solve_embedding_similarity",
    "solve_vector_databases",
    "solve_function_calling",
    # GA 4
    "solve_html_google_sheets",
    "solve_imdb",
    "solve_wiki_headings",
    "solve_weather_api",
    "solve_city_bounding_box",
    "solve_hacker_news",
    "solve_new_github_user",
    "solve_scheduled_github_action",
    "solve_extract_tables",
    "solve_pdf_to_md",
    # GA 5
    "solve_excel_sales_cleanup",
    "solve_student_marks_cleanup",
    "solve_log_requests",
    "solve_log_downloads",
    "solve_cleanup_sales",
    "solve_parse_partial_json",
    "solve_nested_jsonkeys",
    "solve_duckdb_socialmedia",
    "solve_transcribe_yt",
    "solve_image_reconstruction"
]

# --- STEP 2: Create and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(stored_questions)

# --- STEP 3: Save the vectorizer
with open("api/data/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# --- STEP 4: Save the question vectors as numpy array
np.save("api/data/question_vectors.npy", question_vectors.toarray())

# --- STEP 5: Save the function names in order
with open("api/data/function_names.json", "w") as f:
    json.dump(function_names, f)

print("✅ Preprocessing complete. Files saved to api/data/")
