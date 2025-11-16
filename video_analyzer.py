import google.generativeai as genai
import os
import time
import pathlib
import yt_dlp
import traceback
import json
from langchain_tavily import TavilySearch
import tempfile
import uuid

# --- Configuration ---
# API Keys (GOOGLE_API_KEY, TAVILY_API_KEY) are read from
# environment variables by fastapi_app.py.
# ---

def download_video_from_url(url: str) -> str | None:
    """
    Downloads a video from a URL using yt-dlp with cookies to bypass bot detection.
    """
    try:
        # --- 1. Load cookies from Render secret file ---
        cookie_path = os.getenv("YOUTUBE_COOKIES_PATH")
        if not cookie_path or not pathlib.Path(cookie_path).exists():
            print("WARNING: YOUTUBE_COOKIES_PATH not set or file missing. Bot detection likely.")
            cookie_path = None
        else:
            print(f"Using cookies: {cookie_path}")

        # --- 2. Unique temp file ---
        temp_dir = tempfile.gettempdir()
        unique_filename = f"{uuid.uuid4()}.%(ext)s"
        output_template = str(pathlib.Path(temp_dir) / unique_filename)
        print(f"Download target: {output_template}")

        # --- 3. yt-dlp options ---
        ydl_opts = {
            'format': 'best[ext=mp4][height<=1080]/best[ext=mp4]/best',
            'outtmpl': output_template,
            'quiet': False,  # Set to True in prod if you don't want logs
            'noplaylist': True,
            'merge_output_format': 'mp4',
            'cookiefile': cookie_path,  # <-- CRITICAL: Bypass bot check
            'user_agent': (
                'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36'
            ),
            'extractor_args': {
                'youtube': {
                    'client': 'android',  # Helps, but cookies are required
                    'skip': ['dash', 'hls']  # Prefer direct MP4
                }
            },
            'sleep_interval': 1,
            'max_sleep_interval': 3,
        }

        # --- 4. Download ---
        downloaded_filepath = None
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Extracting info & downloading...")
            info = ydl.extract_info(url, download=True)
            downloaded_filepath = ydl.prepare_filename(info)

        # --- 5. Validate ---
        if not downloaded_filepath or not os.path.exists(downloaded_filepath):
            print("Download failed: file not found.")
            return None

        if os.path.getsize(downloaded_filepath) == 0:
            print("Downloaded file is empty.")
            os.remove(downloaded_filepath)
            return None

        print(f"Download SUCCESS: {downloaded_filepath}")
        return downloaded_filepath

    except yt_dlp.utils.DownloadError as e:
        if "Sign in to confirm" in str(e):
            print("BOT DETECTION: YouTube requires login. Update cookies.")
        else:
            print(f"yt-dlp DownloadError: {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Unexpected error in download: {e}")
        traceback.print_exc()
        return None

def extract_title(analysis_text: str, model: genai.GenerativeModel) -> str:
    """
    Uses the LLM to extract a clean movie/game title from the analysis text.
    """
    try:
        print("Extracting title from analysis...")
        prompt = (
            "From the following text, extract only the movie or game title. "
            "If there is a year, include it. "
            "Respond with *nothing but* the title and year (if present). "
            f"For example, from 'The movie clip is from **Fury** (2014), starring...' "
            "you should respond with 'Fury (2014)'.\n\n"
            f"Text: \"{analysis_text}\""
        )
        
        response = model.generate_content(
            prompt,
            request_options={"timeout": 60}
        )
        title = response.text.strip().strip('*"')
        return title
    except Exception as e:
        print(f"Error during title extraction: {e}")
        return "Unknown Title"

def search_for_title(title: str) -> dict:
    """
    Uses Tavily to search for legal streaming options for the extracted title.
    Returns a dictionary with search status and results.
    """
    import traceback
    
    print(f"Searching for streaming options for '{title}'...")
    
    if "TAVILY_API_KEY" not in os.environ:
        error_msg = "TAVILY_API_KEY environment variable not set. Cannot perform search."
        print(f"\nError: {error_msg}")
        return {"status": "error", "message": error_msg}

    try:
        tavily_search = TavilySearch(max_results=5)
        query = f"where to legally watch {title}"
        data = tavily_search.invoke({"query": query})
        search_docs = data.get("results", [])
        
        if not search_docs:
            print(f"No search results found for '{query}'.")
            return {"status": "no_results", "message": f"No search results found for '{query}'."}

        print("Search successful.")
        return {"status": "success", "results": search_docs}

    except Exception as e:
        error_msg = f"An unexpected error occurred during Tavily search: {e}"
        print(error_msg)
        traceback.print_exc()
        return {"status": "error", "message": error_msg}

def analyze_video(video_file_path: str, prompt: str) -> dict:
    """
    Analyzes a local video file using the Gemini API.
    Returns a dictionary containing the analysis, title, and search results.
    """
    import traceback
    
    video_file_name = None
    
    if "GOOGLE_API_KEY" not in os.environ:
        error_msg = "Error: GOOGLE_API_KEY environment variable not set."
        print(error_msg)
        return {"status": "error", "message": error_msg}

    try:
        print("Configuring Generative AI client...")
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

        if not pathlib.Path(video_file_path).exists():
            error_msg = f"Error: Video file not found at path: {video_file_path}"
            print(error_msg)
            return {"status": "error", "message": error_msg}
        
        file_size = os.path.getsize(video_file_path)
        if file_size == 0:
            error_msg = f"Error: Video file is empty: {video_file_path}"
            print(error_msg)
            return {"status": "error", "message": error_msg}
        print(f"File size: {file_size / (1024*1024):.2f} MB")

        print(f"Uploading file: {video_file_path}...")
        video_file = genai.upload_file(path=video_file_path)
        video_file_name = video_file.name

        print(f"File uploaded: {video_file.name}. Waiting for processing...")
        while video_file.state.name == "PROCESSING":
            print("  Waiting for file processing...")
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            error_msg = "Error: File upload failed. State: FAILED"
            print(error_msg)
            return {"status": "error", "message": error_msg}

        if video_file.state.name != "ACTIVE":
            error_msg = f"Error: File is not active. State: {video_file.state.name}"
            print(error_msg)
            return {"status": "error", "message": error_msg}

        print("File processed and active.")
        print("Sending request to Gemini API...")

        model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-09-2025")

        response = model.generate_content(
            [prompt, video_file],
            request_options={"timeout": 600} 
        )

        print("Analysis complete.")
        analysis_text = response.text

        title = extract_title(analysis_text, model)
        search_data = {}
        
        if title and title != "Unknown Title":
            print(f"Extracted Title: {title}")
            search_data = search_for_title(title)
        else:
            print("Could not extract a usable title.")
            search_data = {"status": "skipped", "message": "Unknown title"}
        
        return {
            "status": "success",
            "analysis": analysis_text,
            "extracted_title": title,
            "search_info": search_data
        }

    except Exception as e:
        error_msg = f"An unexpected error occurred during analysis: {e}"
        print(error_msg)
        traceback.print_exc()
        return {"status": "error", "message": error_msg}
    
    finally:
        if video_file_name:
            try:
                print(f"Cleaning up uploaded remote file: {video_file_name}...")
                genai.delete_file(video_file_name)
                print("Remote cleanup complete.")
            except Exception as e:
                print(f"Error during remote file cleanup: {e}")

