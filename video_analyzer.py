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
import shutil

# --- Configuration ---
# API Keys (GOOGLE_API_KEY, TAVILY_API_KEY, YOUTUBE_COOKIES) are read from
# environment variables.
# ---

def download_video_from_url(url: str) -> str | None:
    """
    Downloads a video from a URL to a *unique temporary file*.
    Adapts strategy based on whether FFmpeg is available.
    """
    import pathlib
    import traceback
    
    cookie_file_path = None
    
    try:
        temp_dir = tempfile.gettempdir()
        unique_filename_base = str(uuid.uuid4())
        
        # Check if FFmpeg is installed
        ffmpeg_available = shutil.which('ffmpeg') is not None
        print(f"  FFmpeg available: {ffmpeg_available}")

        # --- Cookies Handling ---
        cookie_args = {}
        cookies_content = os.environ.get("YOUTUBE_COOKIES")
        if cookies_content:
            print("  Found YOUTUBE_COOKIES environment variable. Creating cookie file...")
            # delete=False is important so we can close it and let yt-dlp open it
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as cookie_file:
                cookie_file.write(cookies_content)
                cookie_file_path = cookie_file.name
            cookie_args['cookiefile'] = cookie_file_path
            print("  Cookies configured.")
        else:
            print("  WARNING: No YOUTUBE_COOKIES found.")
        # ------------------------

        # --- STRATEGY 1: Determine Options ---
        if ffmpeg_available:
            print("  [Strategy] High Quality (FFmpeg detected).")
            # With FFmpeg, we can download best video + best audio and merge them.
            output_template = str(pathlib.Path(temp_dir) / f"{unique_filename_base}.%(ext)s")
            ydl_opts = {
                'format': 'bestvideo+bestaudio/best',
                'merge_output_format': 'mp4',
                'outtmpl': output_template,
                'quiet': True,
                'noplaylist': True,
                'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
                **cookie_args
            }
        else:
            print("  [Strategy] Safe Mode (No FFmpeg). Relaxing format constraints.")
            # Without FFmpeg, we CANNOT merge. We must find a single file.
            # We remove [ext=mp4] to allow webm/mkv if that's all that exists.
            output_template = str(pathlib.Path(temp_dir) / f"{unique_filename_base}.%(ext)s")
            ydl_opts = {
                'format': 'best', # Just get the best single file with video+audio
                'outtmpl': output_template,
                'quiet': True,
                'noplaylist': True,
                **cookie_args
            }

        # --- DOWNLOAD ATTEMPT ---
        downloaded_filepath = None
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print("  Fetching info and downloading...")
                info = ydl.extract_info(url, download=True)
                
                if ffmpeg_available:
                    # If we merged, it should be .mp4
                    expected = str(pathlib.Path(temp_dir) / f"{unique_filename_base}.mp4")
                    if os.path.exists(expected):
                        downloaded_filepath = expected
                    else:
                        downloaded_filepath = ydl.prepare_filename(info)
                else:
                    # If we didn't merge, we trust prepare_filename, but check for existence
                    temp_path = ydl.prepare_filename(info)
                    if os.path.exists(temp_path):
                        downloaded_filepath = temp_path
                    else:
                        # Fallback search for any file with our UUID
                        print("  Primary path not found, searching for download...")
                        for file in pathlib.Path(temp_dir).glob(f"{unique_filename_base}.*"):
                            downloaded_filepath = str(file)
                            break

        except Exception as first_error:
            print(f"  [Warning] First download attempt failed: {first_error}")
            # --- STRATEGY 2: Last Resort (If first attempt failed) ---
            if not ffmpeg_available:
                print("  [Strategy] Last Resort. Trying 'worst' quality to ensure success.")
                ydl_opts['format'] = 'worst' # Often guarantees a simple single file (360p)
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        downloaded_filepath = ydl.prepare_filename(info)
                except Exception as second_error:
                    print(f"  [Error] Last resort failed: {second_error}")
                    raise second_error # Re-raise if even this fails

        # Final Validation
        if not downloaded_filepath or not os.path.exists(downloaded_filepath) or os.path.getsize(downloaded_filepath) == 0:
            print("Error: Download failed, file is empty or does not exist.")
            if downloaded_filepath and os.path.exists(downloaded_filepath):
                os.remove(downloaded_filepath) 
            return None

        print(f"  Download complete: {downloaded_filepath}")
        return downloaded_filepath
    
    except Exception as e:
        print(f"Error downloading video from URL: {e}")
        traceback.print_exc()
        return None
    
    finally:
        # Clean up the temporary cookie file
        if cookie_file_path and os.path.exists(cookie_file_path):
            try:
                os.remove(cookie_file_path)
            except Exception:
                pass

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
