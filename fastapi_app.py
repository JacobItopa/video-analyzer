import os
import traceback
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, HttpUrl
import uvicorn
import logging
from dotenv import load_dotenv  # <-- 1. IMPORT THIS

# --- Load Environment Variables ---
load_dotenv()  # <-- 2. CALL THIS FUNCTION

# Import the logic from your video_analyzer.py file
from video_analyzer import download_video_from_url, analyze_video

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment Variable Check ---
# Check for API keys at startup
if not os.environ.get("GOOGLE_API_KEY"):
    logger.error("FATAL ERROR: GOOGLE_API_KEY environment variable not set.")
    # In a real app, you might exit(1) here
    # For simplicity, we'll let it fail on first request
if not os.environ.get("TAVILY_API_KEY"):
    logger.warning("WARNING: TAVILY_API_KEY not set. Search functionality will fail.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Video Analyzer API",
    description="Analyzes a video from a URL, identifies the content, and finds streaming links.",
    version="1.0.0",
)

# --- Request & Response Models ---
class VideoRequest(BaseModel):
    """Pydantic model for the incoming request body."""
    url: HttpUrl # Use HttpUrl for built-in URL validation
    prompt: str = "tell me about this video and what game or movie it is"

class ErrorResponse(BaseModel):
    """Pydantic model for error responses."""
    detail: str

# --- API Endpoint ---
@app.post(
    "/analyze-video",
    summary="Analyze a video from a URL",
    description="Downloads a video from a URL, analyzes it with Gemini to find the title, and searches Tavily for streaming options.",
    response_model=dict, # Return the raw JSON dict from your analyzer
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input or download failure"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)
async def analyze_video_endpoint(request: VideoRequest):
    """
    The main API endpoint to analyze a video.
    
    This endpoint runs the blocking I/O (downloading, uploading, AI analysis)
    in a separate thread pool to avoid blocking the server's event loop.
    """
    local_video_path = None
    try:
        logger.info(f"Received request for URL: {request.url}")
        
        # 1. Download Video
        # Run the blocking download function in a thread pool
        logger.info("Downloading video...")
        local_video_path = await run_in_threadpool(
            download_video_from_url, str(request.url)
        )
        
        if not local_video_path:
            logger.warning(f"Failed to download video from {request.url}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download video from URL: {request.url}. The URL may be invalid or unsupported.",
            )
        
        logger.info(f"Video downloaded to {local_video_path}")

        # 2. Analyze Video
        # Run the blocking analysis function in a thread pool
        logger.info("Analyzing video...")
        analysis_result = await run_in_threadpool(
            analyze_video, local_video_path, request.prompt
        )
        
        if analysis_result.get("status") == "error":
            logger.error(f"Analysis failed: {analysis_result.get('message')}")
            # Pass the error from the analyzer to the client
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis Error: {analysis_result.get('message')}",
            )

        logger.info("Analysis complete.")
        
        # 3. Return Result
        # The result from analyze_video is already a dictionary
        return analysis_result

    except HTTPException as http_exc:
        # Re-raise known HTTP exceptions
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error for {request.url}: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {e}",
        )
    finally:
        # 4. CRITICAL: Clean up the local temporary file
        if local_video_path and os.path.exists(local_video_path):
            try:
                logger.info(f"Cleaning up local file: {local_video_path}")
                os.remove(local_video_path)
            except Exception as e:
                # Log cleanup error but don't crash the request
                logger.error(f"Failed to clean up file {local_video_path}: {e}")

# --- Main entry point to run the server ---
# The if __name__ == "__main__": block has been removed.
# Gunicorn will start the server directly by referencing the 'app' object.
# The command is: gunicorn -w 4 -k uvicorn.workers.UvicornWorker fastapi_app:app