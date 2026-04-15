# api.py - REST API wrapper around the detection pipeline

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, uuid, os
from main import run_pipeline
from config import CONFIG

app = FastAPI(title="Smart Vision Tracker API", version="2.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

jobs = {}   # job_id -> status dict

def process_job(job_id: str, video_path: str):
    jobs[job_id]["status"] = "processing"
    try:
        mp4_path, summary = run_pipeline(video_path)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["summary"] = summary
        jobs[job_id]["output"] = mp4_path
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    job_id = str(uuid.uuid4())[:8]
    save_path = os.path.join("temp", f"{job_id}_{file.filename}")
    os.makedirs("temp", exist_ok=True)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    CONFIG["input_video"] = save_path
    jobs[job_id] = {"status": "queued", "filename": file.filename}
    background_tasks.add_task(process_job, job_id, save_path)
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return jobs[job_id]

@app.get("/download/{job_id}")
def download_result(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return JSONResponse({"error": "Not ready"}, status_code=400)
    return FileResponse(job["output"], media_type="video/mp4",
                        filename="tracked_output.mp4")

@app.get("/analytics/{job_id}")
def get_analytics(job_id: str):
    job = jobs.get(job_id)
    if not job or "summary" not in job:
        return JSONResponse({"error": "No analytics yet"}, status_code=400)
    return job["summary"]

@app.get("/")
def home():
    return {"message": "API is running"}
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000