# VisionEye Zone Monitor — Web App

Browser version of `ultra.py`: upload a video, draw zones by clicking on the
canvas, then run YOLOv10 detection and download the annotated result.

## Setup

```bash
cd web_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The default model is `yolov10n.pt` (fast, good for testing). Ultralytics
downloads it automatically on first run. To use a different model:

```bash
export YOLO_MODEL=yolov10x.pt   # Windows: set YOLO_MODEL=yolov10x.pt
```

## Run

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## How it works

1. **Upload** a video (mp4/avi/mov/mkv). The server grabs the first frame.
2. **Draw zones**: click to add points, double-click (or "Finish Zone") to
   close a polygon with 3+ points. Draw as many zones as you like.
3. **Process Video**: runs YOLOv10 frame-by-frame, draws boxes/zones, and
   counts objects per zone. This happens synchronously in the request, so a
   long video will take a while — the page will just wait.
4. Preview and download the annotated `.mp4`, plus a per-zone count summary.

## Notes / limitations

- Processing is synchronous and single-request. For long videos or a
  multi-user deployment, this should become a background job (e.g. Celery
  or an RQ queue) with a polling/progress endpoint instead.
- No authentication, upload size limits, or file cleanup — fine for local
  personal use, not production-hardened.
- Uploaded videos and outputs are stored under `uploads/` and `outputs/`;
  clear them periodically.

## Deploying beyond localhost

This needs real compute (Python + PyTorch + OpenCV), so it can't be hosted
as a static page. Reasonable options:
- A small VPS or cloud VM (DigitalOcean, EC2, etc.) running this Flask app
  behind gunicorn + nginx.
- A container platform (Render, Railway, Fly.io) that supports persistent
  background workers, since detection is compute-heavy.
- If you expect real traffic, move `/process` to a background task queue
  and add a job-status endpoint instead of blocking the HTTP request.
