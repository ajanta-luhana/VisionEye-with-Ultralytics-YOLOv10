# VisionEye with Ultralytics YOLOv10

VisionEye with Ultralytics YOLOv10, achieving a new level of precision in
real-time object detection and classification. With the powerful
advancements in YOLOv10, the system now delivers faster inference, higher
accuracy, and enhanced multi-object performance.

This repo has two ways to use it:

- **`ultra.py`** — standalone script. Draw zones interactively on your
  desktop (mouse clicks on a popup window), then it processes a local video
  file and writes an annotated output.
- **`web_app/`** — browser version of the same idea. Upload a video, draw
  zones on a web page, and download the processed result. No desktop GUI
  needed, and it can be deployed to a server.

## Option 1: Run the script directly (`ultra.py`)

```bash
pip install ultralytics opencv-python numpy
python ultra.py
```

Edit the `VIDEO_INPUT` path in `ultra.py` to point to your own video file.
A window will open on the first frame — left-click to add zone points,
right-click to close a zone (3+ points), press Enter to start processing,
and Q to stop early. Output goes to `output_interactive/`.

## Option 2: Run the web app (`web_app/`)

```bash
cd web_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`, upload a video, click to draw zones, and hit
"Process Video". See `web_app/README.md` for full details, including how to
deploy it (VPS or Hugging Face Spaces with the included `Dockerfile`).

## Requirements

- Python 3.10+
- See `web_app/requirements.txt` for the web app's dependencies
  (Flask, OpenCV, NumPy, Ultralytics)

## License

See [LICENSE](LICENSE).
