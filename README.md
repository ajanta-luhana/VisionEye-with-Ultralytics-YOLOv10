# VisionEye — YOLOv10 Zone-Based Video Analytics

VisionEye is an AI-powered video analytics project built with Python, Ultralytics YOLOv10, OpenCV, and Flask.

It enables users to analyze traffic or surveillance videos by defining custom polygon zones and detecting objects within those selected areas. The system processes video frames using YOLOv10 and generates an annotated output video with bounding boxes, class labels, zone overlays, frame information, and zone-wise detection counts.

Supported detection classes include:

- Car
- Truck
- Motorcycle
- Bus
- Bicycle
- Person

## Features

- YOLOv10-based object detection for video analytics
- Interactive custom polygon zone selection
- Video upload support for MP4, AVI, MOV, and MKV files
- Zone-based detection and counting of vehicles and persons
- Annotated output video generation
- Bounding boxes and class labels for detected objects
- Colored zone overlays and frame-level visualization
- Zone-wise detection summary
- Two execution modes: desktop script and browser-based web app

## Project Modes

This repository provides two ways to use VisionEye.

### Option 1: Desktop Interactive Script

`ultra.py` is a standalone script for local desktop usage.

It allows users to select polygon zones interactively on the first frame of a local video using mouse clicks. The script then processes the video and saves an annotated output.

#### Run the desktop script

```bash
git clone [https://github.com/ajanta-luhana/VisionEye-with-Ultralytics-YOLOv10.git](https://github.com/ajanta-luhana/VisionEye-with-Ultralytics-YOLOv10.git)
cd VisionEye-with-Ultralytics-YOLOv10

pip install -r requirements.txt

python ultra.py
```

Before running the script, update the `VIDEO_INPUT` path in `ultra.py` to point to your own video file.

### Desktop Controls

- Left-click: Add a zone point
- Right-click: Close the current zone after selecting at least 3 points
- Enter: Start video processing
- Q: Stop processing early

The annotated output video is saved in:

```text
output_interactive/
```

### Option 2: Browser-Based Web Application

The `web_app/` folder contains a Flask-based browser interface.

Users can upload a traffic or surveillance video, define custom polygon zones directly on the extracted video frame, process the video, and download the annotated result.

#### Run the web application

```bash
git clone [https://github.com/ajanta-luhana/VisionEye-with-Ultralytics-YOLOv10.git](https://github.com/ajanta-luhana/VisionEye-with-Ultralytics-YOLOv10.git)
cd VisionEye-with-Ultralytics-YOLOv10

pip install -r requirements.txt

python web_app/app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

### Web App Workflow

1. Upload an MP4, AVI, MOV, or MKV video.
2. The application extracts and displays the first frame.
3. Draw one or more polygon zones on the video frame.
4. Submit the zones for processing.
5. YOLOv10 detects supported objects frame by frame.
6. The system records detections whose center points lie inside a selected zone.
7. Download the generated annotated video and review the zone-wise summary.

## Technology Stack

- Python
- Ultralytics YOLOv10
- Flask
- OpenCV
- NumPy
- HTML, CSS, JavaScript
- GitHub

## Installation Requirements

- Python 3.10 or newer
- A supported input video: MP4, AVI, MOV, or MKV
- Internet connection on first run if YOLOv10 model weights need to be downloaded

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
VisionEye-with-Ultralytics-YOLOv10/
├── README.md
├── requirements.txt
├── ultra.py
├── output_interactive/
└── web_app/
    ├── app.py
    ├── templates/
    │   └── index.html
    ├── uploads/
    └── outputs/
```

## Output

VisionEye generates an annotated video that can include:

- Custom polygon zone overlays
- Bounding boxes around detected objects
- Object class labels
- Object-center visualization
- Frame numbers
- Zone-wise detection counts for supported objects

> Note: The current implementation counts detections per processed frame. Future improvements can add multi-object tracking with unique IDs to prevent the same object from being counted repeatedly across consecutive frames.

## Future Improvements

- Unique object tracking with persistent IDs
- Line-crossing vehicle counting
- Live webcam and RTSP stream support
- Traffic dashboard with charts and downloadable reports
- Database integration for analytics history
- Cloud storage for uploaded and processed videos
- Improved deployment configuration for cloud hosting

## Author

Ajanta Luhana

## License

See the [LICENSE](LICENSE) file for details.
