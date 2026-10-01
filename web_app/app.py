import os
import math
import uuid
from collections import defaultdict

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXT = {"mp4", "avi", "mov", "mkv"}
# yolov10n is small/fast for testing; swap for yolov10x.pt for best accuracy.
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov10n.pt")
CONFIDENCE = 0.3

COLORS = {
    'car': (0, 255, 255),
    'truck': (255, 0, 255),
    'motorcycle': (0, 255, 0),
    'person': (0, 165, 255),
    'bus': (255, 200, 0),
    'bicycle': (0, 255, 255),
}
ZONE_COLORS = [
    (0, 100, 255),
    (255, 0, 200),
    (0, 255, 0),
    (255, 200, 0),
    (255, 0, 100),
]

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(YOLO_MODEL)
    return _model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    xinters = None
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or (xinters is not None and x <= xinters):
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def draw_zone_overlay(img, zone, frame_num):
    points = np.array(zone["points"])
    color = zone["color"]
    overlay = img.copy()
    cv2.fillPoly(overlay, [points], color)
    alpha = 0.25 + 0.1 * math.sin(frame_num * 0.05)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    thickness = int(3 + 2 * abs(math.sin(frame_num * 0.1)))
    cv2.polylines(img, [points], True, (255, 255, 255), thickness + 2, lineType=cv2.LINE_AA)
    cv2.polylines(img, [points], True, color, thickness, lineType=cv2.LINE_AA)
    cx = int(np.mean([p[0] for p in zone["points"]]))
    cy = int(np.mean([p[1] for p in zone["points"]]))
    text = zone["name"]
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(img, (cx - ts[0] // 2 - 8, cy - ts[1] // 2 - 8),
                  (cx + ts[0] // 2 + 8, cy + ts[1] // 2 + 8), (0, 0, 0), -1)
    cv2.rectangle(img, (cx - ts[0] // 2 - 8, cy - ts[1] // 2 - 8),
                  (cx + ts[0] // 2 + 8, cy + ts[1] // 2 + 8), color, 2)
    cv2.putText(img, text, (cx - ts[0] // 2, cy + ts[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)


def draw_gradient_line(img, pt1, pt2, color1, color2, thickness=2):
    x1, y1 = pt1
    x2, y2 = pt2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    steps = int(distance)
    if steps == 0:
        return
    for i in range(steps):
        a = i / steps
        x = int(x1 + (x2 - x1) * a)
        y = int(y1 + (y2 - y1) * a)
        b = int(color1[0] + (color2[0] - color1[0]) * a)
        g = int(color1[1] + (color2[1] - color1[1]) * a)
        r = int(color1[2] + (color2[2] - color1[2]) * a)
        cv2.circle(img, (x, y), thickness, (b, g, r), -1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Please upload an mp4/avi/mov/mkv file"}), 400

    video_id = uuid.uuid4().hex[:12]
    ext = file.filename.rsplit(".", 1)[1].lower()
    video_path = os.path.join(UPLOAD_DIR, f"{video_id}.{ext}")
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return jsonify({"error": "Could not read a frame from that video"}), 400

    frame_filename = f"{video_id}.jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, frame_filename), frame)
    height, width = frame.shape[:2]

    return jsonify({
        "video_id": video_id,
        "frame_url": url_for("uploaded_file", filename=frame_filename),
        "width": width,
        "height": height,
    })


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/outputs/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json(force=True)
    video_id = data.get("video_id")
    raw_zones = data.get("zones", [])
    if not video_id or not raw_zones:
        return jsonify({"error": "Missing video_id or zones"}), 400

    video_path = None
    for ext in ALLOWED_EXT:
        candidate = os.path.join(UPLOAD_DIR, f"{video_id}.{ext}")
        if os.path.exists(candidate):
            video_path = candidate
            break
    if video_path is None:
        return jsonify({"error": "Video not found, please re-upload"}), 404

    zones = []
    for i, z in enumerate(raw_zones):
        zones.append({
            "name": z.get("name") or f"ZONE {i + 1}",
            "points": [(int(p["x"]), int(p["y"])) for p in z["points"]],
            "color": ZONE_COLORS[i % len(ZONE_COLORS)],
        })

    model = get_model()
    names = model.model.names

    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS) or 25
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_filename = f"{video_id}_zones.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    center_point = (width // 2, height // 2)
    frame_count = 0
    zone_totals = {zone["name"]: defaultdict(int) for zone in zones}
    grand_total = 0

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame_count += 1

        results = model.predict(frame, conf=CONFIDENCE, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for zone in zones:
            draw_zone_overlay(frame, zone, frame_count)

        pulse = int(8 + 4 * abs(math.sin(frame_count * 0.1)))
        cv2.circle(frame, center_point, pulse, (255, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, center_point, 6, (255, 255, 0), -1, lineType=cv2.LINE_AA)

        for box, cls in zip(boxes, classes):
            obj_class = names[int(cls)]
            if obj_class not in COLORS:
                continue
            x1, y1, x2, y2 = map(int, box)
            box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            matched_zone = None
            for zone in zones:
                if point_in_polygon(box_center, zone["points"]):
                    matched_zone = zone
                    break
            if matched_zone is None:
                continue

            zone_totals[matched_zone["name"]][obj_class] += 1
            grand_total += 1

            color = COLORS[obj_class]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            label = obj_class.upper()
            ls = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - ls[1] - 6), (x1 + ls[0] + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, lineType=cv2.LINE_AA)
            draw_gradient_line(frame, center_point, box_center, (255, 255, 0), color, 2)
            cv2.circle(frame, box_center, 5, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, box_center, 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        cv2.putText(frame, f"Frame: {frame_count:05d}", (15, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        writer.write(frame)

    cam.release()
    writer.release()

    summary = {name: dict(counts) for name, counts in zone_totals.items()}
    return jsonify({
        "output_url": url_for("output_file", filename=output_filename),
        "frames_processed": frame_count,
        "grand_total": grand_total,
        "zone_summary": summary,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
