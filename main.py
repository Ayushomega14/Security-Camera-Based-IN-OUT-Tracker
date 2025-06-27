# main.py

import os
import cv2
import face_recognition
from datetime import datetime, timedelta
from ultralytics import YOLO
import sqlite3

play_video = {"status": True}
camera_type = {"source": None}
snapshot_dir = "snapshots"

def toggle_playback():
    play_video["status"] = not play_video["status"]

def create_db():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        name TEXT,
        event TEXT)''')
    conn.commit()
    conn.close()

def log_event_sql(name, event):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("INSERT INTO logs (timestamp, name, event) VALUES (?, ?, ?)", (timestamp, name, event))
    conn.commit()
    conn.close()
    print(f"[{timestamp}] {event}: {name}")

def get_logs(name_filter='', date_filter=''):
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    query = "SELECT * FROM logs WHERE 1=1"
    params = []

    if name_filter:
        query += " AND name LIKE ?"
        params.append(f"%{name_filter}%")
    if date_filter:
        query += " AND DATE(timestamp) = ?"
        params.append(date_filter)

    query += " ORDER BY timestamp DESC"
    c.execute(query, params)
    logs = c.fetchall()
    conn.close()
    return logs

def load_known_faces(face_dir):
    known_face_encodings = []
    known_face_names = []
    for person_name in os.listdir(face_dir):
        person_folder = os.path.join(face_dir, person_name)
        if os.path.isdir(person_folder):
            for file in os.listdir(person_folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_folder, file)
                    image = face_recognition.load_image_file(img_path)
                    face_locations = face_recognition.face_locations(image)
                    if face_locations:
                        encodings = face_recognition.face_encodings(image, face_locations)
                        if encodings:
                            known_face_encodings.append(encodings[0])
                            known_face_names.append(person_name)
    return known_face_encodings, known_face_names

def ensure_snapshot_dir(date_str, name):
    folder_path = os.path.join(snapshot_dir, date_str, name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def generate_frames(video_stream):
    create_db()
    yolo = YOLO("yolov8n-face.pt")
    known_face_encodings, known_face_names = load_known_faces("face_db")
    in_logged = set()
    last_seen_time = {}
    snapshot_taken = set()
    absence_threshold = timedelta(seconds=5)

    while True:
        if not play_video["status"]:
            cv2.waitKey(30)
            continue

        frame = video_stream.read()
        if frame is None:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo(frame, conf=0.4, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []

        current_faces = set()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            top, right, bottom, left = y1, x2, y2, x1
            face_location = (top, right, bottom, left)

            encodings = face_recognition.face_encodings(rgb_frame, [face_location])
            if encodings:
                face_encoding = encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
                if True in matches:
                    idx = matches.index(True)
                    name = known_face_names[idx]
                    current_faces.add(name)

                    if name not in in_logged:
                        log_event_sql(name, "IN")
                        in_logged.add(name)
                        snapshot_key = f"{datetime.now().date()}_{name}"
                        if snapshot_key not in snapshot_taken:
                            date_str = datetime.now().strftime("%Y-%m-%d")
                            folder = ensure_snapshot_dir(date_str, name)
                            snapshot_path = os.path.join(folder, f"{datetime.now().strftime('%H-%M-%S')}.jpg")
                            cv2.imwrite(snapshot_path, frame)
                            snapshot_taken.add(snapshot_key)

                    last_seen_time[name] = datetime.now()

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        for name in list(in_logged):
            if name not in current_faces:
                last_time = last_seen_time.get(name)
                if last_time and datetime.now() - last_time > absence_threshold:
                    log_event_sql(name, "OUT")
                    in_logged.remove(name)
                    last_seen_time.pop(name, None)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
