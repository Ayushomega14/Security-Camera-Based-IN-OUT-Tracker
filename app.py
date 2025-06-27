import threading
import os
import cv2
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, Response, send_from_directory, jsonify
from main import generate_frames, play_video, toggle_playback, get_logs, camera_type
from global_video_stream import VideoStream

app = Flask(__name__)
app.config['SNAPSHOT_FOLDER'] = 'snapshots'

video_stream = None

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/start_feed', methods=['POST'])
def start_feed():
    global video_stream
    source_type = request.form['source']
    if source_type == 'ip':
        camera_url = request.form['ip_url']
        camera_type['source'] = 'ip'
        video_stream = VideoStream(camera_url)
    return redirect('/video_feed_page')

@app.route('/video_feed_page')
def video_feed_page():
    return render_template('video.html', play_status=play_video["status"])

@app.route('/video_feed')
def video_feed():
    if video_stream is None:
        return "Camera not initialized", 503
    return Response(generate_frames(video_stream), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    action = request.form['action']
    if action == 'toggle':
        toggle_playback()
    return redirect('/video_feed_page')

@app.route('/logs')
def view_logs():
    name = request.args.get('name', '').strip()
    date = request.args.get('date', '').strip()
    logs = get_logs(name_filter=name, date_filter=date)
    return render_template('logs.html', logs=logs)

@app.route('/live_logs')
def live_logs():
    today = datetime.now().strftime('%Y-%m-%d')
    logs = get_logs(date_filter=today)
    log_list = [
        {"timestamp": log[1], "name": log[2], "event": log[3]}
        for log in logs
    ]
    return jsonify({"logs": log_list})

@app.route('/snapshots')
def snapshots():
    name_filter = request.args.get('name_filter', '').strip().lower()
    date_filter = request.args.get('date_filter', '').strip()

    snapshot_list = []
    base_dir = app.config['SNAPSHOT_FOLDER']

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for date_folder in sorted(os.listdir(base_dir), reverse=True):
        date_path = os.path.join(base_dir, date_folder)
        if os.path.isdir(date_path) and (not date_filter or date_folder == date_filter):
            for person in os.listdir(date_path):
                if name_filter and name_filter not in person.lower():
                    continue
                person_path = os.path.join(date_path, person)
                for file in os.listdir(person_path):
                    if file.endswith('.jpg'):
                        full_path = os.path.join(person_path, file)
                        relative_path = os.path.relpath(full_path, 'static')
                        timestamp = f"{date_folder} {file.replace('.jpg', '').replace('-', ':')}"
                        snapshot_list.append({
                            "path": relative_path.replace('\\', '/'),
                            "name": person,
                            "timestamp": timestamp
                        })

    snapshot_list.sort(key=lambda x: x["timestamp"], reverse=True)
    return render_template('snapshot.html',
                           snapshots=snapshot_list,
                           name_filter=name_filter,
                           date_filter=date_filter)

@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    return send_from_directory(app.config['SNAPSHOT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
