import cv2
import threading
import time

class VideoStream:
    def __init__(self, src):
        self.src = src
        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.frame = None
        self.stopped = False

        if not self.capture.isOpened():
            print("❌ Failed to open video stream.")
            return

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            if not self.capture.isOpened():
                print("⏳ Reopening stream...")
                self.capture.open(self.src)
                time.sleep(0.5)
                continue

            ret, frame = self.capture.read()
            if ret:
                self.frame = frame
            else:
                print("⚠️ Failed to grab frame.")
                time.sleep(0.2)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.capture and self.capture.isOpened():
            self.capture.release()
