import os
import sys
import cv2
from root import Ui_MainWindow
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from threading import Thread
from PyQt6.QtGui import QIcon
import numpy as np
from PIL import Image, ImageTk
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
import requests
import time
from flask_socketio import SocketIO
from flask import Flask, Response, render_template
import random
import socket
from ultralytics import YOLO
import cv2
import math

classnames = ["fire"]


class VideoThread(QThread):
    frame_signal = pyqtSignal(QImage)
    fire_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()        
        self.model = YOLO("fire.pt") 

    def run(self):
        cap = cv2.VideoCapture("fire2.mp4")
        if not cap.isOpened():
            print("Can't open the camera")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            result = self.model(frame, stream=True)

            # Process the detection results
            for info in result:
                boxes = info.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    Class = int(box.cls[0])
                    if confidence > 50:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        state=classnames[Class]
                        conf=confidence
                        # f"{classnames[Class]} {confidence}%"
                        cv2.putText(
                            frame,
                            state+conf,
                            (x1 + 8, y1 + 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

            # Convert the frame to QImage
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            q_image = QImage(
                rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )           

            # Display the QImage on the GUI
            pixmap = QPixmap.fromImage(q_image)

            # Create QImage from QPixmap
            image = pixmap.toImage()
            self.frame_signal.emit(image)
            self.fire_signal.emit(state)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


class Home(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.video_thread = VideoThread()
        self.video_thread.frame_signal.connect(self.update_frame)
        self.video_thread.fire_signal.connect(self.show_state)

        self.show_state_thread()

    def start_video_thread(self):
        self.video_thread.start()

    def update_frame(self, image):
        self.cam.setPixmap(QPixmap.fromImage(image))    

#start thread to update frames
    def update_frame_thread(self):
        pl = Thread(target=self.update_frame)
        pl.start()

    def show_state(self, state):
        print(state)

    def show_state_thread(self):
        st = Thread(target=self.show_state)
        st.start()


    def closeEvent(self, event):
        self.video_thread.cap.release()
        event.accept()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Home()
    MainWindow.showMaximized()

    classnames = ["fire"]

    MainWindow.start_video_thread()
    sys.exit(app.exec())
